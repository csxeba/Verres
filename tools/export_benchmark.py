"""Export a trained OD run to ONNX and benchmark Keras / ONNX Runtime / OpenVINO batch-1 CPU latency.

Usage:
    python tools/export_benchmark.py config/exp_od_small_fcnn.yml artifactory/od/small_fcnn/xp_... \\
        [--out-dir DIR] [--threads 1 4 8] [--iters 300] [--warmup 50] [--image PATH]
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# verres first: it pins KERAS_BACKEND=tensorflow before keras is imported
import verres as V
from verres.architecture import VRSArchitecture
from verres.data.codec import Codec
from verres.data.dataset.cocodoom import _generate_annotation_name
from verres.execution import export

import cv2
import tensorflow as tf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASS_MAPPING = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
SCORE_THRESHOLD = 0.3
HEADS = ("heatmap", "refinement", "box_wh")


def timeit(fn, iters: int, warmup: int):
    """p50/p95 wall-clock ms of one batch-1 call; wall clock includes the runtime's Python overhead."""
    for _ in range(warmup):
        fn()
    samples = np.empty(iters)
    for i in range(iters):
        start = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - start) * 1e3
    return float(np.percentile(samples, 50)), float(np.percentile(samples, 95))


def build(config_path: str):
    """Architecture and decode codec from the experiment YAML; weights are loaded separately."""
    config = V.Config.from_paths(config_path, CLASS_MAPPING)
    config.model.backbone_spec["weights"] = None  # the run's own weights are loaded next; no ImageNet download
    model = VRSArchitecture.factory(config)
    codec = Codec(config, config.training.data[0].transformations)  # the decode chain lives in the training block
    return config, model, codec


def serving(model):
    """Raw [0, 255] image in, head logits out: the graph the SavedModel and the ONNX file both hold."""
    @tf.function
    def forward(image):
        return model(model.preprocess_input(image), training=False)
    return forward


def load_image(config, path: str = None):
    """Batch of one raw [0, 255] float image; the file is resized the way the `image` transformation does."""
    height, width = config.model.input_shape_hw
    if path is None:
        return np.random.default_rng(0).integers(0, 256, (1, height, width, 3)).astype(np.float32)
    image = cv2.resize(cv2.imread(path), (width, height), interpolation=cv2.INTER_LINEAR)
    return image[None].astype(np.float32)


def to_onnx(saved_model_dir, onnx_path: str) -> str:
    """tf2onnx over the fixed-shape serving signature; a separate process keeps the converter's logs off stdout."""
    subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(saved_model_dir),
                    "--opset", "17", "--output", str(onnx_path)], check=True)
    return str(onnx_path)


def ort_runner(onnx_path: str, threads: int):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    names = [output.name for output in session.get_outputs()]

    def run(image):
        return dict(zip(names, session.run(None, {input_name: image})))
    return run


def ov_runner(onnx_path: str, threads: int):
    import openvino as ov
    compiled = ov.Core().compile_model(str(onnx_path), "CPU", {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_NUM_THREADS": threads,
    })
    input_name = compiled.input(0).get_any_name()
    ports = [compiled.output(i) for i in range(len(compiled.outputs))]
    names = [port.get_any_name() for port in ports]

    def run(image):
        result = compiled({input_name: image})
        return {name: result[port] for name, port in zip(names, ports)}
    return run


def decode_heads(codec: Codec, heads: dict):
    """The project's own decode, so every runtime is scored on identical postprocessing."""
    return codec.decode({name: np.asarray(tensor)[0] for name, tensor in heads.items()})


def detections(codec: Codec, heads: dict) -> int:
    return int((decode_heads(codec, heads).object_scores > SCORE_THRESHOLD).sum())


def check_heads(heads: dict, source: str):
    """Head names must survive ONNX conversion, otherwise decode would silently key the wrong tensors."""
    missing = set(HEADS) - set(heads)
    if missing:
        raise SystemExit(f"{source} exports no {sorted(missing)} output; got {sorted(heads)}")


def val_frame(config) -> str:
    """First frame of the inference dataset, read by streaming its annotation JSON, which is ~400 MB."""
    spec = config.inference.data[0]
    name = _generate_annotation_name(spec.kwargs["split"], spec.kwargs["full"], spec.subset)
    with open(pathlib.Path(spec.root) / name, encoding="utf-8") as handle:
        match = re.search(r'"file_name":\s*"([^"]+)"', handle.read(1 << 20))
    if match is None:
        raise RuntimeError(f"No file_name in the head of {name}")
    return str(pathlib.Path(spec.root) / match.group(1).replace("\\/", "/"))


def keras_worker(config_path: str, run_dir: str, threads: int, iters: int, warmup: int):
    """Fresh process per thread count: TF refuses to resize its op pools after the first kernel runs."""
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)

    config, model, codec = build(config_path)
    export.load_weights_into(model, run_dir)
    forward = serving(model)
    image = tf.constant(load_image(config))
    forward_p50, forward_p95 = timeit(lambda: forward(image), iters, warmup)
    end_p50, _ = timeit(lambda: decode_heads(codec, forward(image)), iters, warmup)
    print("JSON " + json.dumps({"forward_p50": forward_p50, "forward_p95": forward_p95, "end_p50": end_p50}))


def keras_row(config_path: str, run_dir: str, threads: int, iters: int, warmup: int):
    command = [sys.executable, __file__, "--keras-worker", config_path, run_dir,
               str(threads), str(iters), str(warmup)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Keras worker failed for {threads} threads:\n{result.stderr[-2000:]}")
    return json.loads([line for line in result.stdout.splitlines() if line.startswith("JSON ")][-1][5:])


def render_table(rows: list, path: pathlib.Path):
    """One markdown table to stdout and to `OUT/benchmark.md`."""
    header = ("| runtime | threads | forward p50 ms | forward p95 ms | forward FPS | end-to-end p50 ms "
              "| end-to-end FPS |\n|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {row['runtime']} | {row['threads']} | {row['forward_p50']:.2f} | {row['forward_p95']:.2f} "
        f"| {row['forward_fps']:.1f} | {row['end_p50']:.2f} | {row['end_fps']:.1f} |\n" for row in rows)
    print("\n" + header + body)
    path.write_text(header + body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to the experiment YAML")
    parser.add_argument("run_dir", help="training run dir holding checkpoints/*.weights.h5")
    parser.add_argument("--out-dir", default=None, help="export destination; defaults to RUN_DIR/export")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8], help="thread counts to sweep")
    parser.add_argument("--iters", type=int, default=300, help="timed iterations per runtime and thread count")
    parser.add_argument("--warmup", type=int, default=50, help="untimed iterations before each measurement")
    parser.add_argument("--image", default=None, help="real 320x200 image; random uint8-range noise by default")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir or (pathlib.Path(args.run_dir) / "export"))
    out_dir.mkdir(parents=True, exist_ok=True)

    config, model, codec = build(args.config)
    weights = export.load_weights_into(model, args.run_dir)
    print(f" [export_benchmark] weights: {weights}")

    saved_model_dir = out_dir / "saved_model"
    export.export_saved_model(model, str(saved_model_dir))
    onnx_path = to_onnx(saved_model_dir, out_dir / "model.onnx")
    saved_bytes = sum(f.stat().st_size for f in saved_model_dir.rglob("*") if f.is_file())
    print(f" [export_benchmark] saved_model: {saved_bytes / 1e6:.2f} MB, "
          f"model.onnx: {pathlib.Path(onnx_path).stat().st_size / 1e6:.2f} MB")

    forward = serving(model)
    image = load_image(config, args.image)
    reference = {name: np.asarray(tensor) for name, tensor in forward(image).items()}
    heads = ort_runner(onnx_path, 1)(image)
    check_heads(reference, "Keras")
    check_heads(heads, "ONNX")
    diffs = {name: float(np.abs(reference[name] - heads[name]).max()) for name in reference}
    print(" [export_benchmark] max abs diff Keras vs ONNX Runtime: "
          + ", ".join(f"{name} {value:.2e}" for name, value in sorted(diffs.items())))
    if max(diffs.values()) > 1e-3:
        raise SystemExit(f"ONNX output mismatch (> 1e-3): {diffs}")

    rows = []
    for threads in args.threads:
        keras = keras_row(args.config, args.run_dir, threads, args.iters, args.warmup)
        rows.append({"runtime": "Keras", "threads": threads,
                     "forward_p50": keras["forward_p50"], "forward_p95": keras["forward_p95"],
                     "forward_fps": 1e3 / keras["forward_p50"], "end_p50": keras["end_p50"],
                     "end_fps": 1e3 / keras["end_p50"]})
        for runtime, runner in (("ONNX Runtime", ort_runner), ("OpenVINO", ov_runner)):
            run = runner(onnx_path, threads)
            forward_p50, forward_p95 = timeit(lambda: run(image), args.iters, args.warmup)
            end_p50, _ = timeit(lambda: decode_heads(codec, run(image)), args.iters, args.warmup)
            rows.append({"runtime": runtime, "threads": threads, "forward_p50": forward_p50,
                         "forward_p95": forward_p95, "forward_fps": 1e3 / forward_p50,
                         "end_p50": end_p50, "end_fps": 1e3 / end_p50})
        print(f" [export_benchmark] threads={threads} done")

    render_table(rows, out_dir / "benchmark.md")

    try:
        frame_path = val_frame(config)
    except Exception as error:  # a missing dataset must not cost the benchmark that already ran
        print(f" [export_benchmark] detection sanity skipped: {error}")
        return
    frame = load_image(config, frame_path)
    keras_count = detections(codec, {name: np.asarray(tensor) for name, tensor in forward(frame).items()})
    onnx_count = detections(codec, ort_runner(onnx_path, 1)(frame))
    print(f" [export_benchmark] detections > {SCORE_THRESHOLD} on {frame_path}: Keras {keras_count}, ORT {onnx_count}")
    if keras_count != onnx_count:
        raise SystemExit(f"Detection count differs on {frame_path}: Keras {keras_count} vs ORT {onnx_count}")


if __name__ == "__main__":
    # The worker re-enters through argv, carrying no CLI of its own.
    if len(sys.argv) > 1 and sys.argv[1] == "--keras-worker":
        keras_worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    else:
        main()
