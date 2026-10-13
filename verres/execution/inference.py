import os
import time
from typing import List

import verres as V
from verres.data.codec import Codec


class Mode:

    DETECTION = "detection"
    RAW_HEATMAP = "raw_heatmap"
    SEMANTIC = "semantic"
    INSTANCE = "instance"
    PANOPTIC = "panoptic"


def _sequence_of(sample: V.data.Sample, root: str) -> str:
    """The `run<N>/map<NN>` prefix of a sample path, the only sequence grouping the recordings have."""
    relative_path = os.path.relpath(sample.input.image_path, root)
    return "/".join(relative_path.split(os.sep)[:2])


def _apply_sequence_selector(pipeline: V.data.Pipeline, spec: V.config.DatasetSpec):
    """Restricts a pipeline to one sequence, optionally to a `[start, end)` frame range inside it."""
    sequence = spec.kwargs.get("sequence")
    frame_range = spec.kwargs.get("frame_range")
    if sequence is None and frame_range is None:
        return

    dataset = pipeline.dataset
    keys = {ID: _sequence_of(dataset.index[ID], dataset.dataset_spec.root) for ID in dataset.IDs}
    IDs = dataset.IDs
    if sequence is not None:
        IDs = [ID for ID in IDs if keys[ID] == sequence]
        if not IDs:
            raise RuntimeError(f"Unknown sequence `{sequence}`. Available sequences: "
                               f"{', '.join(sorted(set(keys.values())))}")
    if frame_range is not None:
        start, end = frame_range
        IDs = IDs[start:end]
    if not IDs:
        raise RuntimeError(f"Sequence `{sequence}` with frame_range {frame_range} selected no frames.")

    dataset.IDs = IDs
    print(f" [Verres.inference] - Sequence selector `{sequence}` {frame_range}: {len(IDs)} frames")


class InferenceExecutor:

    def __init__(self,
                 config: V.Config,
                 model: V.architecture.VRSArchitecture,
                 pipelines: List[V.data.Pipeline],
                 decoder: Codec):

        self.cfg = config
        self.model = model
        self.pipelines = pipelines
        self.decoder = decoder

    @classmethod
    def factory(cls, config: V.Config):
        model = V.architecture.VRSArchitecture.factory(config)
        pipelines = V.data.factory(config, specs=config.inference.data)
        for pipeline in pipelines:
            _apply_sequence_selector(pipeline, pipeline.dataset.dataset_spec)

        # Inference encodes images only, so the decode chain is taken from the training transformations.
        if not config.training.data:
            raise RuntimeError("Inference needs `training.data[0].transformations` to decode model outputs.")
        decoder = Codec(config, config.training.data[0].transformations)
        return cls(config, model, pipelines, decoder)

    def execute(self):

        timer = V.utils.profiling.MultiTimer()
        iterator = iter(V.data.streaming.get_tf_dataset(self.cfg, self.pipelines, shuffle=False, batch_size=1))

        mode = self.cfg.inference.visualization_mode
        num_frames = len(self.pipelines[0].dataset)
        if 0 < self.cfg.inference.total_num_frames < num_frames:
            num_frames = self.cfg.inference.total_num_frames

        ema_fps = 0.
        with V.visualization.PredictionVisualizer(self.cfg,
                                                 score_threshold=self.cfg.inference.score_threshold,
                                                 realtime_playback=self.cfg.inference.realtime_playback) as vis:

            for i in range(1, num_frames + 1):

                with timer.time("data"):
                    tensor = next(iterator)["image"]

                compute_start = time.time()
                with timer.time("model"):
                    output = self.model(self.model.preprocess_input(tensor), training=False)
                    if mode == Mode.DETECTION:
                        output = self.decoder.decode({k: v[0] for k, v in output.items()})
                # Only model and decode count towards the FPS; drawing and IO must not drag it down.
                frame_fps = 1. / max(time.time() - compute_start, 1e-9)
                ema_fps = frame_fps if ema_fps == 0. else 0.9 * ema_fps + 0.1 * frame_fps

                hud = [f"compute {ema_fps:5.1f} FPS", f"frame {i}/{num_frames}"]
                hud += [f"{name} {value * 1e3:5.1f} ms" for name, value in timer.get_results(reset=True).items()]

                if mode == Mode.RAW_HEATMAP:
                    vis.draw_raw_heatmap(tensor, output, 0.4, hud=hud, write=True)
                elif mode == Mode.DETECTION:
                    vis.draw_detection(tensor, output.object_keypoint_coords, output.object_types,
                                       all_box_scores=output.object_scores, hud=hud, write=True)
                elif mode == Mode.PANOPTIC:
                    vis.draw_panoptic_segmentation(tensor, output, 0.4, hud=hud, write=True)
                elif mode == Mode.SEMANTIC:
                    vis.draw_semantic_segmentation(tensor, output, 0.4, hud=hud, write=True)
                elif mode == Mode.INSTANCE:
                    vis.draw_instance_segmentation(tensor, output, 0.4, hud=hud, write=True)
                else:
                    raise NotImplementedError(f"Mode `{mode}` is not implemented!")

                print(f"\r [Verres] - Inference {i / num_frames:>7.2%} - compute {ema_fps:.2f} FPS", end="")

        print()
