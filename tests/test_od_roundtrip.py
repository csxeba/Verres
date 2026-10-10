import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("cv2")

import verres as V
from verres.data.codec import Codec
from verres.data.sample import Input, Label, Sample

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGE_WH = (320, 200)
STRIDE = 4
BOXES_XY = np.array([[40., 50., 120., 130.], [180., 60., 260., 160.]])  # Format: image (x0, y0, x1, y1)
TYPES = np.array([0, 1])


def _logits(probabilities: np.ndarray) -> np.ndarray:
    """Decode sigmoids the head output, so ground-truth heatmaps are fed back as logits."""
    p = np.clip(probabilities.astype(np.float32), 1e-6, 1. - 1e-6)
    return np.log(p / (1. - p))


def _sample() -> Sample:
    width, height = IMAGE_WH
    scale_xyxy = np.array([width, height, width, height], dtype=float)
    centers_xy = np.stack([BOXES_XY[:, [0, 2]].mean(axis=1), BOXES_XY[:, [1, 3]].mean(axis=1)], axis=1)
    return Sample.create(
        ID=0,
        input_=Input(image_path="synthetic.png", shape_whc=(width, height, 3)),
        label=Label(
            object_centers=centers_xy / np.array([width, height]),
            object_types=TYPES,
            object_keypoint_coords=BOXES_XY / scale_xyxy[None, :],
        ),
    )


def _codec() -> Codec:
    config = V.Config.from_paths(
        str(REPO_ROOT / "config/exp_od_baseline.yml"),
        str(REPO_ROOT / "config/class_mappings/cocodoom.yml"),
    )
    return Codec(config, [
        {"name": "uniform_heatmap", "stride": STRIDE},
        {"name": "refinement", "stride": STRIDE},
        {"name": "box_wh", "stride": STRIDE},
    ])


def test_od_roundtrip():
    codec = _codec()

    sample = codec.encode_sample(_sample())
    tensors = dict(sample.encoded_tensors)
    tensors["heatmap"] = _logits(tensors["heatmap"])

    # Sparse (N, 2) targets -> dense (H, W, 2) maps, as the head would output them.
    output_hw = np.array(tensors["heatmap"].shape[:2])
    cells = np.floor(sample.label.object_centers[:, ::-1] * output_hw).astype(int)
    for name in ("refinement", "box_wh"):
        dense = np.zeros(tuple(output_hw) + (2,), dtype=np.float32)
        dense[cells[:, 0], cells[:, 1]] = tensors[name]
        tensors[name] = dense

    decoded = codec.decode(tensors)

    order = np.argsort(np.asarray(decoded.object_keypoint_coords)[:, 0])  # peak order is raster order
    one_output_cell = STRIDE / np.array([IMAGE_WH[0], IMAGE_WH[1], IMAGE_WH[0], IMAGE_WH[1]], dtype=float)
    assert np.allclose(np.asarray(decoded.object_keypoint_coords)[order],
                       sample.label.object_keypoint_coords, atol=one_output_cell)
    assert np.array_equal(np.asarray(decoded.object_types)[order], TYPES)
