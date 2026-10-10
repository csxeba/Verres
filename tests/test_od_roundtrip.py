import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("cv2")

import verres as V
from verres.data.codec import Codec
from verres.data.sample import Input, Label, Sample
from verres.testing import fake_outputs

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGE_WH = (320, 200)
STRIDE = 4
BOXES_XY = np.array([[40., 50., 120., 130.], [180., 60., 260., 160.]])  # Format: image (x0, y0, x1, y1)
TYPES = np.array([0, 1])


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
    decoded = codec.decode(fake_outputs.dense_model_output(sample))

    order = np.argsort(np.asarray(decoded.object_keypoint_coords)[:, 0])  # peak order is raster order
    one_output_cell = STRIDE / np.array([IMAGE_WH[0], IMAGE_WH[1], IMAGE_WH[0], IMAGE_WH[1]], dtype=float)
    assert np.allclose(np.asarray(decoded.object_keypoint_coords)[order],
                       sample.label.object_keypoint_coords, atol=one_output_cell)
    assert np.array_equal(np.asarray(decoded.object_types)[order], TYPES)


def test_od_roundtrip_without_objects():
    """An image with no annotations must decode to empty boxes, never to None."""
    codec = _codec()
    sample = _sample()
    sample.label = Label(
        object_centers=np.zeros((0, 2)),
        object_types=np.zeros((0,), dtype=int),
        object_keypoint_coords=np.zeros((0, 4)),
    )

    sample = codec.encode_sample(sample)
    decoded = codec.decode(fake_outputs.dense_model_output(sample))

    assert decoded.object_keypoint_coords.shape == (0, 4)
    assert decoded.object_types.shape == decoded.object_scores.shape == (0,)
    detections = V.utils.cocodoom.generate_coco_detections(
        boxes_xy_01=decoded.object_keypoint_coords,
        types=decoded.object_types,
        scores=decoded.object_scores,
        image_shape_xy=IMAGE_WH,
        image_id=sample.ID)
    assert detections == []
