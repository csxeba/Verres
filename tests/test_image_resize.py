import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")
cv2 = pytest.importorskip("cv2")

import verres as V
from verres.data.sample import Input, Sample
from verres.data.transformation import ImageProcessor

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = str(REPO_ROOT / "config/exp_od_overfit_mnv3.yml")
CLASS_MAPPING_PATH = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
NATIVE_WH = (320, 200)


def _config(input_height: int) -> V.Config:
    config = V.Config.from_paths(CONFIG_PATH, CLASS_MAPPING_PATH)
    config.model.input_height = input_height
    return config


def _sample(image_path: pathlib.Path) -> Sample:
    return Sample.create(ID=0, input_=Input(image_path=str(image_path), shape_whc=NATIVE_WH + (3,)))


def _write_image(path: pathlib.Path) -> np.ndarray:
    width, height = NATIVE_WH
    image = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return image


def test_resizes_image_to_model_input(tmp_path):
    sample = _sample(tmp_path / "native.png")
    _write_image(pathlib.Path(sample.input.image_path))

    processor = ImageProcessor(_config(input_height=208), {"name": "image", "stride": 1})
    output = processor.call(sample)["image"]

    assert output.shape == (208, NATIVE_WH[0], 3)


def test_matching_size_is_pixel_identical(tmp_path):
    sample = _sample(tmp_path / "native.png")
    image = _write_image(pathlib.Path(sample.input.image_path))

    processor = ImageProcessor(_config(input_height=NATIVE_WH[1]), {"name": "image", "stride": 1})
    output = processor.call(sample)["image"]

    assert output.shape == (NATIVE_WH[1], NATIVE_WH[0], 3)
    assert np.array_equal(output, image)


def test_missing_image_raises(tmp_path):
    processor = ImageProcessor(_config(input_height=208), {"name": "image", "stride": 1})
    with pytest.raises(RuntimeError, match="No image found"):
        processor.call(_sample(tmp_path / "absent.png"))
