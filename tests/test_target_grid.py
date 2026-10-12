import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import verres as V
from verres.data.sample import Input, Label, Sample
from verres.data.transformation.regression import RefinementTensor
from verres.operation import numeric as T

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = str(REPO_ROOT / "config/exp_od_overfit_mnv3.yml")
CLASS_MAPPING_PATH = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
NATIVE_WH = (320, 200)
MODEL_HW = (208, 320)
STRIDE = 4
GRID_WH = (MODEL_HW[1] // STRIDE, MODEL_HW[0] // STRIDE)  # (80, 52)


def _config() -> V.Config:
    return V.Config.from_paths(CONFIG_PATH, CLASS_MAPPING_PATH)


def _sample(center_xy: tuple) -> Sample:
    width, height = NATIVE_WH
    return Sample.create(
        ID=0,
        input_=Input(image_path="synthetic.png", shape_whc=(width, height, 3)),
        label=Label(object_centers=np.array([center_xy], dtype=np.float32)),
    )


def _refinement(sample: Sample) -> np.ndarray:
    processor = RefinementTensor(_config(), {"name": "refinement", "stride": STRIDE})
    return processor.call(sample)["refinement"]


def test_on_grid_center_has_zero_refinement_and_cell_40_26():
    sample = _sample((0.5, 0.5))

    assert np.allclose(_refinement(sample), 0.)

    locations = T.get_object_center_locations([sample], STRIDE, MODEL_HW)
    assert locations.tolist() == [[0, 26, 40]]  # (batch, y, x)


def test_off_grid_center_refinement_is_the_cell_fraction():
    center_xy = (0.51, 0.49)
    sample = _sample(center_xy)

    expected_x = (0.51 * GRID_WH[0] % 1.) / GRID_WH[0]
    expected_y = (0.49 * GRID_WH[1] % 1.) / GRID_WH[1]
    assert np.allclose(_refinement(sample), [[expected_x, expected_y]])

    locations = T.get_object_center_locations([sample], STRIDE, MODEL_HW)
    assert locations.tolist() == [[0, 25, 40]]
