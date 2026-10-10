import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

import verres as V

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = str(REPO_ROOT / "config/exp_od_small_fcnn.yml")  # sparse `mean_absolute_error` regression terms
CLASS_MAPPING_PATH = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
H, W = 12, 16
NUM_OBJECTS = 2
REFINEMENT = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
BOX_WH = np.array([[0.1, 0.1], [0.3, 0.3]], dtype=np.float32)


def _losses(locations, refinement, box_wh):
    config = V.Config.from_paths(CONFIG_PATH, CLASS_MAPPING_PATH)
    criteria = V.optim.criteria.VRSCriteria(config, config.training.criteria_spec)
    y_true = {
        "object_center_locations": locations,
        "heatmap": np.zeros((1, H, W, 1), dtype=np.float32),
        "refinement": refinement,
        "box_wh": box_wh,
    }
    y_pred = {name: tf.zeros(shape, dtype=tf.float32) for name, shape in
              (("heatmap", (1, H, W, 1)), ("refinement", (1, H, W, 2)), ("box_wh", (1, H, W, 2)))}
    return {name: float(value) for name, value in criteria(y_true, y_pred).items()}


def test_sparse_losses_keep_the_old_formula_for_batches_with_objects():
    locations = np.array([[0, 1, 2], [0, 4, 5]], dtype=np.int64)  # (batch, y, x) format: matrix

    losses = _losses(locations, REFINEMENT, BOX_WH)

    # mean(|target|) over the gathered cells, weighted and divided by the number of objects.
    assert losses["refinement"] == pytest.approx(np.abs(REFINEMENT).mean())
    assert losses["box_wh"] == pytest.approx(np.abs(BOX_WH).mean())
    assert losses["loss"] == pytest.approx(losses["heatmap"] / NUM_OBJECTS
                                           + np.abs(REFINEMENT).mean() * 10. / NUM_OBJECTS
                                           + np.abs(BOX_WH).mean() / NUM_OBJECTS)


def test_batch_without_objects_has_no_regression_loss_and_no_nan():
    empty_locations = np.zeros((0, 3), dtype=np.int64)

    losses = _losses(empty_locations, np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32))
    reference = _losses(np.array([[0, 1, 2], [0, 4, 5]], dtype=np.int64), REFINEMENT, BOX_WH)

    assert losses["refinement"] == 0.
    assert losses["box_wh"] == 0.
    assert losses["heatmap"] == pytest.approx(reference["heatmap"])  # the dense term is unaffected
    assert losses["loss"] == pytest.approx(losses["heatmap"] / 1.)  # count is clamped to 1, so no blow-up
    assert all(np.isfinite(list(losses.values())))
