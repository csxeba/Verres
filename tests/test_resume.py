import csv
import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf
import yaml

import verres as V
from verres.execution.training import TrainingExecutor
from verres.optim.checkpoint import restore_latest
from verres.optim.schedule import ConstantSchedule

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASS_MAPPING = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
BASE_CONFIG = REPO_ROOT / "config/exp_od_small_fcnn.yml"

HW = (8, 8)
STEPS = 2
BATCH = 2
SAMPLES = 8


class TinyModel(tf.keras.Model):
    """Stand-in for VRSArchitecture: same preprocess_input/train_step wiring, no real head."""

    def __init__(self):
        super().__init__()
        self.hidden = tf.keras.layers.Conv2D(2, 3, padding="same")
        self.out = tf.keras.layers.Conv2D(1, 1)

    def preprocess_input(self, image):
        return image

    def call(self, inputs, training=None, mask=None):
        return self.out(self.hidden(inputs))


class TinyCriteria:
    """Stand-in for VRSCriteria: the training step only reads the `loss` key."""

    def __call__(self, batch, prediction):
        return {"loss": tf.reduce_mean(tf.square(prediction - batch["y"]))}


@pytest.fixture(autouse=True)
def artifactory_singleton():
    V.Artifactory.default_instance = None
    yield
    V.Artifactory.default_instance = None


@pytest.fixture
def experiment(tmp_path):
    """Tiny training config derived from the small_fcnn experiment plus a synthetic stream."""
    spec = yaml.safe_load(BASE_CONFIG.read_text())
    spec["context"]["artifactory_root"] = str(tmp_path / "artifactory")
    spec["context"]["experiment_set"] = "tests"
    spec["context"]["experiment_name"] = "resume"
    spec["context"]["verbose"] = 1
    spec["model"]["input_height"], spec["model"]["input_width"] = HW
    spec["training"]["epochs"] = 1
    spec["training"]["steps_per_epoch"] = STEPS
    spec["training"]["batch_size"] = BATCH
    spec["training"]["callbacks"] = [{"name": "CSVLogger"}]
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(spec))

    rng = np.random.default_rng(0)
    stream = tf.data.Dataset.from_tensor_slices({
        "image": rng.normal(size=(SAMPLES, *HW, 3)).astype("float32"),
        "y": rng.normal(size=(SAMPLES, *HW, 1)).astype("float32"),
    }).batch(BATCH).repeat()

    return V.Config.from_paths(str(config_path), CLASS_MAPPING), stream


def _executor(config):
    schedule = ConstantSchedule(0.01)
    model = TinyModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    return TrainingExecutor(config, model, TinyCriteria(), optimizer, schedule), model, optimizer


def _read_epochs(path):
    with open(path) as handle:
        return [int(row["epoch"]) for row in csv.DictReader(handle)]


def test_resume_continues_in_place(experiment):
    config, stream = experiment
    executor, model, optimizer = _executor(config)
    executor.execute(stream=stream)

    run_dir = pathlib.Path(V.Artifactory.default_instance.root)
    assert int(optimizer.iterations.numpy()) == STEPS  # 1 epoch x 2 steps

    # The checkpoint alone rebuilds the run: weights, optimizer step counter and epoch.
    probe, probe_optimizer = TinyModel(), tf.keras.optimizers.Adam(learning_rate=0.01)
    probe.build((None, *HW, 3))
    probe_optimizer.build(probe.trainable_variables)
    assert restore_latest(run_dir, probe, probe_optimizer) == 1
    assert int(probe_optimizer.iterations.numpy()) == STEPS
    for trained, restored in zip(model.get_weights(), probe.get_weights()):
        assert np.allclose(trained, restored)

    # Adam slots must come back as well, not just the step counter.
    assert len(probe_optimizer.variables()) == len(optimizer.variables())
    for trained, restored in zip(optimizer.variables(), probe_optimizer.variables()):
        assert np.allclose(trained.numpy(), restored.numpy()), trained.name

    # Resume the way main(continue_train=run_dir) does: same directory, more epochs.
    config = V.Config.from_paths(str(run_dir / "config.yml"), CLASS_MAPPING)
    config.training.epochs = 2
    config.training.initial_epoch = V.utils.logging_utils.extract_last_epoch(run_dir)
    assert config.training.initial_epoch == 1
    V.Artifactory.default_instance = None  # main() drops the singleton of the previous launch
    assert V.Artifactory(config, attach_to=str(run_dir)).root == run_dir

    executor, _, optimizer = _executor(config)
    executor.execute(stream=stream)

    assert pathlib.Path(V.Artifactory.default_instance.root) == run_dir
    assert list(run_dir.parent.iterdir()) == [run_dir]  # no new timestamped directory
    assert int(optimizer.iterations.numpy()) == 2 * STEPS  # step counter continued, not reset
    assert _read_epochs(run_dir / "training_logs.csv") == [0, 1]

    header = (run_dir / "training_logs.csv").read_text().strip().splitlines()[0]
    assert header.startswith("epoch")  # the log was appended to, not truncated or restarted
