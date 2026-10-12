import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorflow")

import verres as V

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # `tools` is a plain directory, not an installed package

from tools import colab_launch as cl  # noqa: E402

CLASS_MAPPING = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
COS = "config/exp_od_small_fcnn_colab_cos.yml"
DEBUG = "config/exp_od_small_fcnn_colab_debug.yml"


def _load(relative_path: str) -> V.Config:
    return V.Config.from_paths(str(REPO_ROOT / relative_path), CLASS_MAPPING)


@pytest.fixture
def fixture_repo(tmp_path) -> Path:
    """Working tree holding tracked, untracked, ignored and symlinked entries."""
    repo = tmp_path / "repo"
    (repo / "verres").mkdir(parents=True)
    (repo / "config").mkdir()
    (repo / "artifactory").mkdir()
    (repo / "main.py").write_text("# entry point\n")
    (repo / "verres" / "__init__.py").write_text("")
    (repo / "config" / "exp.yml").write_text("context: {}\n")
    (repo / "notebook.ipynb").write_text("{}")
    (repo / "artifactory" / "run.txt").write_text("output")
    (repo / ".gitignore").write_text("artifactory/\n")
    (repo / "verres-memory").symlink_to(tmp_path / "vault")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "main.py", "verres", "config", ".gitignore", "notebook.ipynb"],
                   cwd=repo, check=True)
    (repo / "untracked_new.py").write_text("# never added to the index\n")
    return repo


def test_tarball_carries_the_working_tree(fixture_repo, tmp_path):
    with tarfile.open(cl.build_tarball(tmp_path / "verres.tar.gz", repo=fixture_repo)) as tar:
        names = tar.getnames()

    assert {"verres/main.py", "verres/verres/__init__.py", "verres/config/exp.yml",
            "verres/untracked_new.py"} <= set(names)
    assert all(name.startswith("verres/") for name in names)
    for name in names:
        parts = Path(name).parts
        assert not {"verres-memory", "artifactory", ".git"} & set(parts)
        assert not name.endswith(".ipynb")


def test_start_runs_detached_with_an_exit_file(monkeypatch):
    sources = []
    monkeypatch.setattr(cl, "exec_remote", lambda session, source, timeout: sources.append(source) or "")

    cl.start(SimpleNamespace(session="verres", config=DEBUG, epochs=7))
    source = sources[-1]

    assert f"main.py -c {DEBUG}" in source
    assert "-u training.epochs=7" in source
    assert "echo $? > /content/train.exit" in source
    assert "start_new_session=True" in source


def test_start_without_epochs_override(monkeypatch):
    sources = []
    monkeypatch.setattr(cl, "exec_remote", lambda session, source, timeout: sources.append(source) or "")

    cl.start(SimpleNamespace(session="verres", config=DEBUG, epochs=None))

    assert "-u training.epochs" not in sources[-1]


@pytest.mark.parametrize("line,expected", [("STATUS RUNNING", cl.RUNNING),
                                           ("STATUS FINISHED", cl.FINISHED),
                                           ("STATUS DIED exit=1", cl.DIED),
                                           ("kernel restarted", cl.DIED)])
def test_status_exit_code_mapping(monkeypatch, line, expected):
    monkeypatch.setattr(cl, "exec_remote", lambda session, source, timeout: f"{line}\nSTATUS_END\n")

    assert cl.status(SimpleNamespace(session="verres")) == expected


def _run_status(monkeypatch, tmp_path, capsys, pid, exit_code):
    """Exec the generated remote snippet against tmp_path files and return its STATUS line."""
    exit_file = tmp_path / "train.exit"
    if exit_code is not None:
        exit_file.write_text(exit_code)
    pid_file = tmp_path / "train.pid"
    pid_file.write_text(str(pid))
    monkeypatch.setattr(cl, "REMOTE_PID", str(pid_file))
    monkeypatch.setattr(cl, "REMOTE_EXIT", str(exit_file))
    monkeypatch.setattr(cl, "REMOTE_LOG", str(tmp_path / "train.log"))

    exec(cl.status_source(), {})
    out = capsys.readouterr().out
    return next(line for line in out.splitlines() if line.startswith("STATUS "))


@pytest.mark.parametrize("pid,exit_code,expected", [
    (os.getpid(), "0", "STATUS FINISHED"),  # the exit file wins over a not-yet-reaped pid
    (os.getpid(), "1", "STATUS DIED exit=1"),
    (os.getpid(), None, "STATUS RUNNING"),
    (99999999, None, "STATUS DIED exit=None"),
])
def test_status_snippet_resolution_order(monkeypatch, tmp_path, capsys, pid, exit_code, expected):
    assert _run_status(monkeypatch, tmp_path, capsys, pid, exit_code) == expected


def _proc_state(pid) -> str:
    with open(f"/proc/{pid}/stat") as handle:
        return handle.read().rsplit(")", 1)[-1].split()[0]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="zombie pids need /proc")
def test_status_snippet_ignores_a_zombie_child(monkeypatch, tmp_path, capsys):
    pid = os.fork()
    if pid == 0:
        os._exit(0)  # stays a zombie until reaped below
    try:
        deadline = time.monotonic() + 5
        while _proc_state(pid) != "Z" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _proc_state(pid) == "Z"
        assert _run_status(monkeypatch, tmp_path, capsys, pid, None) == "STATUS DIED exit=None"
    finally:
        os.waitpid(pid, 0)


@pytest.mark.parametrize("config_path", [COS, DEBUG])
def test_colab_configs_parse(config_path):
    config = _load(config_path)

    assert config.context.artifactory_root == "/content/artifactory"
    assert config.training.lr_schedule_spec["name"] == "CosineDecay"
    assert config.training.lr_schedule_spec["decay_steps"] == config.training.epochs * config.training.steps_per_epoch
    assert [loss["weight"] for loss in config.training.criteria_spec["losses"]
            if loss["feature"] == "box_wh"] == [20.]
    assert {callback["name"] for callback in config.training.callbacks} == {
        "LatestModelCheckpoint", "ObjectMAP", "CSVLogger", "TensorBoard"}


REAL_LOG = REPO_ROOT / "artifactory/colab/xp_20261011.184306/train.log"


def test_eval_marker_counts_the_pulled_log():
    if not REAL_LOG.exists():
        pytest.skip("pulled Colab log not available")

    assert REAL_LOG.read_text(errors="replace").count(cl.EVAL_MARKER) == 1


def test_debug_config_evaluates_the_map_exactly_once(tmp_path):
    config = _load(DEBUG)
    assert config.training.epochs == 5
    assert config.training.data[0].subset == "train-mini"
    assert config.evaluation.data[0].subset == "val-mini"
    config.context.artifactory_root = str(tmp_path)  # the callback's checkpoint path is built from it

    every_n_epochs = next(callback["every_n_epochs"] for callback in config.training.callbacks
                          if callback["name"] == "ObjectMAP")
    callback = V.optim.callbacks.ObjectMAP(config, checkpoint_best=False, every_n_epochs=every_n_epochs)
    evaluations = []
    callback.evaluator = SimpleNamespace(execute=lambda *args, **kwargs: evaluations.append(args) or {})

    for epoch in range(config.training.epochs):
        callback.on_epoch_end(epoch)

    assert len(evaluations) == 1
