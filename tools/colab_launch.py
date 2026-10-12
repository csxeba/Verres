#!/usr/bin/env python3
"""Drive a Verres training on a Colab VM through the `colab` CLI.

The working tree is tar'ed locally (gitignore honoured, so the uncommitted state goes up as is),
uploaded, extracted to /content/verres and trained detached, so a disconnect cannot kill the run.
"""
import argparse
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/content/verres"
DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/share/cocodoom-v1.0.1.tar.gz"
DATASET_ROOT = "/content/cocodoom"
REMOTE_TARBALL = "/content/verres.tar.gz"
REMOTE_LOG = "/content/train.log"
REMOTE_EXIT = "/content/train.exit"
REMOTE_PID = "/content/train.pid"
EXCLUDE_DIRS = {".git", "verres-memory", "artifactory", "verres-artifactory", "experiments", "tmp", "models"}
EXCLUDE_SUFFIXES = {".ipynb", ".avi", ".mp4", ".png", ".jpg", ".jpeg", ".h5"}
RUNNING, FINISHED, DIED = 0, 3, 4
EVAL_MARKER = "COCO eval progress: 100.00%"  # counts evaluations even when no detections are produced


def colab(*args, timeout=None) -> subprocess.CompletedProcess:
    result = subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        tail = "\n".join(((result.stderr or "") + (result.stdout or "")).strip().splitlines()[-15:])
        raise RuntimeError(f"colab {' '.join(args)} failed with {result.returncode}:\n{tail}")
    return result


def exec_remote(session: str, source: str, timeout: float) -> str:
    """Run `source` in the VM's persistent kernel and return its stdout."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(source)
        path = handle.name
    try:
        return colab("exec", "-s", session, "-f", path, "--timeout", str(timeout), timeout=timeout + 60).stdout
    finally:
        os.unlink(path)


def tarball_files(repo: Path = REPO):
    """Working tree files that belong on the VM: tracked + untracked, .gitignore honoured."""
    listing = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"],
                             cwd=repo, capture_output=True, text=True, check=True).stdout
    for name in listing.splitlines():
        if any(part in EXCLUDE_DIRS for part in Path(name).parts) or Path(name).suffix in EXCLUDE_SUFFIXES:
            continue
        # verres-memory is a symlink into a private vault: it must never be followed into the archive.
        if os.path.islink(repo / name):
            continue
        yield name


def build_tarball(dest: Path, repo: Path = REPO) -> Path:
    with tarfile.open(dest, "w:gz") as tar:
        for name in tarball_files(repo):
            tar.add(repo / name, arcname=str(Path("verres") / name), recursive=False)
    return dest


def setup_source() -> str:
    return f'''import os, shutil, subprocess, sys
os.chdir("/content")
shutil.rmtree({REMOTE_REPO!r}, ignore_errors=True)
subprocess.run(["tar", "-xzf", {REMOTE_TARBALL!r}, "-C", "/content"], check=True)
os.chdir({REMOTE_REPO!r})
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
if not os.path.exists(os.path.join({DATASET_ROOT!r}, "run-train.json")):
    archive = os.path.join("/content", os.path.basename({DATASET_URL!r}))
    if not os.path.exists(archive):
        subprocess.run(["wget", "-c", "-q", "-O", archive, {DATASET_URL!r}], check=True)
    os.makedirs({DATASET_ROOT!r}, exist_ok=True)
    subprocess.run(["tar", "-xzf", archive, "-C", {DATASET_ROOT!r}, "--strip-components=1"], check=True)
    os.remove(archive)
import keras, tensorflow as tf
print("SETUP tensorflow", tf.__version__, "| keras", keras.__version__, keras.backend.backend())
print("SETUP gpus", [gpu.name for gpu in tf.config.list_physical_devices("GPU")])
'''


def start_source(command: str) -> str:
    return f'''import os, subprocess
os.chdir({REMOTE_REPO!r})
if os.path.exists({REMOTE_EXIT!r}):
    os.remove({REMOTE_EXIT!r})
log = open({REMOTE_LOG!r}, "wb")
proc = subprocess.Popen(["bash", "-c", {command!r}], cwd={REMOTE_REPO!r},
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
with open({REMOTE_PID!r}, "w") as handle:
    handle.write(str(proc.pid))
print("STARTED pid", proc.pid, "log", {REMOTE_LOG!r})
'''


def status_source() -> str:
    return f'''import glob, os
pid = open({REMOTE_PID!r}).read().strip() if os.path.exists({REMOTE_PID!r}) else ""
exit_code = open({REMOTE_EXIT!r}).read().strip() if os.path.exists({REMOTE_EXIT!r}) else None
if exit_code is not None:
    print("STATUS FINISHED" if exit_code == "0" else "STATUS DIED exit=" + exit_code)
else:
    # The pid survives as a zombie until its parent reaps it, so check the /proc state field.
    stat = "/proc/" + pid + "/stat"
    state = open(stat).read().rsplit(")", 1)[-1].split()[0] if pid and os.path.exists(stat) else "Z"
    print("STATUS RUNNING" if state != "Z" else "STATUS DIED exit=None")
print("---- {REMOTE_LOG} tail ----")
tail = open({REMOTE_LOG!r}, errors="replace").read().splitlines()[-15:] if os.path.exists({REMOTE_LOG!r}) else []
print("\\n".join(tail))
runs = sorted(glob.glob("/content/artifactory/*/*/xp_*"), key=os.path.getmtime)
if runs:
    csv = os.path.join(runs[-1], "training_logs.csv")
    if os.path.exists(csv):
        print("---- " + csv + " ----")
        print("\\n".join(open(csv, errors="replace").read().splitlines()[-3:]))
print("STATUS_END")
'''


def pull_source(what: str) -> str:
    return f'''import glob, os, subprocess
runs = sorted(glob.glob("/content/artifactory/*/*/xp_*"), key=os.path.getmtime)
if not runs:
    print("PULL_NONE")
else:
    run = runs[-1]
    os.chdir(run)
    members = ["."] if {what!r} == "all" else (
        ["checkpoints", "config.yml", "training_logs.csv"]
        + glob.glob("*-kpi.json") + glob.glob("*/**/*-kpi.json", recursive=True))
    # tar exits 1 when a file changes while being read, which is expected on a live run.
    subprocess.run("tar -czf /content/run.tar.gz " + " ".join(members) + " || [ $? -eq 1 ]",
                   shell=True, check=True)
    print("PULLED", run, os.path.getsize("/content/run.tar.gz"))
'''


def up(args):
    colab("new", "-s", args.session, "--gpu", args.gpu, timeout=900)
    with tempfile.TemporaryDirectory() as tmp:
        tarball = build_tarball(Path(tmp) / "verres.tar.gz")
        print(f"[up] tarball {tarball.stat().st_size / 1e6:.1f} MB -> {REMOTE_TARBALL}")
        colab("upload", str(tarball), REMOTE_TARBALL, "-s", args.session, timeout=1800)
    print(exec_remote(args.session, setup_source(), timeout=1800).strip())


def start(args):
    command = f"KERAS_BACKEND=tensorflow python main.py -c {shlex.quote(args.config)}"
    if args.epochs:
        command += f" -u training.epochs={args.epochs}"
    command += f"; echo $? > {REMOTE_EXIT}"
    print(exec_remote(args.session, start_source(command), timeout=120).strip())


def status(args) -> int:
    output = exec_remote(args.session, status_source(), timeout=120)
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("STATUS "):
            print("\n".join(lines[index:]))
            return {"RUNNING": RUNNING, "FINISHED": FINISHED}.get(line.split()[1], DIED)
    print(output.strip()[-1000:])
    return DIED


def newest_run(dest: Path) -> Path:
    runs = sorted(dest.glob("xp_*"), key=os.path.getmtime)
    if not runs:
        raise RuntimeError(f"No pulled run under {dest}")
    return runs[-1]


def pull(args) -> Path:
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    output = exec_remote(args.session, pull_source(args.what), timeout=600)
    line = next((l for l in output.splitlines() if l.startswith("PULLED ")), None)
    if line is None:
        raise RuntimeError(f"Nothing to pull: {output.strip()[-500:]}")
    remote_dir, size = line.split()[1], int(line.split()[2])
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "run.tar.gz"
        colab("download", "/content/run.tar.gz", str(tarball), "-s", args.session, timeout=1800)
        target = dest / Path(remote_dir).name
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball) as tar:
            tar.extractall(target, filter="data")
    print(f"[pull] {remote_dir} ({size / 1e6:.1f} MB) -> {target}")
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.stat().st_size:
            print(f"[pull]   {path.relative_to(target)}  {path.stat().st_size / 1e6:.2f} MB")
    return target


def wait(args) -> int:
    next_pull = time.time() + args.pull_every * 60 if args.pull_every else None
    while True:
        code = status(args)
        if code != RUNNING:
            break
        if next_pull and time.time() >= next_pull:
            pull(args)
            next_pull = time.time() + args.pull_every * 60
        time.sleep(args.interval)
    pull(args)
    return code


def debug(args):
    try:
        up(args)
        start(argparse.Namespace(session=args.session, config=args.config, epochs=None))
        code = wait(argparse.Namespace(session=args.session, interval=30, pull_every=None,
                                      dest=args.dest, what="checkpoints"))
        run_dir = newest_run(Path(args.dest))
        log = run_dir / "train.log"
        colab("download", REMOTE_LOG, str(log), "-s", args.session, timeout=600)
        failures = []
        if code != FINISHED:
            failures.append(f"training ended with status {code}")
        checkpoints = run_dir / "checkpoints" / "tf_ckpt"
        if not (checkpoints.is_dir() and any(p.stat().st_size for p in checkpoints.iterdir() if p.is_file())):
            failures.append("checkpoints/tf_ckpt is missing or empty")
        logs = run_dir / "training_logs.csv"
        rows = logs.read_text().splitlines()[1:] if logs.exists() else []
        if len(rows) != 5:
            failures.append(f"training_logs.csv has {len(rows)} epoch rows, expected 5")
        evaluations = log.read_text(errors="replace").count(EVAL_MARKER) if log.exists() else 0
        if evaluations != 1:
            failures.append(f"ObjectMAP ran {evaluations} times, expected exactly 1")
        print("[debug] run dir:", run_dir)
        print("[debug] PASS" if not failures else "[debug] FAIL: " + "; ".join(failures))
        return 0 if not failures else 1
    finally:
        if not args.keep:
            down(args)


def down(args):
    colab("stop", "-s", args.session, timeout=600)
    print(f"[down] session {args.session} stopped")


def main():
    parser = argparse.ArgumentParser(prog="colab_launch", description=__doc__)
    parser.add_argument("--session", "-s", default="verres")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("up").add_argument("--gpu", default="T4")
    parser_start = sub.add_parser("start")
    parser_start.add_argument("--config", required=True)
    parser_start.add_argument("--epochs", type=int)
    sub.add_parser("status")
    parser_pull = sub.add_parser("pull")
    parser_pull.add_argument("--dest", default="artifactory/colab")
    parser_pull.add_argument("--what", choices=["checkpoints", "all"], default="checkpoints")
    parser_wait = sub.add_parser("wait")
    parser_wait.add_argument("--interval", type=int, default=120)
    parser_wait.add_argument("--pull-every", type=int, default=None, help="minutes between pulls")
    parser_wait.add_argument("--dest", default="artifactory/colab")
    parser_wait.add_argument("--what", choices=["checkpoints", "all"], default="checkpoints")
    sub.add_parser("down")
    parser_debug = sub.add_parser("debug")
    parser_debug.add_argument("--config", default="config/exp_od_small_fcnn_colab_debug.yml")
    parser_debug.add_argument("--gpu", default="T4")
    parser_debug.add_argument("--dest", default="artifactory/colab")
    parser_debug.add_argument("--keep", action="store_true", help="leave the VM running")

    args = parser.parse_args()
    if args.command == "status":
        sys.exit(status(args))
    {"up": up, "start": start, "pull": pull, "wait": wait, "down": down, "debug": debug}[args.command](args)


if __name__ == "__main__":
    main()
