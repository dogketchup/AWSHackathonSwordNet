"""kubectl helpers. The workshop role only allows `get pods` and `pods/exec`, so everything
here is built on exec: running commands, and moving files as tar streams over stdin/stdout.
(`kubectl cp` is the same thing, but it cannot stream a directory in from a pipe.)
"""
import io
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path

CONTAINER = "app"
APP_DIR = "/workspace/swordnet-app"          # code (the swordnet/ package)
DATA_DIR = "/workspace/swordnet-data"        # datasets, runs, autolabel scratch
VENV = "/workspace/swordnet-venv"            # ultralytics + CPU torch, kept apart from vLLM's torch
VENV_PY = f"{VENV}/bin/python"
SYS_PY = "python3"                           # the image's python: has httpx, used by the agent


def _kubectl(*args):
    return ["kubectl", *args]


def sh(seat, script, check=True, input=None, timeout=None):
    """Run a bash script in the seat pod; return stdout as text."""
    cmd = _kubectl("exec", "-i" if input is not None else "", seat, "-c", CONTAINER, "--", "bash", "-c", script)
    cmd = [c for c in cmd if c]
    p = subprocess.run(cmd, input=input, capture_output=True, timeout=timeout,
                       text=isinstance(input, str) or input is None)
    if check and p.returncode != 0:
        err = p.stderr if isinstance(p.stderr, str) else p.stderr.decode(errors="replace")
        raise RuntimeError(f"{seat}: command failed ({p.returncode}):\n{script}\n{err[-2000:]}")
    return p.stdout


def py(seat, module, args=(), stdin_json=None, python=SYS_PY, timeout=None):
    """Run `python -m module` from APP_DIR; optional JSON on stdin; return stdout."""
    script = f"cd {APP_DIR} && SWORDNET_HOME={DATA_DIR} {python} -m {module} {' '.join(map(shlex.quote, args))}"
    return sh(seat, script, input=json.dumps(stdin_json) if stdin_json is not None else None, timeout=timeout)


def spawn(seat, script, log):
    """Start a long job detached from the exec session; it survives the bridge disconnecting."""
    sh(seat, f"mkdir -p $(dirname {log}) && setsid nohup bash -c {shlex.quote(script)} > {log} 2>&1 < /dev/null &")


def push(seat, local, remote_dir, arcname=None, exclude=()):
    """Copy a local file or directory into remote_dir on the pod."""
    buf = io.BytesIO()
    local = Path(local)
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(local, arcname=arcname or local.name,
                filter=lambda ti: None if any(x in ti.name for x in exclude) else ti)
    sh(seat, f"mkdir -p {remote_dir} && tar xzf - -C {remote_dir}", input=buf.getvalue())


def pull(seat, remote_path, local_dir, exclude=()):
    """Copy a remote file or directory into local_dir. Returns False if it does not exist."""
    remote_path = remote_path.rstrip("/")
    parent, name = os.path.split(remote_path)
    ex = " ".join(f"--exclude={shlex.quote(e)}" for e in exclude)
    cmd = _kubectl("exec", seat, "-c", CONTAINER, "--", "bash", "-c",
                   f"test -e {remote_path} && tar czf - {ex} -C {parent} {name}")
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        return False
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(p.stdout), mode="r:gz") as tar:
        tar.extractall(local_dir, filter="data")
    return True


def read(seat, remote_path):
    return sh(seat, f"cat {remote_path} 2>/dev/null || true").strip()
