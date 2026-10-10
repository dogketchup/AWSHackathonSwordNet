"""kubectl helpers. The workshop role only allows `get pods` and `pods/exec`, so everything
here is built on exec: running commands, and moving files as tar archives over stdin/stdout.

This cluster cuts exec streams off at random (and `kubectl cp` is the same mechanism), so files
move in verified pieces; see send_bytes / fetch_bytes.
"""
import hashlib
import io
import json
import os
import shlex
import subprocess
import tarfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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


# Measured on this cluster: one exec stream moves ~1 MB/s and is cut off at random, sometimes
# silently (exit 0, fewer bytes). So files move as 1 MB pieces, several at once, and every piece
# is checked against its own SHA-256 and retried until it matches.
CHUNK = 1024 * 1024
PARALLEL = 6
RETRIES = 6


def _exec_bytes(seat, script, input=None, timeout=120):
    """One exec call returning raw stdout bytes; raises on a non-zero exit or timeout."""
    cmd = _kubectl("exec", *(["-i"] if input is not None else []), seat, "-c", CONTAINER, "--", "bash", "-c", script)
    try:
        p = subprocess.run(cmd, input=input, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("timed out")
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace")[-300:])
    return p.stdout


def _retry(fn, what):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except RuntimeError as e:
            if attempt == RETRIES:
                raise RuntimeError(f"{what} failed {RETRIES} times: {e}")
            time.sleep(attempt)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def send_bytes(seat, data, remote_file):
    """Upload bytes to remote_file as verified 1 MB pieces, PARALLEL at a time."""
    parts = f"{remote_file}.parts"
    sh(seat, f"rm -rf {parts} && mkdir -p {parts} $(dirname {remote_file})")
    chunks = [data[o:o + CHUNK] for o in range(0, len(data), CHUNK)] or [b""]

    def put(i):
        def once():
            out = _exec_bytes(seat, f"cat > {parts}/{i:06d} && sha256sum {parts}/{i:06d} | cut -c1-64",
                              input=chunks[i]).decode().strip()
            if out != _sha(chunks[i]):
                raise RuntimeError(f"piece {i} arrived corrupted")
        _retry(once, f"{seat}: upload piece {i}/{len(chunks)}")

    with ThreadPoolExecutor(PARALLEL) as pool:
        list(pool.map(put, range(len(chunks))))
    got = sh(seat, f"cat {parts}/* > {remote_file} && rm -rf {parts} && sha256sum {remote_file} | cut -c1-64").strip()
    if got != _sha(data):
        raise RuntimeError(f"{seat}: upload of {remote_file} corrupted after reassembly")


def fetch_bytes(seat, remote_file):
    """Download remote_file as verified 1 MB pieces, PARALLEL at a time."""
    parts = f"{remote_file}.parts"
    listing = sh(seat, f"""rm -rf {parts} && mkdir -p {parts} && cd {parts} && split -a 6 -d -b {CHUNK} {remote_file} p
                          [ -n "$(ls)" ] || : > p000000
                          sha256sum {remote_file} | cut -c1-64; sha256sum p* """).split("\n")
    want_all = listing[0].strip()
    want = {}
    for line in listing[1:]:
        if line.strip():
            h, name = line.split()
            want[name] = h

    def get(name):
        def once():
            b = _exec_bytes(seat, f"cat {parts}/{name}")
            if _sha(b) != want[name]:
                raise RuntimeError(f"{name} came back truncated ({len(b)} bytes)")
            return b
        return _retry(once, f"{seat}: download {name} of {len(want)}")

    try:
        with ThreadPoolExecutor(PARALLEL) as pool:
            data = b"".join(pool.map(get, sorted(want)))
    finally:
        sh(seat, f"rm -rf {parts}", check=False)
    if _sha(data) != want_all:
        raise RuntimeError(f"{seat}: download of {remote_file} corrupted after reassembly")
    return data


def push(seat, local, remote_dir, arcname=None, exclude=()):
    """Copy a local file or directory into remote_dir on the pod."""
    buf = io.BytesIO()
    local = Path(local)
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(local, arcname=arcname or local.name,
                filter=lambda ti: None if any(x in ti.name for x in exclude) else ti)
    tmp = f"/tmp/swordnet-xfer-{uuid.uuid4().hex[:8]}.tgz"
    send_bytes(seat, buf.getvalue(), tmp)
    sh(seat, f"mkdir -p {remote_dir} && tar xzf {tmp} -C {remote_dir}; rc=$?; rm -f {tmp}; exit $rc")


def pull(seat, remote_path, local_dir, exclude=()):
    """Copy a remote file or directory into local_dir. Returns False if it does not exist."""
    remote_path = remote_path.rstrip("/")
    parent, name = os.path.split(remote_path)
    ex = " ".join(f"--exclude={shlex.quote(e)}" for e in exclude)
    tmp = f"/tmp/swordnet-xfer-{uuid.uuid4().hex[:8]}.tgz"
    made = sh(seat, f"test -e {remote_path} && tar czf {tmp} {ex} -C {parent} {name} && echo ok || true").strip()
    if made != "ok":
        return False
    try:
        data = fetch_bytes(seat, tmp)
    finally:
        sh(seat, f"rm -f {tmp}", check=False)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(local_dir, filter="data")
    return True


def running(seat, pattern):
    """Is a process matching the regex `pattern` alive in the pod?

    The first character of each alternative goes in brackets ("[s]wordnet...|[/]workspace..."), so
    the pattern no longer matches the text of this very command, which bash -c would otherwise
    report as a hit.
    """
    guarded = "|".join(f"[{alt[0]}]{alt[1:]}" for alt in pattern.split("|"))
    return sh(seat, f"pgrep -f {shlex.quote(guarded)} >/dev/null && echo y || true", check=False).strip() == "y"


def read(seat, remote_path):
    return sh(seat, f"cat {remote_path} 2>/dev/null || true").strip()
