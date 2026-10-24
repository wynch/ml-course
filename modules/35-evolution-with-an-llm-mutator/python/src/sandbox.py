"""Run one candidate heuristic in isolation and score it.

The candidate never touches the packer or the checker. The parent process
packs every instance with ``task.pack`` and checks it with
``task.check_packing``; the only thing it asks the candidate is the value of
``priority(item, bins)``, one call at a time, over a pipe. So a candidate
cannot patch ``np.argmax`` or ``np.isclose`` in the parent, and cannot print
a fake result: anything but a well-formed reply is a crash.

Layers, cheapest first:

1. ``screen``: parse the code; it must define ``priority(item, bins)``; only
   ``numpy`` and ``math`` may be imported; no name or attribute that starts
   with ``_``; no ``open``, ``eval``, ``exec`` and the like; no numpy I/O or
   foreign-function names (``lib``, ``ctypeslib``, ``save``, ``load``,
   ``fromfile``, ``memmap``, ``DataSource``, ...). A screen failure costs no
   process.
2. The child: a fresh ``python -I -B`` with few builtins, a CPU-time limit,
   a data-size limit, an empty temporary folder as its working directory and
   no environment. It runs under macOS's ``sandbox-exec`` with a profile
   that denies the network, new processes, file writes outside that folder,
   and file reads outside the Python install, the venv, that folder and this
   file. Elsewhere the module refuses to run candidates, unless the script
   is given ``--unsafe-no-os-sandbox``.
3. The parent: a wall-clock deadline, and a resident-memory cap it enforces
   by watching the child and killing it on breach (macOS ignores most
   memory rlimits).
4. The packer and the checker, in the parent: wrong shapes and NaN are
   ``invalid``; every packing is checked.

This is defence in depth for code a local model wrote. It is not a proof
against a determined attacker: run untrusted code from elsewhere in a
virtual machine.
"""

from __future__ import annotations

import ast
import hashlib
import os
import platform
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKER = Path(__file__).resolve()
TIMEOUT_S = 10.0          # wall clock for one candidate on all its instances
CPU_LIMIT_S = 6           # RLIMIT_CPU inside the child
MEMORY_LIMIT = 1 << 30    # bytes: RLIMIT_DATA in the child, and the RSS cap the parent enforces
RSS_POLL_S = 0.25         # how often the parent reads the child's resident size

ALLOWED_IMPORTS = {"numpy", "math"}
BANNED_NAMES = {
    "open", "eval", "exec", "compile", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
    "memoryview", "os", "sys", "subprocess", "importlib", "builtins",
    # numpy's I/O and ctypes functions under their own names, in case one is
    # ever bound without an attribute (a wildcard import is refused anyway)
    "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile", "fromregex", "memmap",
    "DataSource", "ctypeslib", "ctypes", "savez", "savez_compressed", "fromstring",
    "frombuffer", "f2py", "recfromtxt", "recfromcsv",
}
# numpy's file, network and foreign-function doors, banned as attributes and
# as names imported from numpy (a local variable may still be called "load").
BANNED_ATTRS = BANNED_NAMES | {
    "lib", "ctypeslib", "ctypes", "save", "savez", "savez_compressed", "load", "loadtxt",
    "savetxt", "genfromtxt", "fromfile", "tofile", "fromregex", "memmap", "DataSource",
    "f2py", "testing", "distutils", "dump", "dumps", "tostring", "fromstring", "frombuffer",
    "show_config", "show_runtime", "info", "source", "who", "lookfor",
}
SAFE_BUILTINS = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
    "isinstance", "len", "list", "map", "max", "min", "pow", "range", "reversed",
    "round", "set", "sorted", "sum", "tuple", "zip", "ValueError", "ZeroDivisionError",
]

UNSAFE_FLAG = "--unsafe-no-os-sandbox"
UNSAFE_ENV = "MLCOURSE35_UNSAFE_NO_OS_SANDBOX"
OS_SANDBOX = platform.system() == "Darwin" and shutil.which("sandbox-exec") is not None
# The flag is taken out of argv here, before any script parses its own
# arguments, and passed on to child processes (OpenEvolve's evaluator)
# through the environment.
UNSAFE = UNSAFE_FLAG in sys.argv or os.environ.get(UNSAFE_ENV) == "1"
if UNSAFE_FLAG in sys.argv:
    sys.argv.remove(UNSAFE_FLAG)
    os.environ[UNSAFE_ENV] = "1"


def profile(tmp: str) -> str:
    """The sandbox-exec profile for one child.

    Found by trial on macOS 26 with uv's Python: the child starts and loads
    numpy with read access to the folder of Python installs, the venv, its
    own temporary folder, this file, and the root directory itself (without
    "/" the interpreter aborts at start). File metadata (names, sizes) stays
    readable everywhere; file contents do not.
    """
    reads = {os.path.dirname(os.path.realpath(sys.base_prefix)), sys.base_prefix,
             os.path.realpath(sys.prefix), sys.prefix, tmp}
    rules = "".join(f'(allow file-read* (subpath "{r}"))\n' for r in sorted(reads))
    return ("(version 1)\n(allow default)\n(deny network*)\n(deny process-fork)\n"
            "(deny file-write*)\n"
            f'(allow file-write* (subpath "{tmp}") (literal "/dev/null"))\n'
            "(deny file-read*)\n(allow file-read-metadata)\n" + rules +
            f'(allow file-read* (literal "/") (literal "{WORKER}") (literal "/dev/null"))\n')


class ScreenError(Exception):
    pass


class CandidateCrash(Exception):
    pass


class CandidateTimeout(Exception):
    pass


def screen(code: str) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise ScreenError(f"syntax: {error.msg}") from None
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if "priority" not in names:
        raise ScreenError("no function named priority")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS or "." in alias.name:
                    raise ScreenError(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_IMPORTS:
                raise ScreenError(f"import from {node.module}")
            for alias in node.names:
                if alias.name == "*" or alias.name.startswith("_") or alias.name in BANNED_ATTRS:
                    raise ScreenError(f"import {alias.name}")
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("_")):
            raise ScreenError(f"name {node.id}")
        elif isinstance(node, ast.Attribute) and (node.attr in BANNED_ATTRS or node.attr.startswith("_")):
            raise ScreenError(f"attribute {node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            raise ScreenError(type(node).__name__)
    return tree


def normalize(code: str) -> str:
    """The code with comments, docstrings and spacing removed, for dedupe."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Module)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def code_hash(code: str) -> str:
    return hashlib.sha256(normalize(code).encode()).hexdigest()[:16]


# ---- the parent's side of the pipe ---------------------------------------------------------

def _limits() -> None:
    """In the child, before Python starts: CPU and data-size limits."""
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_S, CPU_LIMIT_S))
    for name in ("RLIMIT_DATA", "RLIMIT_AS"):
        try:
            resource.setrlimit(getattr(resource, name), (MEMORY_LIMIT, MEMORY_LIMIT))
        except (ValueError, OSError, AttributeError):
            pass                      # macOS refuses some of these; the RSS watch remains


def _rss(pid: int) -> int:
    """Resident size of a process in bytes, 0 if it is gone."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=2).stdout.strip()
        return int(out) * 1024 if out else 0
    except (subprocess.SubprocessError, ValueError):
        return 0


class Child:
    """One sandboxed candidate, answering priority calls until closed."""

    def __init__(self, code: str):
        self.tmp = tempfile.TemporaryDirectory(prefix="mlcourse35-")
        command = [sys.executable, "-I", "-B", str(WORKER), "--worker"]
        if OS_SANDBOX:
            tmp = os.path.realpath(self.tmp.name)
            command = ["sandbox-exec", "-p", profile(tmp)] + command
        elif not UNSAFE:
            self.tmp.cleanup()
            raise RuntimeError(
                "No OS sandbox: candidates run only under macOS's sandbox-exec. To run "
                f"them anyway, without the file, network and process walls, add {UNSAFE_FLAG}.")
        else:
            print(f"warning: {UNSAFE_FLAG}: candidates run with no OS sandbox; they can read "
                  "your files and use the network", file=sys.stderr)
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, cwd=self.tmp.name, env={},
                                     preexec_fn=_limits)
        self.deadline = time.monotonic() + TIMEOUT_S
        self.next_poll = 0.0
        self.peak_rss = 0
        body = code.encode()
        self._write(struct.pack("<I", len(body)) + body)
        tag = self._read(1)
        if tag != b"R":
            raise CandidateCrash(self._message() if tag == b"E" else "no reply after loading")

    def _guard(self) -> None:
        now = time.monotonic()
        if now > self.deadline:
            raise CandidateTimeout(f"over {TIMEOUT_S:.0f} s")
        if now >= self.next_poll:
            self.next_poll = now + RSS_POLL_S
            rss = _rss(self.proc.pid)
            self.peak_rss = max(self.peak_rss, rss)
            if rss > MEMORY_LIMIT:
                raise CandidateCrash(f"memory over {MEMORY_LIMIT >> 20} MB")

    def _write(self, data: bytes) -> None:
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise CandidateCrash(self._died()) from None

    def _read(self, n: int) -> bytes:
        fd = self.proc.stdout.fileno()
        chunks, got = [], 0
        while got < n:
            self._guard()
            ready, _, _ = select.select([fd], [], [], RSS_POLL_S)
            if not ready:
                continue
            part = os.read(fd, n - got)
            if not part:
                raise CandidateCrash(self._died())
            chunks.append(part)
            got += len(part)
        return b"".join(chunks)

    def _died(self) -> str:
        code = self.proc.poll()
        if code is None:
            try:
                code = self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                return "child stopped answering"
        if code in (-24, -9):          # SIGXCPU or SIGKILL
            raise CandidateTimeout(f"CPU limit {CPU_LIMIT_S} s")
        return f"child exited with code {code}"

    def _message(self) -> str:
        (n,) = struct.unpack("<I", self._read(4))
        return self._read(n).decode("utf-8", "replace")[:200]

    def priority(self, item: float, bins):
        import numpy as np

        bins = np.ascontiguousarray(bins, dtype="<f8")
        self._write(struct.pack("<dI", float(item), len(bins)) + bins.tobytes())
        tag = self._read(1)
        if tag == b"S":
            (k,) = struct.unpack("<I", self._read(4))
            return np.frombuffer(self._read(8 * k), dtype="<f8").copy()
        if tag == b"B":                 # a reply of the wrong shape
            import task

            raise task.InvalidHeuristic(f"bad scores: shape {self._message()}, want ({len(bins)},)")
        if tag == b"E":
            raise CandidateCrash(self._message())
        raise CandidateCrash("malformed reply")

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            pass
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except OSError:
                pass
        self.tmp.cleanup()


def run(code: str, sets=("evolve",)) -> dict:
    """Score code on the named sets without the screen. Tests call this to
    check that the child's own walls hold when the screen is bypassed."""
    import numpy as np

    import task

    t0 = time.perf_counter()
    out = {"status": "valid", "error": None, "bins": {}}
    child = None
    try:
        child = Child(code)
        with np.errstate(all="ignore"):
            for name in sets:
                out["bins"][name] = task.bins_used(child.priority, name)
    except task.InvalidHeuristic as error:
        out = {"status": "invalid", "error": str(error)[:200], "bins": {}}
    except CandidateTimeout as error:
        out = {"status": "timeout", "error": str(error), "bins": {}}
    except CandidateCrash as error:
        out = {"status": "crash", "error": str(error)[:200], "bins": {}}
    finally:
        if child is not None:
            child.close()
    out["seconds"] = round(time.perf_counter() - t0, 3)
    return out


def evaluate(code: str, sets=("evolve",)) -> dict:
    """Score ``code`` on the named instance sets.

    Returns {"status": "valid" | "screen" | "crash" | "timeout" | "invalid",
    "error": str | None, "bins": {set: [bins per instance]}, "seconds": float}.
    """
    try:
        screen(code)
    except ScreenError as error:
        return {"status": "screen", "error": str(error), "bins": {}, "seconds": 0.0}
    return run(code, sets)


# ---- the child's side ----------------------------------------------------------------------

def _worker() -> None:
    """Inside the child: load the candidate, then answer priority calls."""
    import builtins
    import math

    import numpy as np

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    # The candidate gets no handle on the pipe: replies go through these
    # closures only, and print() is not among its builtins.
    sys.stdout = sys.stderr = open(os.devnull, "w")

    def read(n: int) -> bytes:
        data = stdin.read(n)
        if len(data) != n:
            raise SystemExit(0)
        return data

    def reply(tag: bytes, payload: bytes = b"") -> None:
        stdout.write(tag + payload)
        stdout.flush()

    def message(tag: bytes, text: str) -> None:
        body = text.encode("utf-8", "replace")[:400]
        reply(tag, struct.pack("<I", len(body)) + body)

    real_import = builtins.__import__

    # numpy's C code imports its own submodules (numpy._core._methods for
    # .mean()) through the calling frame's builtins, so submodules of an
    # allowed package must pass here; the screen keeps the candidate's own
    # import statements to the top-level names.
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    safe["__import__"] = guarded_import
    env = {"__builtins__": safe, "np": np, "numpy": np, "math": math}
    (n,) = struct.unpack("<I", read(4))
    try:
        exec(compile(read(n).decode(), "<candidate>", "exec"), env)
        priority = env["priority"]
    except Exception as error:  # noqa: BLE001 - any failure of the candidate is data
        message(b"E", f"{type(error).__name__}: {error}")
        return
    reply(b"R")
    np.seterr(all="ignore")
    while True:
        item, k = struct.unpack("<dI", read(12))
        bins = np.frombuffer(read(8 * k), dtype="<f8").copy()
        try:
            scores = np.asarray(priority(item, bins), dtype=np.float64)
        except Exception as error:  # noqa: BLE001
            message(b"E", f"{type(error).__name__}: {error}")
            return
        if scores.shape != (k,):
            message(b"B", str(scores.shape))
            continue
        reply(b"S", struct.pack("<I", k) + scores.astype("<f8").tobytes())


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
