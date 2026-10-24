"""The sandbox holds against known escapes, with and without its screen. No model.

Each escape is tried twice: through ``evaluate`` (the screen should stop it)
and through ``run``, which skips the screen, to check that the child's own
walls hold too.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import sandbox
import task

MAC = sandbox.OS_SANDBOX
BEST = task.excess(task.bins_used(task.best_fit, "evolve"), "evolve")
needs_os_sandbox = pytest.mark.skipif(not MAC, reason="candidates run only under macOS sandbox-exec")


def body(lines: str) -> str:
    return "import numpy as np\ndef priority(item, bins):\n" + lines + "    return -(bins - item)\n"


@pytest.fixture()
def secret():
    # A scratch file under the home folder, outside the venv and the child's folder.
    path = Path(tempfile.mkstemp(prefix="mlcourse35-secret-", suffix=".txt",
                                 dir=os.path.expanduser("~"))[1])
    path.write_text("7 7 7\n")
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture()
def outside(tmp_path_factory):
    # A folder that is not the child's temporary folder.
    d = Path(tempfile.mkdtemp(prefix="mlcourse35-outside-", dir=os.path.expanduser("~")))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@needs_os_sandbox
def test_os_system_through_numpy(outside):
    marker = outside / "pwned"
    code = body(f"    np.lib._datasource.os.system('touch {marker}')\n")
    assert sandbox.evaluate(code)["status"] == "screen"
    out = sandbox.run(code)
    assert not marker.exists()
    assert out["status"] in ("valid", "crash")   # system() fails under the profile


@needs_os_sandbox
def test_savetxt_outside_the_folder(outside):
    target = outside / "written.txt"
    code = body(f"    np.savetxt('{target}', bins)\n")
    assert sandbox.evaluate(code)["status"] == "screen"
    out = sandbox.run(code)
    assert not target.exists()
    assert out["status"] == "crash"


@needs_os_sandbox
def test_datasource_cannot_reach_the_network():
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"1 2 3")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/x.txt"
    code = body(f"    np.lib.npyio.DataSource(None).open('{url}').read()\n")
    try:
        assert sandbox.evaluate(code)["status"] == "screen"
        out = sandbox.run(code)
    finally:
        server.shutdown()
    assert hits == [] and out["status"] == "crash"


@needs_os_sandbox
def test_patching_argmax_does_not_reach_the_packer():
    # Inside the child this makes argmax always pick the first bin; the
    # packer runs in the parent, so the candidate still packs as best fit.
    code = ("import numpy as np\n"
            "np.argmax = lambda a, *k, **w: 0\n"
            "np.isclose = lambda *a, **k: True\n"
            "def priority(item, bins):\n    return -(bins - item)\n")
    out = sandbox.evaluate(code)
    assert out["status"] == "valid"
    assert task.excess(out["bins"]["evolve"], "evolve") == pytest.approx(BEST)


@needs_os_sandbox
def test_fake_result_line_is_a_crash():
    fake = '{"status": "valid", "error": null, "bins": {"evolve": [1, 1, 1, 1]}}'
    code = body(f"    np.lib._datasource.os.write(1, b'{fake}\\n')\n"
                "    np.lib._datasource.os._exit(0)\n")
    assert sandbox.evaluate(code)["status"] == "screen"
    out = sandbox.run(code)
    assert out["status"] == "crash" and out["bins"] == {}


@needs_os_sandbox
def test_large_allocation_is_stopped():
    code = body("    x = np.ones(1 << 31)\n    x[::4096] = 2.0\n")
    out = sandbox.evaluate(code)
    assert out["status"] in ("crash", "timeout") and out["bins"] == {}


@needs_os_sandbox
def test_child_writes_only_inside_its_folder():
    # Writing in the working directory is allowed and leaves nothing behind.
    code = body("    np.save('here.npy', bins)\n")
    out = sandbox.run(code)
    assert out["status"] == "valid"


@pytest.mark.parametrize("code", [
    body("    import os\n"),
    body("    f = open('x')\n"),
    body("    t = bins.__class__\n"),
    body("    s = np.ctypeslib\n"),
    body("    m = np.memmap\n"),
    "from numpy import load\ndef priority(item, bins):\n    return -(bins - item)\n",
    "import numpy.lib as l\ndef priority(item, bins):\n    return -(bins - item)\n",
])
def test_screen_rejects(code):
    assert sandbox.evaluate(code)["status"] == "screen"


@needs_os_sandbox
def test_a_local_name_called_load_is_fine():
    code = "import numpy as np\ndef priority(item, bins):\n    load = bins - item\n    return -load\n"
    out = sandbox.evaluate(code)
    assert out["status"] == "valid"
    assert task.excess(out["bins"]["evolve"], "evolve") == pytest.approx(BEST)


def test_wildcard_import_is_rejected(secret):
    code = ("from numpy import *\n"
            "def priority(item, bins):\n"
            f"    return -(bins - item) + loadtxt('{secret}').sum()\n")
    assert sandbox.evaluate(code)["status"] == "screen"


@pytest.mark.parametrize("name", ["loadtxt", "fromfile", "savetxt", "DataSource", "memmap"])
def test_bare_io_names_are_rejected(name):
    code = f"import numpy as np\ndef priority(item, bins):\n    f = {name}\n    return -(bins - item)\n"
    assert sandbox.evaluate(code)["status"] == "screen"


@needs_os_sandbox
def test_reads_outside_the_venv_fail(secret):
    # Around the screen: the file's contents would leak through the scores.
    code = ("from numpy import *\n"
            "def priority(item, bins):\n"
            f"    return -(bins - item) + loadtxt('{secret}').sum()\n")
    out = sandbox.run(code)
    assert out["status"] == "crash" and "not permitted" in out["error"]
    zshrc = os.path.expanduser("~/.zshrc")
    code = body(f"    v = np.fromfile('{zshrc}', dtype=np.uint8)\n")
    out = sandbox.run(code)
    assert out["status"] == "crash"


def test_fails_closed_without_an_os_sandbox(monkeypatch):
    monkeypatch.setattr(sandbox, "OS_SANDBOX", False)
    monkeypatch.setattr(sandbox, "UNSAFE", False)
    with pytest.raises(RuntimeError, match="unsafe-no-os-sandbox"):
        sandbox.evaluate(task.BEST_FIT)


def test_unsafe_flag_runs_with_a_warning(monkeypatch, capfd):
    monkeypatch.setattr(sandbox, "OS_SANDBOX", False)
    monkeypatch.setattr(sandbox, "UNSAFE", True)
    out = sandbox.evaluate(task.BEST_FIT)
    assert out["status"] == "valid"
    assert "no OS sandbox" in capfd.readouterr().err
