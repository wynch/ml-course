"""Put the local ``src/`` directory on sys.path so ``import origins`` works
without installing the package. Imported for side effects by every script:

    import _bootstrap  # noqa: F401
"""

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FIGDIR = pathlib.Path(__file__).resolve().parents[2] / "figures"
FIGDIR.mkdir(exist_ok=True)
