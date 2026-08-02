# Setup

This course targets an **Apple-silicon Mac**. Everything below runs locally.
Work through this page once before module 01; each check should print a version
or confirmation. If a command is missing, install the tool from the link in its
section, then re-run the check.

The reference toolchain this course was built against:

- **uv** 0.11.28
- **Zig** 0.16.0
- **`hf` CLI** 1.24.0 (logged in)

---

## Open the course reader

The dependency-free reader is the easiest way to navigate the material. It
includes full-text search, local progress, quizzes, and all interactive
explorables. Build it with Python's standard library and serve it locally:

```bash
python3 scripts/build_reader.py
python3 -m http.server 8000 --directory reader
```

Then open <http://localhost:8000>. After the first visit, the installed web app
can reopen the reader without a connection. To prepare model and dataset caches
too, follow [the offline pack guide](OFFLINE.md).

---

## Verify your toolchain

**[uv](https://docs.astral.sh/uv/)** — Python environment and script runner.

```bash
uv --version
```

**[Zig](https://ziglang.org/download/)** — the from-scratch systems lane.

```bash
zig version
```

**[`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)** — Hugging Face Hub client.

```bash
hf version
```

Confirm you are logged in to the Hub (needed for `datasets`, model downloads,
and the cloud lane):

```bash
hf auth whoami
```

If `hf auth whoami` reports you are not logged in, run `hf auth login` and paste
a token from <https://huggingface.co/settings/tokens>.

---

## How uv is used per module

Each module is its **own uv project** — its dependencies are pinned in that
module's `pyproject.toml`, isolated from every other module. You never manage a
global environment and never activate a virtualenv by hand.

Run a script; uv resolves and installs the module's dependencies on first use,
then executes:

```bash
uv run python/train.py
```

Materialize or refresh a module's environment explicitly (optional — `uv run`
does this for you):

```bash
uv sync
```

Run commands from **inside the module directory** (e.g. `modules/02-neural-networks`)
so uv picks up that module's project file.

---

## Optional installs for later modules

You do **not** need these for module 01. Install them when a module asks.

**PyTorch** (modules 04+), added to a module's uv project:

```bash
uv add torch
```

**mlx-lm** — Apple-silicon training/inference lane used in module 06:

```bash
uv add mlx-lm
```

**graphviz `dot`** — optional for module 01, used only to render the autograd
computation graph as a diagram. The module works without it (it falls back to a
text dump); install it if you want the picture:

```bash
brew install graphviz
```
