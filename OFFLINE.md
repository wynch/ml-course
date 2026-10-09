# Offline use

The course has three offline levels so you can download only what you need.
Preparing a pack requires a connection once; using and verifying it does not.

| Pack | Contains | Best for |
|---|---|---|
| **Reader** | Lessons, figures, source files, search, progress, quizzes, and every browser explorable | Reading and visual experiments on any laptop |
| **Core** | Reader + Python 3.12, locked dependencies, FashionMNIST, the SmolLM3 tokenizer, and modules 01–04, 05½, and 07 | The from-scratch path |
| **Full** | Core + all model snapshots, datasets, and environments used by modules 05–10 | A completely disconnected workshop |

External models and datasets are downloaded from their original repositories
on your machine. They are pinned to commit revisions in
[`offline/manifest.json`](offline/manifest.json); the course does not
redistribute them.

## Reader pack

The reader is published at <https://wynch.github.io/ml-course/reader/>, rebuilt
by CI from this repository on every push to `main`. Open it once and install it,
and every lesson, figure, explorable and quiz is available with the network off.
Everything below builds the same thing locally — `reader/` is a build output and
is not committed.

Prepare the static reader:

```bash
python3 scripts/prefetch_offline.py --pack reader
```

Then disconnect from the network and serve it:

```bash
python3 -m http.server 8000 --directory reader
```

Open <http://localhost:8000>. Progress and quiz answers stay in the browser.
Use **Export JSON** in the top-right progress menu to move them to another
device.

The original files under [`explorables/`](explorables/) also work independently
and contain no remote runtime assets.

## Core and full lab packs

The pack builder uses a portable cache under `.offline-cache/` rather than
assuming your global `uv` or Hugging Face cache will travel with the course.

```bash
# Foundations and from-scratch work
python3 scripts/prefetch_offline.py --pack core

# Every local lab; expect several gigabytes
python3 scripts/prefetch_offline.py --pack full
```

Use `--cache-dir /path/on/an/external/drive` to put the cache elsewhere. The
full pack includes the 1.7B agent model and is intentionally optional.

## Prove it works before leaving

Verification explicitly disables package, Hub, and dataset network access:

```bash
python3 scripts/check_offline.py --pack reader
python3 scripts/check_offline.py --pack core
python3 scripts/check_offline.py --pack full
```

Do not treat “it ran once while online” as an offline test. The verifier creates
or checks environments with `UV_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`.

## Run a lab from the portable cache

Set these variables in the shell that will run the course:

```bash
export UV_CACHE_DIR="$PWD/.offline-cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.offline-cache/python"
export HF_HOME="$PWD/.offline-cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MLCOURSE_OFFLINE=1
```

Then use the normal module commands, adding `--offline --frozen` when invoking
`uv` directly:

```bash
uv run --offline --frozen --project modules/05a-data-evaluation/python pytest -q
```

## What still needs a connection

- Links in the Resources page open the original external source.
- The optional cloud lane, Hub pushes, Spaces, Jobs, and hosted tracking.
- A model or dataset not listed in the selected pack.
- Preparing a pack for a different operating system or CPU architecture.

Python wheels and managed Python builds are platform-specific. Prepare the lab
pack on the same macOS/Apple-silicon family on which it will run. The Reader
pack is platform-independent.

## Updating a pack

Pack revisions are deliberate. To update one:

1. change its commit revision in `offline/manifest.json`;
2. prepare the pack while online;
3. run the network-disabled verifier;
4. regenerate any figures or expected outputs affected by the update; and
5. record the change in the commit message.

This prevents a model, dataset, or dependency update from silently changing a
lesson.
