#!/usr/bin/env python3
"""Verify the reader and selected labs with network access disabled."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


class DependencyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.remote_dependencies: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        dependency = None
        if tag in {"script", "img", "audio", "video", "source"}:
            dependency = values.get("src")
        elif tag == "link" and values.get("rel") != "canonical":
            dependency = values.get("href")
        if dependency and dependency.startswith(("http://", "https://", "//")):
            self.remote_dependencies.append(dependency)


def check_reader() -> None:
    reader = ROOT / "reader"
    required = [
        reader / "index.html",
        reader / "sw.js",
        reader / "offline-assets.json",
        reader / "explorables" / "01-gradient-descent.html",
        reader / "explorables" / "05a-evaluation-lab.html",
        reader / "content" / "modules" / "10-agents" / "README.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing reader artifacts: " + ", ".join(missing))

    assets = json.loads((reader / "offline-assets.json").read_text())
    if len(assets) < 250:
        raise RuntimeError(f"offline asset list is unexpectedly small ({len(assets)})")

    for html in (reader / "explorables").glob("*.html"):
        parser = DependencyParser()
        parser.feed(html.read_text())
        if parser.remote_dependencies:
            raise RuntimeError(
                f"{html.name} loads remote runtime assets: {parser.remote_dependencies}"
            )
    print(f"reader: PASS ({len(assets)} assets, no remote explorable dependencies)")


def run_offline(
    command: list[str], env: dict[str, str], *, cwd: Path = ROOT
) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=["reader", "core", "full"], default="reader")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".offline-cache")
    args = parser.parse_args()

    check_reader()
    if args.pack == "reader":
        return

    cache = args.cache_dir.expanduser().resolve()
    env = os.environ.copy()
    env.update(
        {
            "UV_CACHE_DIR": str(cache / "uv"),
            "UV_PYTHON_INSTALL_DIR": str(cache / "python"),
            "UV_OFFLINE": "1",
            "HF_HOME": str(cache / "huggingface"),
            "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "MLCOURSE_OFFLINE": "1",
        }
    )

    manifest = json.loads((ROOT / "offline" / "manifest.json").read_text())
    for project in manifest["packs"][args.pack]["projects"]:
        run_offline(["uv", "sync", "--locked", "--offline", "--project", project], env)

    run_offline(
        [
            "uv",
            "run",
            "--offline",
            "--frozen",
            "pytest",
            "-q",
        ],
        env,
        cwd=ROOT / "modules/05a-data-evaluation/python",
    )
    run_offline(
        [
            "uv",
            "run",
            "--offline",
            "--frozen",
            "pytest",
            "-q",
        ],
        env,
        cwd=ROOT / "modules/07-inference-internals/python",
    )
    if args.pack == "full":
        run_offline(
            [
                "uv",
                "run",
                "--offline",
                "--frozen",
                "pytest",
                "-q",
            ],
            env,
            cwd=ROOT / "modules/10-agents/python",
        )
    print(f"{args.pack} pack: PASS with network disabled")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"offline verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
