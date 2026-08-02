#!/usr/bin/env python3
"""Prepare deterministic reader, core, or full offline course packs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "offline" / "manifest.json"
PACK_ORDER = {"reader": 0, "core": 1, "full": 2}


def run(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download everything needed for a network-free course session."
    )
    parser.add_argument("--pack", choices=PACK_ORDER, default="core")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".offline-cache",
        help="Portable cache directory (default: .offline-cache)",
    )
    parser.add_argument("--skip-reader", action="store_true")
    parser.add_argument("--archive-reader", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())

    cache = args.cache_dir.expanduser().resolve()
    env = os.environ.copy()
    env.update(
        {
            "UV_CACHE_DIR": str(cache / "uv"),
            "UV_PYTHON_INSTALL_DIR": str(cache / "python"),
            "HF_HOME": str(cache / "huggingface"),
            "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )

    if not args.skip_reader:
        run(
            [sys.executable, str(ROOT / "scripts" / "build_reader.py")],
            env=env,
            dry_run=args.dry_run,
        )

    if args.archive_reader and not args.dry_run:
        archive_root = cache / "packs"
        archive_root.mkdir(parents=True, exist_ok=True)
        path = shutil.make_archive(
            str(archive_root / "ml-course-reader"),
            "zip",
            root_dir=ROOT / "reader",
        )
        print(f"reader archive: {path}")

    if args.pack == "reader":
        print("Reader pack ready. No Python, model, or dataset downloads requested.")
        return

    if not shutil.which("uv") or not shutil.which("hf"):
        raise SystemExit("uv and hf must be installed before preparing lab packs")

    run(["uv", "python", "install", manifest["python"]], env=env, dry_run=args.dry_run)
    projects = manifest["packs"][args.pack]["projects"]
    for project in projects:
        run(
            ["uv", "sync", "--locked", "--project", project],
            env=env,
            dry_run=args.dry_run,
        )

    selected_assets = [
        asset
        for asset in manifest["assets"]
        if PACK_ORDER[asset["pack"]] <= PACK_ORDER[args.pack]
    ]
    for asset in selected_assets:
        if asset["kind"] == "model":
            command = [
                "hf",
                "download",
                asset["id"],
                "--revision",
                asset["revision"],
                "--cache-dir",
                str(cache / "huggingface" / "hub"),
            ]
            for pattern in asset.get("include", []):
                command.extend(["--include", pattern])
            run(command, env=env, dry_run=args.dry_run)
        else:
            command = [
                "uv",
                "run",
                "--frozen",
                "--project",
                asset["project"],
                "python",
                str(ROOT / "scripts" / "cache_dataset.py"),
                asset["id"],
                "--revision",
                asset["revision"],
            ]
            if asset.get("config"):
                command.extend(["--config", asset["config"]])
            if asset.get("split"):
                command.extend(["--split", asset["split"]])
            run(command, env=env, dry_run=args.dry_run)

    print(f"\n{args.pack.title()} offline pack ready under {cache}")
    print("Verify it with: python3 scripts/check_offline.py --pack", args.pack)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"\nPreparation stopped: command exited {error.returncode}", file=sys.stderr)
        raise SystemExit(error.returncode)
