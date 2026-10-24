#!/usr/bin/env python3
"""Verify the reader and selected labs with network access disabled."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STRAY_MODULE_LINK = re.compile(r'href="(?:\.\./)*/?modules/[^"]*"')
# GitHub Pages serves the reader from /ml-course/reader/, so a root-absolute
# URL escapes the reader and 404s. Every internal reference must be relative.
ROOT_ABSOLUTE_URL = re.compile(r'(?:href|src)="(/(?!/)[^"]*)"')
PAGE_REFERENCE = re.compile(r'(?:href|src)="([^"]*)"')
MARKDOWN_IMAGE = re.compile(r'!\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
MARKDOWN_LINK = re.compile(r'(?<!!)\[[^\]]+\]\(([^)]+)\)')
# Kept in step with VIEWABLE in build_reader.py and app.js.
VIEWABLE = re.compile(r"\.(cfg|csv|ini|json|lock|md|py|sh|toml|txt|ya?ml|zig|zon)$", re.I)
ASSET_ROOTS = ("explorables", "quizzes", "research", "moves")


def normalize(path: str) -> str:
    parts: list[str] = []
    for part in path.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def resolves(reader: Path, target: str) -> bool:
    """True when a reader-relative path is something a browser can open."""
    if not target:
        return True
    destination = reader / target
    if destination.is_dir():
        return (destination / "index.html").is_file()
    return destination.is_file()


def check_links(reader: Path) -> int:
    """Every link the reader can present must land on a file or a folder page.

    This replays what app.js does with a markdown link and what the browser does
    with an href in a bundled page, then resolves the result on disk. It is the
    check that folder links need: a link to exercises/ is only alive because
    build_reader.py writes an index.html into it.
    """
    payload = json.loads(
        (reader / "course-content.js").read_text().split("=", 1)[1].rstrip().rstrip(";")
    )
    pages = payload["guides"] + payload["modules"]
    dead: list[str] = []
    checked = 0

    def page_for(path: str):
        for candidate in pages:
            if candidate["path"] in (path, f"{path.rstrip('/')}/README.md"):
                return candidate
        return None

    for page in pages:
        base = "/".join(page["path"].split("/")[:-1])
        if page.get("explorable"):
            checked += 1
            if not resolves(reader, page["explorable"]):
                dead.append(f"{page['id']} explorable -> {page['explorable']}")
        for source in MARKDOWN_IMAGE.findall(page["markdown"]):
            if re.match(r"^(https?:|data:|/)", source):
                continue
            target = f"content/{normalize(f'{base}/{source}')}"
            checked += 1
            if not resolves(reader, target):
                dead.append(f"{page['id']} image {source} -> {target}")
        for raw in MARKDOWN_LINK.findall(page["markdown"]):
            href = raw.strip().strip("<>")
            if re.match(r"^(https?:|mailto:|#)", href):
                continue
            resolved = normalize(f"{base}/{href.split('#')[0]}")
            if page_for(resolved):
                continue  # an in-app lesson route, not a file
            bundled = any(
                resolved == root or resolved.startswith(f"{root}/")
                for root in ASSET_ROOTS
            )
            if bundled:
                target = resolved
            else:
                target = f"content/{resolved}"
                if VIEWABLE.search(resolved):
                    target += ".html"
            checked += 1
            if not resolves(reader, target):
                dead.append(f"{page['id']} link {href} -> {target}")

    for html in sorted(reader.rglob("*.html")):
        directory = html.parent.relative_to(reader).as_posix()
        for raw in PAGE_REFERENCE.findall(html.read_text()):
            href = raw.split("#")[0].split("?")[0]
            if not href or re.match(r"^(https?:|mailto:|data:|//)", href):
                continue
            if re.search(r"['\"]\s*\+|\$\{", raw) or "\n" in raw:
                continue  # a JavaScript-built href, not a literal one
            checked += 1
            if not resolves(reader, normalize(f"{directory}/{href}")):
                dead.append(f"{html.relative_to(reader)} -> {href}")

    if dead:
        raise RuntimeError(
            f"{len(dead)} reader links resolve to nothing: {dead[:5]}"
        )
    return checked


# The service worker warms offline-assets.json in this order; explorables,
# quizzes, research and moves must come straight after the shell.
OFFLINE_GROUPS = ["core", "explorables", "quizzes", "research", "moves", "lessons", "rest"]
OFFLINE_BLOB = re.compile(r"\.(bin|ckpt|gguf|npz|pt|pth|safetensors)$|(^|/)uv\.lock$|\.lock$")
# What a phone has to hold; the old list was 127 MB.
OFFLINE_MAX_MB = 100


def check_offline_list(reader: Path) -> list[str]:
    """The offline list is small, in priority order, and saves every page that matters."""
    manifest = json.loads((reader / "offline-assets.json").read_text())
    if not isinstance(manifest, dict) or "groups" not in manifest:
        raise RuntimeError("offline-assets.json must hold priority groups")
    order = [group["id"] for group in manifest["groups"]]
    if order != OFFLINE_GROUPS:
        raise RuntimeError(f"offline groups out of order: {order}")
    groups = {group["id"]: group["urls"] for group in manifest["groups"]}
    assets = [url for group in manifest["groups"] for url in group["urls"]]
    if len(assets) < 250:
        raise RuntimeError(f"offline asset list is unexpectedly small ({len(assets)})")
    if len(set(assets)) != len(assets):
        raise RuntimeError("offline asset list repeats entries")
    listed = set(assets)

    for root in ("explorables", "quizzes", "research", "moves"):
        pages = sorted(
            path.relative_to(reader).as_posix() for path in (reader / root).rglob("*.html")
        )
        missing = [page for page in pages if page not in listed]
        if missing:
            raise RuntimeError(f"{root} pages left out of the offline list: {missing[:5]}")
        files = sorted(
            path.relative_to(reader).as_posix()
            for path in (reader / root).rglob("*")
            if path.is_file()
        )
        misplaced = [path for path in files if path in listed and path not in groups[root]]
        if misplaced:
            raise RuntimeError(f"{root} files outside their priority group: {misplaced[:5]}")

    # <lk-review> reloads a quiz by its JSON path, so every quiz file must be saved.
    quiz_files = sorted(path.relative_to(reader).as_posix() for path in (reader / "quizzes").glob("*.json"))
    missing = [path for path in quiz_files if path not in groups["quizzes"]]
    if missing:
        raise RuntimeError(f"quiz JSON files left out of the offline list: {missing}")

    for core in ("index.html", "app.js", "styles.css", "course-content.js", "sw.js",
                 "learnkit/lk.js", "learnkit/lk.css"):
        if core not in groups["core"]:
            raise RuntimeError(f"the reader shell is missing {core} from the offline list")

    lessons = manifest.get("lessons", {})
    payload = json.loads(
        (reader / "course-content.js").read_text().split("=", 1)[1].rstrip().rstrip(";")
    )
    for module in payload["modules"]:
        if module["id"] not in lessons:
            raise RuntimeError(f"lesson {module['id']} has no entry in the offline list")
        base = "/".join(module["path"].split("/")[:-1])
        figures = [
            f"content/{normalize(f'{base}/{source}')}"
            for source in MARKDOWN_IMAGE.findall(module["markdown"])
            if not re.match(r"^(https?:|data:|/)", source)
        ]
        lost = [figure for figure in figures if (reader / figure).is_file() and figure not in listed]
        if lost:
            raise RuntimeError(f"lesson {module['id']} figures left out of the offline list: {lost[:3]}")
        if module["explorable"] not in groups["explorables"]:
            raise RuntimeError(f"explorable {module['explorable']} is not saved offline")

    blobs = [asset for asset in assets if OFFLINE_BLOB.search(asset.removesuffix(".html"))]
    if blobs:
        raise RuntimeError(f"weights, data blobs or lockfiles in the offline list: {blobs[:5]}")
    hidden = [asset for asset in assets if any(part.startswith(".") for part in asset.split("/")[:-1])]
    if hidden:
        raise RuntimeError(f"hidden folders in the offline list: {hidden[:5]}")
    stale = [asset for asset in assets if not (reader / asset).is_file()]
    if stale:
        raise RuntimeError(f"offline list names files the reader does not have: {stale[:5]}")
    megabytes = sum((reader / asset).stat().st_size for asset in assets) / 1e6
    if megabytes > OFFLINE_MAX_MB:
        raise RuntimeError(f"offline list is {megabytes:.0f} MB, over the {OFFLINE_MAX_MB} MB budget")
    return assets


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
        reader / "quizzes" / "index.html",
        reader / "quizzes" / "00a.html",
        reader / "quizzes" / "01.html",
        reader / "quizzes" / "05a.html",
        reader / "content" / "modules" / "33-text-by-denoising" / "README.md",
        reader / "explorables" / "33-text-by-denoising.html",
        reader / "quizzes" / "33.html",
        reader / "content" / "modules" / "34-prompts-as-parameters" / "README.md",
        reader / "explorables" / "34-prompts-as-parameters.html",
        reader / "quizzes" / "34.html",
        reader / "quizzes" / "34.json",
        reader / "explorables" / "35-evolution-with-an-llm-mutator.html",
        reader / "quizzes" / "35.html",
        reader / "quizzes" / "35.json",
        reader / "learnkit" / "lk.js",
        reader / "research" / "agenda.html",
        reader / "research" / "index.html",
        reader / "research" / "why-atlas.html",
        reader / "research" / "jepa-field-map.html",
        reader / "research" / "assets" / "fonts.css",
        reader / "moves" / "index.html",
        reader / "favicon.svg",
        reader / "favicon.ico",
        reader / "apple-touch-icon.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing reader artifacts: " + ", ".join(missing))

    assets = check_offline_list(reader)

    quizzes = sorted((reader / "quizzes").glob("*.html"))
    if len(quizzes) < len(list((ROOT / "quizzes").glob("*.html"))):
        raise RuntimeError("the reader is missing quizzes bundled at the source")

    absolute = [asset for asset in assets if asset.startswith("/")]
    if absolute:
        raise RuntimeError(
            "the offline asset list must stay relative to the reader root so the "
            f"service worker works under a subpath: {absolute[:3]}"
        )

    for html in [
        reader / "index.html",
        *(reader / "explorables").glob("*.html"),
        *quizzes,
        *(reader / "research").glob("*.html"),
        *(reader / "moves").glob("*.html"),
    ]:
        text = html.read_text()
        parser = DependencyParser()
        parser.feed(text)
        if parser.remote_dependencies:
            raise RuntimeError(
                f"{html.name} loads remote runtime assets: {parser.remote_dependencies}"
            )
        stray = STRAY_MODULE_LINK.findall(text)
        if stray:
            raise RuntimeError(
                f"{html.name} links the module tree, which the reader serves from "
                f"../content/modules/ and its lesson routes: {stray}"
            )
    # Generated folder indexes and source views are checked too: they are the
    # newest pages in the reader and the easiest place for an absolute path to
    # creep back in.
    for html in sorted(reader.rglob("*.html")):
        rooted = ROOT_ABSOLUTE_URL.findall(html.read_text())
        if rooted:
            raise RuntimeError(
                f"{html.relative_to(reader)} uses root-absolute URLs, which break "
                f"when the reader is served under /ml-course/reader/: {rooted[:3]}"
            )

    shell = (reader / "app.js").read_text() + (reader / "sw.js").read_text()
    if re.search(r'(?:register|fetch|match)\(\s*"/', shell):
        raise RuntimeError("the reader shell still resolves URLs from the domain root")

    links = check_links(reader)
    print(
        f"reader: PASS ({len(assets)} offline assets, "
        f"{sum((reader / asset).stat().st_size for asset in assets) / 1e6:.1f} MB, "
        f"{len(quizzes) - 1} quizzes, "
        f"{links} links resolved, no remote dependencies, no unrouted module "
        "links, no root-absolute URLs)"
    )


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
            cwd=ROOT / "modules/11-inside-a-language-model/python",
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
            cwd=ROOT / "modules/12-features-and-dictionaries/python",
        )
        for module in (
            "13-inside-vision-and-jepa",
            "14-how-training-finds-a-solution",
            "15-geometry-of-representations",
            "16-self-supervised-and-jepa",
            "17-learning-from-preferences",
            "18-beyond-attention",
            "19-scaling-laws",
            "20-in-context-learning",
            "21-retrieval-and-memory",
            "22-reasoning-at-test-time",
            "23-compression",
            "24-reinforcement-learning",
            "25-world-models-and-planning",
            "26-rl-with-a-checker",
            "27-memory-as-a-matrix",
            "28-fewer-bits-trained-in",
            "29-learning-new-keeping-old",
            "30-learning-without-a-backward-pass",
            "31-world-model-from-pixels",
            "32-is-the-explanation-real",
            "33-text-by-denoising",
            "34-prompts-as-parameters",
            "35-evolution-with-an-llm-mutator",
        ):
            run_offline(
                ["uv", "run", "--offline", "--frozen", "pytest", "-q"],
                env,
                cwd=ROOT / "modules" / module / "python",
            )
    print(f"{args.pack} pack: PASS with network disabled")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"offline verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
