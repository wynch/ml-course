# ML course — working agreement

This file is the whole agreement for this folder. An agent that has read it and
`git log --oneline` knows everything it needs. Nothing about this project lives in
any agent's private memory.

## What this is

A hands-on machine-learning course Vincent built for himself: fourteen modules that
each build an idea from scratch, then rebuild it the Hugging Face way, with figures,
runnable scripts, a browser explorable and a quiz. Work started 2026-07-23 and the
course was finished 2026-08-04. `README.md` is the front door — tracks, modules,
explorables, quizzes, map. `SETUP.md` holds the toolchain, `CLOUD.md` the optional
cloud lane, `OFFLINE.md` the offline packs, `docs/algorithm-cards.md` the
Python-and-Zig pairing, `RESOURCES.md` the outside reading.

## Its role today

The course is finished and read-only. It serves the ML Compass project at
`../ml-compass` as a quarry: the compass points at an explorable, a figure or a quiz
question when a notion needs a door, through the `see:` lines in its `spine.yaml`.
Pull one page or one number when a question calls for it. Do not extend, reorganize
or improve the course, and build nothing on top of it, unless Vincent asks. A new
module, a new explorable, a rewritten lesson: each needs his yes first.

## State on 2026-09-25

| Fact | Value |
|---|---|
| Last course commit | `0da7607`, 2026-08-04; later commits change only this file |
| Branch | `main`, level with `origin/main`, tree clean |
| Content | 14 modules, 14 explorables, 14 quizzes (140 questions), course map |
| Site | built and deployed from `main` by the Pages workflow |
| Work since 2026-08-04 | none |

Per-module results and measured numbers live in each module README and on the map.

## Links

| What | Where |
|---|---|
| Repo | https://github.com/wynch/ml-course |
| Site | https://wynch.github.io/ml-course/ |
| Map | https://wynch.github.io/ml-course/map.html |
| Reader | https://wynch.github.io/ml-course/reader/ |

The map first lived as a copy in a chat artifact. That copy is dead. Never
republish it; the Pages map is the only home.

## How work runs

The orchestrating agent reads, proposes, commits and reports; it does not build.
One worker builds from a written brief. A second, independent worker reviews the
result against the brief, fixes real defects, and commits the fixes. Workers that
touch the same files run one after another. Results go to Vincent in chat before
the next proposal. Write and talk to him in English, in short words.

A push to `main` republishes the public site, so commit only finished work.

## Open gates (Vincent's call)

- The cloud lane in `CLOUD.md` is written but has never been run. It needs prepaid
  Hub credits, about $0.50 a run on a small GPU. Do not spend without his yes.
- The local branch `codex/offline-course-reader` is fully merged into `main` and its
  remote copy is gone. Deleting the local branch is his call.
- `dist/`, `app/`, `build/` and `worker/` are leftovers from that old side
  experiment. Nothing in the course reads them. Removing them is his call.
