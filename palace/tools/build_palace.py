#!/usr/bin/env python3
"""Write ../source/palace.json from the tables below, then validate it.

These tables are the source of the palace: rename a room or change an object here,
then run this script and tools/build_page.py.

    python3 course/palace/tools/build_palace.py          # write source/palace.json
    python3 course/palace/tools/build_palace.py --check  # write nothing; exit 1 on an error or a stale file

Grid: metres, x east, y north, origin at the south-west corner of the house.
Levels: -2 foundations, -1 cellar, 0 ground (and garden, yard), 1, 2; the tower
stacks its own levels 0..4 on its own footprint. One storey is 4 m.
"""
import json, re, sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "source" / "palace.json"
COURSE = HERE.parents[1]

EXPL = {p.stem.split("-")[0]: p.name for p in (COURSE / "explorables").glob("*.html")
        if p.stem not in ("index", "05b-architecture-explorer")}
SLUG = {p.name.split("-")[0]: p.name for p in (COURSE / "modules").iterdir() if p.is_dir()}

def links(m):
    return {"module": SLUG[m], "explorable": EXPL[m], "quiz": m}

FLOORS = [
    {"id": "garden", "level": 0, "name": "The old garden",
     "reason": "Before the house: the classic methods. It lies beyond the output balcony, so the balcony looks back at where the walk began; the first learning machine stands farthest out."},
    {"id": "cellar", "level": -1, "name": "The cellar",
     "reason": "Weights being learned. Entered from the garden under the balcony, where the loss is computed; walked west, the way the gradient flows, so it ends under the gate, the last thing the gradient reaches."},
    {"id": "ground", "level": 0, "name": "The ground floor",
     "reason": "Weights used: one forward pass. The south row reads text in, west to east; the north row is the way back, where each written token returns to the gate."},
    {"id": "foundations", "level": -2, "name": "The footings",
     "reason": "How big to build: loss against size, data and compute. Below the cellar because every floor stands on it."},
    {"id": "first", "level": 1, "name": "The first floor",
     "reason": "Weights changed after training: fine-tuning, what that forgets, choices, and making the model smaller."},
    {"id": "second", "level": 2, "name": "The second floor",
     "reason": "Weights frozen: only the words going in or coming out change. Examples, steps, instructions, fetched pages, tools."},
    {"id": "yard", "level": 0, "name": "The yard",
     "reason": "Outside the model: a score from the world decides, a reward or a program that checks."},
    {"id": "tower", "level": 0, "name": "The tower",
     "reason": "Weights read, never changed: looking inside. The higher you climb, the harder the question; the top asks whether any of it is real."},
]

# Halls are not lessons: corridors, stairs and paths the walk may cross more than once.
HALLS = [
    ("garden_path", "garden", 0, [68, -4, 106, 2], "Garden path; its west end drops by steps into the cellar."),
    ("loss_pit", "cellar", -1, [60, 0, 68, 24], "Foot of the garden steps, under the balcony: where the loss is computed and the backward pass begins."),
    ("gradient_corridor", "cellar", -1, [12, 10, 60, 14], "The gradient channel: gold water running west, under the residual stream."),
    ("core", "ground", 0, [0, 10, 12, 24], "Stair core beside the gate, from the footings to the second floor."),
    ("residual_corridor", "ground", 0, [12, 10, 60, 14], "The residual stream: a channel of blue water from the gate to the balcony; every ground-floor room pours into it."),
    ("balcony", "ground", 0, [60, 0, 68, 24], "The output balcony, looking east over the old garden."),
    ("gallery1", "first", 1, [12, 10, 60, 14], "First-floor gallery."),
    ("gallery2", "second", 2, [12, 10, 60, 14], "Second-floor gallery."),
    ("outside_stair", "second", 2, [56, -4, 68, 0], "Outside stair from the workshop of hands down to the yard path."),
    ("yard_path", "yard", 0, [-12, -4, 56, 0], "Yard path along the south wall, walked west."),
    ("tower_stair", "tower", 0, [-12, -16, -10, -4], "Spiral stair inside the tower."),
]

# Rooms: module, name, floor, level, rect, colour family.
ROOMS = [
    ("00a", "The string-line bed", "garden", 0, [94, 2, 106, 20], "gold"),
    ("00b", "The four-tree orchard", "garden", 0, [82, 2, 94, 20], "gold"),
    ("00c", "The hollow pond", "garden", 0, [70, 2, 82, 20], "gold"),
    ("01", "The echo room", "cellar", -1, [50, 0, 60, 10], "gold"),
    ("02", "The mill", "cellar", -1, [44, 14, 60, 24], "gold"),
    ("05a", "The vault", "cellar", -1, [40, 0, 50, 10], "gold"),
    ("14", "The sunken slope", "cellar", -1, [26, 14, 44, 24], "gold"),
    ("16", "The blindfold room", "cellar", -1, [20, 0, 40, 10], "gold"),
    ("30", "The one-way room", "cellar", -1, [12, 14, 26, 24], "gold"),
    ("03", "The gate", "ground", 0, [0, 0, 12, 10], "ink"),
    ("04", "The atrium", "ground", 0, [12, 0, 30, 10], "ink"),
    ("08", "The picture door", "ground", 0, [30, 0, 42, 10], "ink"),
    ("05", "The balcony room", "ground", 0, [42, 0, 60, 10], "ink"),
    ("07", "The scriptorium", "ground", 0, [50, 14, 60, 24], "blue"),
    ("18", "The satchel room", "ground", 0, [41, 14, 50, 24], "blue"),
    ("27", "The slate room", "ground", 0, [32, 14, 41, 24], "blue"),
    ("09", "The fog room", "ground", 0, [22, 14, 32, 24], "blue"),
    ("33", "The blank-line room", "ground", 0, [12, 14, 22, 24], "blue"),
    ("19", "The footings", "foundations", -2, [12, 0, 36, 24], "gold"),
    ("06", "The tailor's room", "first", 1, [12, 0, 22, 10], "blue"),
    ("29", "The fading shelf", "first", 1, [22, 0, 31, 10], "blue"),
    ("17", "The tasting room", "first", 1, [31, 0, 41, 10], "blue"),
    ("23", "The press", "first", 1, [41, 0, 51, 10], "blue"),
    ("28", "The three-coin room", "first", 1, [51, 0, 60, 10], "blue"),
    ("20", "The example desk", "second", 2, [12, 0, 22, 10], "faint"),
    ("22", "The scratch room", "second", 2, [22, 0, 31, 10], "faint"),
    ("34", "The sign-painter's room", "second", 2, [31, 0, 41, 10], "faint"),
    ("21", "The library", "second", 2, [41, 0, 51, 10], "faint"),
    ("10", "The workshop of hands", "second", 2, [51, 0, 60, 10], "faint"),
    ("24", "The lever shed", "yard", 0, [46, -16, 56, -4], "ink"),
    ("26", "The marking hut", "yard", 0, [34, -16, 46, -4], "ink"),
    ("35", "The greenhouse", "yard", 0, [22, -16, 34, -4], "ink"),
    ("25", "The sand table", "yard", 0, [10, -16, 22, -4], "ink"),
    ("31", "The camera obscura", "yard", 0, [0, -16, 10, -4], "ink"),
    ("15", "The map room", "tower", 0, [-10, -16, 0, -4], "faint"),
    ("11", "The listening room", "tower", 1, [-10, -16, 0, -4], "faint"),
    ("12", "The dictionary room", "tower", 2, [-10, -16, 0, -4], "faint"),
    ("13", "The lens room", "tower", 3, [-10, -16, 0, -4], "faint"),
    ("32", "The lookout", "tower", 4, [-10, -16, 0, -4], "faint"),
]

DOORS = [
    ("00a", "garden_path"), ("00b", "garden_path"), ("00c", "garden_path"),
    ("00a", "00b"), ("00b", "00c"),
    ("garden_path", "loss_pit"),
    ("loss_pit", "gradient_corridor"), ("loss_pit", "01"),
    *[(r, "gradient_corridor") for r in ("01", "02", "05a", "14", "16", "30")],
    ("01", "05a"), ("05a", "16"), ("02", "14"), ("14", "30"),
    ("gradient_corridor", "core"), ("30", "core"),
    ("core", "03"), ("core", "residual_corridor"), ("core", "33"), ("core", "19"),
    ("core", "gallery1"), ("core", "gallery2"),
    *[(r, "residual_corridor") for r in ("04", "08", "05", "07", "18", "27", "09", "33")],
    ("03", "04"), ("04", "08"), ("08", "05"),
    ("residual_corridor", "balcony"), ("05", "balcony"), ("balcony", "07"),
    ("07", "18"), ("18", "27"), ("27", "09"), ("09", "33"),
    ("08", "yard_path"),
    *[(r, "gallery1") for r in ("06", "29", "17", "23", "28")],
    ("06", "29"), ("29", "17"), ("17", "23"), ("23", "28"),
    *[(r, "gallery2") for r in ("20", "22", "34", "21", "10")],
    ("20", "22"), ("22", "34"), ("34", "21"), ("21", "10"),
    ("10", "outside_stair"), ("outside_stair", "yard_path"),
    *[(r, "yard_path") for r in ("24", "26", "35", "25", "31")],
    ("31", "15"), ("yard_path", "tower_stair"),
    *[(r, "tower_stair") for r in ("15", "11", "12", "13", "32")],
]

# The walk: the order of the rooms. The sentence said at each threshold, the loci kept on
# the walk and the ones demoted to the list are in source/loci.json, each checked against
# its module.
ROUTE = ["00a", "00b", "00c", "01", "02", "05a", "14", "16", "30", "03", "04", "08", "05",
         "07", "18", "27", "09", "33", "19", "06", "29", "17", "23", "28", "20", "22", "34",
         "21", "10", "24", "26", "35", "25", "31", "15", "11", "12", "13", "32"]

# Steps that do not take the shortest door path. From the gate the walk goes into the
# stair core and east down the residual corridor, so the stream is met at its source.
VIA = {("03", "04"): ["core", "residual_corridor"]}

LOCI = json.loads((HERE.parent / "source" / "loci.json").read_text())["rooms"]

def locus(o, rid, k, kept):
    out = {"id": f"{rid}-{k}" if kept else f"{rid}-x{k}", "from": o["from"], "idea": o["idea"],
           "image": o["image"], "moves": o["moves"] if o.get("moves_ok", True) else [],
           "moves_unshown": [] if o.get("moves_ok", True) else o["moves"],
           "card": o["card"], "checked": o["checked"], "links": links(rid)}
    if kept:
        out["formation"] = o["formation"]
    else:
        out["near"] = o["near"]
    return out


def build():
    rooms = []
    for (m, name, floor, level, rect, colour) in ROOMS:
        L = LOCI[m]
        objs = [locus(o, m, k, True) for k, o in enumerate(L["keep"], 1)]
        by_from = {o["from"]: o["id"] for o in objs}
        also = [locus(o, m, k, False) for k, o in enumerate(L["demote"], 1)]
        for a in also:
            a["near"] = by_from[a["near"]]
        doors = sorted({b for a, b in DOORS if a == m} | {a for a, b in DOORS if b == m})
        rooms.append({"id": m, "module": SLUG[m], "name": L.get("name") or name, "floor": floor, "level": level,
                      "rect": rect, "size": [rect[2] - rect[0], rect[3] - rect[1]],
                      "colour": colour, "doors": doors, "objects": objs, "also": also, "links": links(m)})
    halls = []
    for (hid, floor, level, rect, note) in HALLS:
        doors = sorted({b for a, b in DOORS if a == hid} | {a for a, b in DOORS if b == hid})
        halls.append({"id": hid, "floor": floor, "level": level, "rect": rect, "note": note, "doors": doors})
    route, prev = [], None
    for n, rid in enumerate(ROUTE, 1):
        via = (VIA.get((prev, rid)) or path_via(prev, rid)) if prev else []
        route.append({"n": n, "room": rid, "via": via, "why": LOCI[rid]["why"]})
        prev = rid
    return {"title": "The model house", "units": "metres; x east, y north; level = storey, 4 m each",
            "rule": "Ask what happens to the weights, and you know where you are: before the house (garden), being learned (below ground), used once (ground floor), changed after training (first floor), frozen while the words change (second floor), scored by the world (yard), read only (tower).",
            "floors": FLOORS, "halls": halls, "rooms": rooms, "route": route,
            "landmarks": LANDMARKS,
            "prologue": {"where": "garden_path, east end", "moves": ["a1", "a2", "a3", "a4"],
                         "what": "Before the first bower, the garden gate: 420 points in a loose field (a1), then a grid of numbers (a2), then two kinds of points and a first wrong line (a3, a4). It opens the walk the way the Moves page opens."},
            "epilogue": {"where": "32, the lookout", "moves": ["a94", "a95", "a96", "a97", "a98", "a99", "a100"],
                         "what": "At the top, the Moves page's last chapter: five ideas that kept coming back, each pointing at rooms below: hide and predict (16, 33), one local rule (01), lookup as a dot product (04, 21, 27), propose, score, keep (22, 34, 35), no number without its control (05a, 32)."}}

LANDMARKS = [
    {"id": "residual_stream", "where": "residual_corridor", "what": "A channel of blue water down the ground-floor corridor from the gate to the balcony. Every ground-floor room has a door on it and pours in; from any room you can see it."},
    {"id": "gradient_channel", "where": "gradient_corridor", "what": "Directly below, in the cellar, a channel of gold water runs the other way, west. Grates in the ground-floor corridor show it."},
    {"id": "balcony_view", "where": "balcony", "what": "The output balcony looks east over the old garden: you can see the string-line bed where the walk began, farthest away because it is oldest."},
    {"id": "garden_steps", "where": "garden_path", "what": "Steps from the garden go down under the balcony into the cellar: the loss is computed at the output, so learning starts beneath it."},
    {"id": "core_by_gate", "where": "core", "what": "The stair core stands beside the gate. The cellar stair comes up there because the gradient reaches the embedding table last; every upper floor is reached from where text enters."},
    {"id": "picture_door", "where": "08", "what": "The picture door opens south onto the yard path: images come from the world and skip the gate's merges: they are cut into patches instead."},
    {"id": "light_well", "where": "16", "what": "A light well in the yard path looks down into the blindfold room (16): the JEPA objective below, the JEPA world models (25, 31) across the path."},
    {"id": "glass_over_atrium", "where": "06", "what": "The tailor's room floor is glass over the atrium: thin matrices are added above the frozen block."},
    {"id": "grate_over_shelves", "where": "28", "what": "The north row is one storey high with a skylight roof; from the first-floor gallery outside the three-coin room you look down through the skylight onto the scriptorium's KV shelves (07), which 28 also rounds."},
    {"id": "outside_stair", "where": "outside_stair", "what": "The only way from the house's top floor to the yard is the outside stair from the workshop of hands: agents are where the model starts to act."},
    {"id": "tower", "where": "32", "what": "The tower stands at the south-west corner and is the tallest thing. From the lookout you see the whole roof, the yard and, far east, the garden: the place to rehearse the walk."},
]

def neighbours(node):
    return {b for a, b in DOORS if a == node} | {a for a, b in DOORS if b == node}

HALL_IDS = {h[0] for h in HALLS}

def path_via(a, b):
    """Shortest door path from room a to room b that crosses halls only."""
    q, seen = deque([(a, [])]), {a}
    while q:
        node, path = q.popleft()
        for nb in sorted(neighbours(node)):
            if nb == b:
                return path
            if nb in HALL_IDS and nb not in seen:
                seen.add(nb)
                q.append((nb, path + [nb]))
    raise ValueError(f"no door path from {a} to {b}")

def validate(p):
    errs = []
    ids = [r["id"] for r in p["rooms"]]
    modules = sorted(p_.name for p_ in (COURSE / "modules").iterdir() if p_.is_dir())
    if sorted(r["module"] for r in p["rooms"]) != modules:
        errs.append("rooms do not cover each module exactly once")
    if len(set(ids)) != len(ids):
        errs.append("duplicate room ids")
    nodes = {r["id"]: r for r in p["rooms"]} | {h["id"]: h for h in p["halls"]}
    for a, b in DOORS:
        if a not in nodes or b not in nodes:
            errs.append(f"door to unknown node {a}-{b}")
    for n in nodes.values():
        for d in n["doors"]:
            if n["id"] not in nodes[d]["doors"]:
                errs.append(f"door {n['id']}->{d} not symmetric")
    seq = [s["room"] for s in p["route"]]
    if sorted(seq) != sorted(ids) or len(seq) != len(set(seq)):
        errs.append("route does not visit every room exactly once")
    for s0, s1 in zip(p["route"], p["route"][1:]):
        chain = [s0["room"], *s1["via"], s1["room"]]
        for x, y in zip(chain, chain[1:]):
            if y not in nodes[x]["doors"]:
                errs.append(f"route step {s1['n']}: no door {x}-{y}")
        if any(v not in HALL_IDS for v in s1["via"]):
            errs.append(f"route step {s1['n']} crosses a room")
    moves = json.loads((HERE / "moves_index.json").read_text())["moves"]
    for r in p["rooms"]:
        if not 3 <= len(r["objects"]) <= 4:
            errs.append(f"{r['id']} has {len(r['objects'])} loci, not 3 or 4")
        old = [o["from"] for o in r["objects"] + r["also"]]
        if len(old) != len(set(old)):
            errs.append(f"{r['id']}: an old locus appears twice")
        for o in r["objects"] + r["also"]:
            c = o["card"]
            if not (c.get("title") and c.get("text") and o.get("checked") and o.get("idea") and o.get("image")):
                errs.append(f"{o['id']}: card, idea, image or 'checked' note missing")
        for o in r["objects"] + r["also"]:
            for mv in o["moves"] + o["moves_unshown"]:
                if mv not in moves:
                    errs.append(f"{o['id']}: unknown move {mv}")
                elif moves[mv].get("module") not in (None, r["module"]):
                    errs.append(f"{o['id']}: move {mv} links {moves[mv]['module']}, not {r['module']}")
            L = o["links"]
            for f in (COURSE / "explorables" / L["explorable"], COURSE / "quizzes" / f"{L['quiz']}.html", COURSE / "modules" / L["module"] / "README.md"):
                if not f.exists():
                    errs.append(f"{o['id']}: missing {f}")
    used = [mv for r in p["rooms"] for o in r["objects"] + r["also"] for mv in o["moves"] + o["moves_unshown"]]
    dup = sorted({m for m in used if used.count(m) > 1})
    return errs, dup, sorted(set(moves) - set(used))

if __name__ == "__main__":
    p = build()
    errs, dup, unused = validate(p)
    text = json.dumps(p, indent=1, ensure_ascii=False) + "\n"
    check = "--check" in sys.argv[1:]
    if check:
        if not OUT.exists() or OUT.read_text() != text:
            errs.append(f"{OUT.name} is stale: run tools/build_palace.py")
    else:
        OUT.write_text(text)
    print("rooms", len(p["rooms"]), "loci", sum(len(r["objects"]) for r in p["rooms"]),
          "demoted", sum(len(r["also"]) for r in p["rooms"]), "route", len(p["route"]))
    print("moves used twice:", dup)
    print("moves not placed:", unused)
    print("errors:", errs or "none")
    sys.exit(1 if errs else 0)
