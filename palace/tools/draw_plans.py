#!/usr/bin/env python3
"""Draw one SVG floor plan per level from ../source/palace.json into ../source/plans/ (route as a numbered line)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE.parent / "source" / "palace.json").read_text())
S = 9  # px per metre
INK, BLUE, GOLD, FAINT, BG, MUTED = "#0F1D2B", "#2D57CF", "#B87708", "#B9C2CE", "#ECEFF3", "#4B5868"
FILL = {"gold": "#F3E7CF", "ink": "#DCE3EC", "blue": "#D6E0F7", "faint": "#E8ECF2"}

rooms = {r["id"]: r for r in P["rooms"]}
halls = {h["id"]: h for h in P["halls"]}
nodes = rooms | halls
step = {s["room"]: s["n"] for s in P["route"]}

def lvl(n):
    return ("tower", n["level"]) if n.get("floor") == "tower" and n["id"] in rooms and n["level"] > 0 else ("main", n["level"])

def shared(a, b):
    """Midpoint of the wall two rectangles share, or None."""
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    if ax1 == bx0 or bx1 == ax0:
        x = ax1 if ax1 == bx0 else ax0
        lo, hi = max(ay0, by0), min(ay1, by1)
        return (x, (lo + hi) / 2) if hi > lo else None
    if ay1 == by0 or by1 == ay0:
        y = ay1 if ay1 == by0 else ay0
        lo, hi = max(ax0, bx0), min(ax1, bx1)
        return ((lo + hi) / 2, y) if hi > lo else None
    return None

def centre(r):
    return ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)

def plan(name, title, level, ids, box, notes=()):
    x0, y0, x1, y1 = box
    W, H = (x1 - x0) * S + 40, (y1 - y0) * S + 90
    X = lambda x: 20 + (x - x0) * S
    Y = lambda y: 60 + (y1 - y) * S
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="IBM Plex Mono, Menlo, monospace">',
           f'<rect width="{W}" height="{H}" fill="{BG}"/>',
           f'<text x="20" y="28" font-size="17" font-weight="700" fill="{INK}" font-family="Avenir Next, Helvetica, sans-serif">{title}</text>',
           f'<text x="20" y="46" font-size="11" fill="{MUTED}">metres, north up · numbered line = the walk · grey = halls · gaps in walls = doors</text>']
    here = [i for i in ids if i in nodes]
    for i in here:  # halls first
        n = nodes[i]
        if i in halls:
            r = n["rect"]
            out.append(f'<rect x="{X(r[0])}" y="{Y(r[3])}" width="{(r[2]-r[0])*S}" height="{(r[3]-r[1])*S}" fill="#F7F8FA" stroke="{FAINT}" stroke-width="1" stroke-dasharray="3 3"/>')
            label = i.replace("_", " ")
            if i in ("outside_stair", "tower_stair"):
                continue
            out.append(f'<text x="{X(r[0])+4}" y="{Y(r[3])+12}" font-size="9.5" fill="{MUTED}">{label}</text>')
    for i in here:
        n = nodes[i]
        if i in rooms:
            r = n["rect"]
            out.append(f'<rect x="{X(r[0])}" y="{Y(r[3])}" width="{(r[2]-r[0])*S}" height="{(r[3]-r[1])*S}" fill="{FILL[n["colour"]]}" stroke="{INK}" stroke-width="1.6"/>')
    # doors: white gaps on shared walls between two nodes drawn on this plan
    for i in here:
        for d in nodes[i]["doors"]:
            if d in here and i < d:
                m = shared(nodes[i]["rect"], nodes[d]["rect"])
                if m:
                    vertical = nodes[i]["rect"][2] == nodes[d]["rect"][0] or nodes[d]["rect"][2] == nodes[i]["rect"][0]
                    if vertical:
                        out.append(f'<line x1="{X(m[0])}" y1="{Y(m[1])-9}" x2="{X(m[0])}" y2="{Y(m[1])+9}" stroke="{BG}" stroke-width="4"/>')
                    else:
                        out.append(f'<line x1="{X(m[0])-9}" y1="{Y(m[1])}" x2="{X(m[0])+9}" y2="{Y(m[1])}" stroke="{BG}" stroke-width="4"/>')
    # route segments whose whole chain is on this plan
    seq = P["route"]
    for a, b in zip(seq, seq[1:]):
        chain = [a["room"], *b["via"], b["room"]]
        if not all(c in here for c in chain):
            continue
        pts = [centre(nodes[chain[0]]["rect"])]
        ok = True
        for u, v in zip(chain, chain[1:]):
            m = shared(nodes[u]["rect"], nodes[v]["rect"])
            if m is None:
                ok = False
                break
            pts.append(m)
        if not ok:
            continue
        pts.append(centre(nodes[chain[-1]]["rect"]))
        d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{BLUE}" stroke-width="2.2" stroke-linejoin="round" opacity=".85"/>')
    for i in here:
        n = nodes[i]
        if i in rooms:
            r = n["rect"]; cx, cy = centre(r)
            w = (r[2] - r[0]) * S
            fs = 10 if w > 80 else 8.5
            words = n["name"].replace("The ", "").split(" ")
            lines, cur = [], ""
            for wd in words:
                if len(cur) + len(wd) + 1 > max(8, int(w / (fs * .62))):
                    lines.append(cur); cur = wd
                else:
                    cur = (cur + " " + wd).strip()
            lines.append(cur)
            ty = Y(cy) + 17
            out.append(f'<text x="{X(cx)}" y="{ty - 2}" font-size="{fs+1.5}" font-weight="700" text-anchor="middle" fill="{INK}">{i}</text>')
            for k, ln in enumerate(lines):
                out.append(f'<text x="{X(cx)}" y="{ty + 12 + k*11}" font-size="{fs}" text-anchor="middle" fill="{MUTED}">{ln}</text>')
            if i in step:
                out.append(f'<circle cx="{X(cx)}" cy="{Y(cy)-6}" r="11" fill="{BLUE}"/>')
                out.append(f'<text x="{X(cx)}" y="{Y(cy)-2}" font-size="11" font-weight="700" text-anchor="middle" fill="#fff">{step[i]}</text>')
    for (x, y, t) in notes:
        for k, ln in enumerate(t.split("|")):
            out.append(f'<text x="{X(x)}" y="{Y(y) + k*12}" font-size="10" fill="{GOLD}" font-weight="600">{ln}</text>')
    out.append("</svg>")
    (HERE.parent / "source" / "plans" / f"{name}.svg").write_text("\n".join(out))
    return name

def tower():
    order = ["15", "11", "12", "13", "32"]
    W, H = 520, 470
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="IBM Plex Mono, Menlo, monospace">',
           f'<rect width="{W}" height="{H}" fill="{BG}"/>',
           f'<text x="20" y="28" font-size="17" font-weight="700" fill="{INK}" font-family="Avenir Next, Helvetica, sans-serif">The tower · section, levels 0 to 4 (route 35 to 39)</text>',
           f'<text x="20" y="46" font-size="11" fill="{MUTED}">footprint 10 × 12 m at the south-west corner; spiral stair on the west side</text>']
    bx, bw, bh = 150, 220, 70
    top = 70
    for k, rid in enumerate(reversed(order)):
        r = rooms[rid]; y = top + k * bh
        out.append(f'<rect x="{bx}" y="{y}" width="{bw}" height="{bh}" fill="{FILL[r["colour"]]}" stroke="{INK}" stroke-width="1.6"/>')
        out.append(f'<circle cx="{bx+30}" cy="{y+bh/2}" r="12" fill="{BLUE}"/><text x="{bx+30}" y="{y+bh/2+4}" font-size="11" font-weight="700" text-anchor="middle" fill="#fff">{step[rid]}</text>')
        out.append(f'<text x="{bx+52}" y="{y+bh/2-3}" font-size="12" font-weight="700" fill="{INK}">{rid} · {r["name"].replace("The ","")}</text>')
        out.append(f'<text x="{bx+52}" y="{y+bh/2+13}" font-size="9.5" fill="{MUTED}">level {r["level"]} · {len(r["objects"])} loci</text>')
    out.append(f'<rect x="{bx-34}" y="{top}" width="30" height="{bh*5}" fill="#F7F8FA" stroke="{FAINT}" stroke-dasharray="3 3"/>')
    out.append(f'<text x="{bx-19}" y="{top+bh*5+16}" font-size="9.5" text-anchor="middle" fill="{MUTED}">stair</text>')
    out.append(f'<polyline points="{bx+30},{top+bh*4.5} {bx-19},{top+bh*4.5} {bx-19},{top+bh*0.5} {bx+30},{top+bh*0.5}" fill="none" stroke="{BLUE}" stroke-width="2" opacity=".7"/>')
    for k in (1, 2, 3):
        out.append(f'<line x1="{bx-19}" y1="{top+bh*(k+.5)}" x2="{bx+18}" y2="{top+bh*(k+.5)}" stroke="{BLUE}" stroke-width="2" opacity=".7"/>')
    out.append(f'<line x1="40" y1="{top+bh*5}" x2="{W-40}" y2="{top+bh*5}" stroke="{INK}" stroke-width="1"/>')
    out.append(f'<text x="{bx+bw+12}" y="{top+bh*5-6}" font-size="10" fill="{GOLD}">← door from 31,</text><text x="{bx+bw+12}" y="{top+bh*5+8}" font-size="10" fill="{GOLD}">the camera obscura</text>')
    out.append(f'<text x="{bx+bw+12}" y="{top+20}" font-size="10" fill="{GOLD}">lookout: the whole</text><text x="{bx+bw+12}" y="{top+34}" font-size="10" fill="{GOLD}">palace in view</text>')
    out.append("</svg>")
    (HERE.parent / "source" / "plans" / "plan-tower.svg").write_text("\n".join(out))

on = lambda level, floors: [i for i, n in nodes.items() if n["level"] == level and n.get("floor") in floors and not (n.get("floor") == "tower" and n["level"] > 0)]
made = [
    plan("plan-ground", "Ground level · garden, ground floor, yard, tower foot (route 1–3, 10–18, 30–35)", 0,
         on(0, {"garden", "ground", "yard", "tower"}) + ["outside_stair"], (-14, -18, 108, 26),
         [(68.6, -0.4, "↓ steps to cellar (3→4)"), (0.5, 19.5, "↑ 9→10|↓ 18→19|↑ 19→20"), (56.3, -6.2, "↑ outside stair|from 10 (29→30)")]),
    plan("plan-cellar", "Cellar · weights learned; walked west with the gradient (route 4–9)", -1,
         on(-1, {"cellar", "ground"}) + ["core"], (-2, -4, 70, 26),
         [(60.4, 19, "↑ garden|steps (3→4)"), (0.6, 19, "↑ to the gate|(9→10)")]),
    plan("plan-footings", "Footings · how big to build (route 19)", -2, ["19", "core"], (-2, -4, 40, 26),
         [(0.6, 19, "↑ core (18→19)|↑ first (19→20)")]),
    plan("plan-first", "First floor · weights changed after training (route 20–24)", 1,
         on(1, {"first"}) + ["core"], (-2, -4, 64, 26), [(0.6, 19, "↑ from footings|↑ second (24→25)")]),
    plan("plan-second", "Second floor · weights frozen, words change (route 25–29)", 2,
         on(2, {"second"}) + ["core"], (-2, -6, 70, 26), [(0.6, 19, "↑ from first|(24→25)"), (56.3, -5.6, "↓ outside stair|to the yard (29→30)")]),
]
tower()
# report doors between same-plan nodes that share no wall (stairs are expected here)
for i, n in nodes.items():
    for d in n["doors"]:
        if i < d and n["level"] == nodes[d]["level"] and not shared(n["rect"], nodes[d]["rect"]):
            print("door without a shared wall:", i, d)
print("wrote", made + ["plan-tower"])
