"""Check architecture records against schema.json, source coverage, the glossary and the stack.

Usage, from course/:
    uv run --with jsonschema python -I architectures/validate.py
        checks every records/*.json against schema.json and glossary.json beside this file
    uv run --with jsonschema python -I architectures/validate.py schema.json seed.json glossary.json
        older form: checks a {"models": [...]} file

build.py imports leaves() and covered(); record_problems() holds every other check, and list_drift()
checks that model_list.json agrees with the records. parts.py derives the parts a model has; schema 1.2.0
removed the hand tags in `teaches`, so a record that still has the field fails the schema check.
"""
import datetime
import json
import re
import sys
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent
EXEMPT = {"id", "name", "organisation", "family", "parent_id", "closed", "refs",
          "field_sources", "figure_links", "unknowns", "notes"}
ARRAY_LEAVES = {"specials", "omitted_towers", "attention.other", "attention.mixers",
                "block_order.steps", "stack.pattern", "stack.prefix", "stack.suffix",
                "modality.inputs", "modality.outputs"}
# Values allowed besides glossary ids: 'none' = no such part; 'mixed' = the scheme differs by layer kind.
PLACEMENT = {"pre": "pre_norm", "post": "post_norm", "sandwich": "sandwich_norm",
             "post_in_residual": "post_in_residual"}


def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if not path and k in EXEMPT:
                continue
            if p in ARRAY_LEAVES:
                if v:
                    yield p
                continue
            yield from leaves(v, p)
    elif obj is not None and obj != [] and obj != "":
        yield path


def covered(path, fs):
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        head = ".".join(parts[:i])
        if head in fs or head + ".*" in fs:
            return True
    return False


def load_glossary(path):
    terms = json.load(open(path))["terms"]
    ids = {t["id"] for t in terms}
    alias = {a: t["id"] for t in terms for a in t.get("aliases", [])}
    return ids, alias


def vocab(rec):
    """(field, value, extra allowed values) for every controlled-vocabulary field."""
    a, ffn, norm = rec["attention"], rec["ffn"], rec["norm"]
    pos = a["position"]
    yield "attention.kind", a["kind"], ()
    for m in a.get("mixers") or []:
        yield "attention.mixers.kind", m["kind"], ()
    for x in a.get("other") or []:
        yield "attention.other", x, ()
    if a.get("sparse"):
        yield "attention.sparse.kind", a["sparse"]["kind"], ()
    yield "attention.position.scheme", pos["scheme"], ("none", "mixed")
    if pos.get("scaling"):
        yield "attention.position.scaling", pos["scaling"], ()
    if a.get("qk_norm") is not None:
        yield "attention.qk_norm", a["qk_norm"], ("none",)
    yield "norm.kind", norm["kind"], ()
    yield "ffn.kind", ffn["kind"], ("none",)
    if ffn.get("activation") is not None:
        yield "ffn.activation", ffn["activation"], ()
    if (rec.get("tokenizer") or {}).get("kind"):
        yield "tokenizer.kind", rec["tokenizer"]["kind"], ()
    for s in rec["block_order"]["steps"]:
        yield "block_order.steps.kind", s["kind"], ()
    for s in rec["specials"]:
        yield "specials.name", s["name"], ()
    for k, v in (rec.get("layer_kinds") or {}).items():
        yield f"layer_kinds.{k}.mixer", v["mixer"], ("none",)
        if "ffn" in v:
            yield f"layer_kinds.{k}.ffn", v["ffn"], ("none",)
        if "position" in v:
            yield f"layer_kinds.{k}.position", v["position"], ("none",)


def stack_types(st):
    out = []
    for part in ("prefix", "pattern", "suffix"):
        for run in st.get(part) or []:
            if run["type"] not in out:
                out.append(run["type"])
    return out


def record_problems(rec, gloss, alias, ids=None):
    """Every check except schema and source coverage; ids is the set of record ids, for parent links."""
    out = []
    for field, value, extra in vocab(rec):
        if value in gloss or value in extra:
            continue
        hint = f" (merged into {alias[value]})" if value in alias else ""
        out.append(f"{field}: {value!r} is not a glossary id{hint}")
    if rec["norm"]["placement"] not in PLACEMENT:
        out.append(f"norm.placement: {rec['norm']['placement']!r} not one of {sorted(PLACEMENT)}")

    st = rec["stack"]
    total = sum(r["count"] for r in st.get("prefix") or []) + st["repeat"] * sum(r["count"] for r in st["pattern"]) \
        + sum(r["count"] for r in st.get("suffix") or [])
    if rec["layers"] is not None and total != rec["layers"]:
        out.append(f"stack sums to {total} layers, record says {rec['layers']}")
    types = stack_types(st)
    lk = rec.get("layer_kinds")
    if len(types) > 1 and not lk:
        out.append(f"layer_kinds missing for stack types {types}")
    if lk and set(lk) != set(types):
        out.append(f"layer_kinds keys {sorted(lk)} differ from stack types {sorted(types)}")
    for s in rec["block_order"]["steps"]:
        on = s.get("only_on")
        if on and not on[0].isdigit() and not set(on.split(", ")) <= set(types):
            out.append(f"block_order only_on {on!r} names no stack type {types}")

    for rid, r in rec["refs"].items():
        if not re.match(r"https?://", r["url"]):
            out.append(f"ref {rid} has no web URL")
    d = rec["release_date"]["value"]
    if d and not ("2017" <= d[:4] <= str(datetime.date.today().year)):
        out.append(f"release_date {d} out of range")
    if ids is not None and rec.get("parent_id") and rec["parent_id"] not in ids:
        out.append(f"parent_id {rec['parent_id']} is not a record")
    if rec.get("parent_id") == rec["id"]:
        out.append("parent_id points to itself")

    if rec.get("moe") and "first_dense" not in rec["moe"]:
        out.append("moe.first_dense missing (0 when no layer keeps a dense FFN)")
    sp = rec["attention"].get("sparse")
    if sp and lk:
        missing = [k for k, v in lk.items() if "sparse" not in v]
        if missing:
            out.append(f"layer_kinds {missing} need sparse: true or false, since attention.sparse is set")
        elif not any(v["sparse"] for v in lk.values()):
            out.append("attention.sparse is set but no layer kind has sparse: true")
    names = {x["name"] for x in rec["specials"]}
    if ("logit_softcap" in names) != (rec.get("logit_softcap") is not None):
        out.append("logit_softcap: set the cap exactly when specials name logit_softcap")
    if rec.get("attn_res_block") is not None and "attention_residuals" not in names:
        out.append("attn_res_block is set but no attention_residuals special describes it")
    if lk and not sp and any(v.get("sparse") for v in lk.values()):
        out.append("a layer kind has sparse: true but attention.sparse is null")
    return out


def check(models, schema, gloss_path):
    gloss, alias = load_glossary(gloss_path)
    validator = jsonschema.Draft202012Validator(schema)
    ids = {m["id"] for m in models}
    bad = 0
    for m in models:
        problems = [f"schema: {list(e.path)} {e.message[:120]}" for e in validator.iter_errors(m)]
        fs = m["field_sources"]
        problems += [f"no source for {p}" for p in leaves(m) if not covered(p, fs)]
        problems += [f"unknown ref {r} in {k}" for k, refs in fs.items() for r in refs if r not in m["refs"]]
        problems += record_problems(m, gloss, alias, ids)
        for p in problems:
            print(m["id"], p)
        bad += len(problems)
    print("records", len(models), "problems", bad)
    return 1 if bad else 0


def main(argv):
    if len(argv) == 3:
        schema_p, seed_p, gloss_p = argv
        return check(json.load(open(seed_p))["models"], json.load(open(schema_p)), gloss_p)
    if argv:
        sys.exit(__doc__)
    models, misnamed = [], 0
    for p in sorted((HERE / "records").glob("*.json")):
        m = json.loads(p.read_text())
        if m.get("id") != p.stem:
            misnamed += 1
            print(p.name, "id does not match the file name")
        models.append(m)
    drift = list_drift(models, json.load(open(HERE / "model_list.json")))
    for line in drift:
        print(line)
    status = check(models, json.load(open(HERE / "schema.json")), HERE / "glossary.json")
    return status or (1 if misnamed or drift else 0)


def list_drift(models, planned):
    """model_list.json keeps name, org, date and open_weights for models with no record yet; once a record exists they must match it."""
    recs = {m["id"]: m for m in models}
    out = []
    for p in planned:
        r = recs.get(p["id"])
        if not r:
            continue
        want = {"name": r["name"], "org": r["organisation"], "date": r["release_date"]["value"], "open_weights": r["weights"]["open"]}
        out += [f"model_list.json {p['id']}: {k} {p.get(k)!r} differs from the record's {v!r}" for k, v in want.items() if p.get(k) != v]
    return out


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
