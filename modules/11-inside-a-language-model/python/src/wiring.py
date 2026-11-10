"""Section 5: who feeds a neuron, and who reads it.

SmolLM2-360M is a Llama model. Its MLP is gated (SwiGLU):

    value = silu(gate_proj(x)) * up_proj(x),   x = RMSNorm(h)

where h is the residual stream at that token. h is a plain sum of writes: the
token embedding, every earlier attention output, every earlier neuron's
value times its column of down_proj. RMSNorm divides h by one number, its
root-mean-square, and scales each coordinate by a fixed weight. Hold that
number at its actual value and the gate and up pre-activations are linear in
h, so they split exactly into one term per write. "Who feeds this neuron" is
then arithmetic. The script asserts that the terms add up.

The other direction is not exact. A neuron's write reaches a later neuron
directly, through that neuron's norm, and also through every block between
them. We estimate the direct part to first order, then zero the neuron and
measure, to see how far the estimate is off.

The neuron is L4/N2177 from section 4. Its + pole fires on a dash between two
dates and on word pieces such as "Cour·ant": two contexts, one neuron, one
write direction.

Runs on the CPU in under a minute once SmolLM2 is cached.

Run me:
    uv run python src/wiring.py            # numbers, results/wiring.json
"""

from __future__ import annotations

from itertools import combinations
from statistics import median

from common import MODEL_ID, Guard, load_hf_model, load_tokenizer, write_results

import torch
import torch.nn.functional as F

LAYER, NEURON = 4, 2177
# Two of the neuron's top + contexts in section 4 (results/poles.json), cut at
# the firing token. The model is causal, so later tokens change nothing here.
CONTEXTS = {
    "date dash": "\n Charles Eaton , OBE , AFC ( 21 December 1895 –",
    "word piece": " James Endrst of the Hartford Courant",
}
FIRES_ON = {"date dash": " –", "word piece": "ant"}
TOP = 5          # feeders and readers printed per context
OVERLAP = 20     # size of the sets compared across contexts
TOL = 1e-4       # float32; pre-activations here are of order 1 to 10


# ------------------------------------------------------------- the toy
def toy() -> list[dict]:
    """Hand-built numbers: one neuron N, two readers, two contexts.

    Both readers take N through their up input. Each reader's gate listens
    to a context unit ("primer") instead. N writes the same value in both
    contexts; the primers decide which reader passes it on.
    """
    rows = []
    for context, primers in (("marine", {"marine": 1.0, "dental": 0.0}),
                             ("dental", {"marine": 0.0, "dental": 1.0}),
                             ("marine, N silent", {"marine": 1.0, "dental": 0.0})):
        n = 0.0 if "silent" in context else 2.0
        row = {"context": context, "N": n}
        for reader in ("marine", "dental"):
            gate = 4.0 * primers[reader] - 2.0        # +2 when its primer is on, -2 when off
            up = 1.0 * n                               # reads N, and only N
            row[f"{reader} reader"] = round(float(F.silu(torch.tensor(gate)) * up), 3)
        rows.append(row)
    return rows


# --------------------------------------------------- one recorded forward
def check_gated(model) -> dict:
    """Confirm from the loaded weights that every layer's MLP is SwiGLU with no biases."""
    for layer in model.model.layers:
        mlp = layer.mlp
        names = sorted(n for n, _ in mlp.named_children())
        assert names == ["act_fn", "down_proj", "gate_proj", "up_proj"], names
        assert all(getattr(mlp, p).bias is None for p in ("gate_proj", "up_proj", "down_proj"))
        assert type(mlp.act_fn).__name__ == "SiLUActivation", type(mlp.act_fn)
    return {"class": type(mlp).__name__, "children": names, "act_fn": type(mlp.act_fn).__name__,
            "biases": False, "intermediate_size": model.config.intermediate_size}


@torch.no_grad()
def record(model, ids: torch.Tensor, zero: tuple[int, int, int] | None = None) -> dict:
    """One forward pass; per layer, at every position: the attention write, the
    MLP values, gate and up pre-activations, and the residual that the MLP's
    norm reads. ``zero=(layer, position, neuron)`` sets one value to 0 first."""
    rec = {"attn": [], "value": [], "gate": [], "up": [], "h_mid": []}
    handles = []
    layers = model.model.layers

    def keep(key):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            rec[key].append(out[0].float().clone())
        return hook

    for layer in layers:
        handles += [
            layer.self_attn.register_forward_hook(keep("attn")),
            layer.mlp.gate_proj.register_forward_hook(keep("gate")),
            layer.mlp.up_proj.register_forward_hook(keep("up")),
            layer.post_attention_layernorm.register_forward_pre_hook(
                lambda m, inp: rec["h_mid"].append(inp[0][0].float().clone())),
        ]

    def value_hook(i):
        def hook(module, inputs):
            x = inputs[0]
            if zero is not None and zero[0] == i:
                x = x.clone()
                x[0, zero[1], zero[2]] = 0.0
            rec["value"].append(x[0].float().clone())
            return (x,)
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(value_hook(i)))
    try:
        emb = model.model.embed_tokens(ids)[0].float()
        model(ids)
    finally:
        for h in handles:
            h.remove()
    rec = {k: torch.stack(v) for k, v in rec.items()}     # (layers, seq, ...)
    rec["embed"] = emb
    return rec


def norm_lens(model, layer: int, h: torch.Tensor) -> torch.Tensor:
    """RMSNorm with its scale frozen at the actual value: x = h * this."""
    ln = model.model.layers[layer].post_attention_layernorm
    rms = torch.sqrt(h.pow(2).mean() + ln.variance_epsilon)
    return ln.weight.float() / rms


# ---------------------------------------------------------- who feeds it
@torch.no_grad()
def feeders(model, rec: dict, t: int) -> dict:
    """Split the target's gate and up pre-activations at position t into one
    term per write: embedding, attention of layers 0..LAYER, and each of the
    2,560 neurons in every layer below LAYER. Exact: the terms sum to the whole."""
    W_down = [model.model.layers[l].mlp.down_proj.weight.float() for l in range(LAYER)]
    writes = [("embedding", rec["embed"][t])]
    writes += [(f"attn L{l}", rec["attn"][l, t]) for l in range(LAYER + 1)]
    h = rec["h_mid"][LAYER, t]
    resid_sum = sum(w for _, w in writes) + sum(rec["value"][l, t] @ W_down[l].T for l in range(LAYER))
    resid_err = float((resid_sum - h).abs().max())
    assert resid_err < TOL * float(h.abs().max()), resid_err

    lens = norm_lens(model, LAYER, h)
    mlp = model.model.layers[LAYER].mlp
    out = {"residual_sum_max_abs_err": resid_err}
    for lane, W in (("gate", mlp.gate_proj.weight), ("up", mlp.up_proj.weight)):
        v = W[NEURON].float() * lens                              # (960,)
        terms = {name: float(v @ w) for name, w in writes}
        per_neuron = torch.stack([rec["value"][l, t] * (W_down[l].T @ v) for l in range(LAYER)])
        total = sum(terms.values()) + float(per_neuron.sum())
        actual = float(rec[lane][LAYER, t, NEURON])
        assert abs(total - actual) < TOL * max(1.0, abs(actual)), (lane, total, actual)
        flat = per_neuron.flatten()
        order = flat.abs().argsort(descending=True)
        top = [{"from": f"L{int(i) // 2560}/N{int(i) % 2560}", "term": round(float(flat[i]), 4)}
               for i in order[:OVERLAP]]
        out[lane] = {
            "actual": round(actual, 4),
            "sum_of_terms": round(total, 4),
            "abs_err": abs(total - actual),
            "embedding": round(terms["embedding"], 4),
            "attention": round(sum(v for k, v in terms.items() if k.startswith("attn")), 4),
            "neurons": round(float(per_neuron.sum()), 4),
            "by_attention_layer": {k: round(v, 4) for k, v in terms.items() if k.startswith("attn")},
            "top_neurons": top,
        }
    return out


# ---------------------------------------------------------- who reads it
@torch.no_grad()
def readers(model, rec: dict, t: int) -> list[dict]:
    """First-order, direct-path effect of the target's write on every later
    neuron at position t, through that neuron's norm frozen at its actual scale.

    d = value * down_proj[:, NEURON]; dz = v . d for each lane; then
        da ~= silu(z_gate) * dz_up  +  silu'(z_gate) * z_up * dz_gate.
    Paths through the blocks in between are left out.
    """
    a = float(rec["value"][LAYER, t, NEURON])
    d = a * model.model.layers[LAYER].mlp.down_proj.weight[:, NEURON].float()
    rows = []
    for k in range(LAYER + 1, model.config.num_hidden_layers):
        mlp = model.model.layers[k].mlp
        lens = norm_lens(model, k, rec["h_mid"][k, t])
        dz_g = mlp.gate_proj.weight.float() @ (lens * d)
        dz_u = mlp.up_proj.weight.float() @ (lens * d)
        zg, zu = rec["gate"][k, t], rec["up"][k, t]
        s = torch.sigmoid(zg)
        via_up = F.silu(zg) * dz_u
        via_gate = s * (1 + zg * (1 - s)) * zu * dz_g          # silu'(z) = s (1 + z (1 - s))
        for n in range(zg.numel()):
            rows.append((k, n, float(via_up[n] + via_gate[n]), float(via_up[n]), float(via_gate[n]),
                         float(F.silu(zg[n])), float(dz_u[n])))
    rows.sort(key=lambda r: -abs(r[2]))
    return [{"reader": f"L{k}/N{n}", "layer": k, "neuron": n, "da_first_order": round(da, 4),
             "via_up": round(vu, 4), "via_gate": round(vg, 4), "silu_gate": round(sg, 4),
             "dz_up": round(du, 4)} for k, n, da, vu, vg, sg, du in rows]


def reader_id(r: str) -> tuple[int, int]:
    k, n = r.split("/")
    return int(k[1:]), int(n[1:])


def run() -> dict:
    guard = Guard("wiring.py")
    toy_rows = toy()
    print("toy: one neuron N, two readers that take N through up and a primer through gate")
    for r in toy_rows:
        print(f"  {r['context']:<17} N = {r['N']:.1f}   marine reader {r['marine reader']:+.3f}   "
              f"dental reader {r['dental reader']:+.3f}")

    tok = load_tokenizer()
    model = load_hf_model("cpu")
    gated = check_gated(model)
    guard.tick("load")

    per_context, top_readers, measured_all, gate_moved = {}, {}, {}, {}
    for name, text in CONTEXTS.items():
        ids = tok(text, return_tensors="pt").input_ids
        t = ids.shape[1] - 1
        assert tok.decode([int(ids[0, t])]) == FIRES_ON[name], tok.decode([int(ids[0, t])])
        rec = record(model, ids)
        # The hook value is what down_proj reads; check it is silu(gate) * up.
        prod = F.silu(rec["gate"][LAYER, t]) * rec["up"][LAYER, t]
        swiglu_err = float((prod - rec["value"][LAYER, t]).abs().max())
        assert swiglu_err < 1e-5, swiglu_err
        value = float(rec["value"][LAYER, t, NEURON])
        feed = feeders(model, rec, t)
        reads = readers(model, rec, t)

        # Zero the neuron at this token, rerun, and compare with the estimate.
        abl = record(model, ids, zero=(LAYER, t, NEURON))
        check = []
        for r in reads[:10]:
            k, n = r["layer"], r["neuron"]
            check.append({"reader": r["reader"], "estimate": r["da_first_order"],
                          "measured": round(float(rec["value"][k, t, n] - abl["value"][k, t, n]), 4)})
        measured = {}                 # reader id -> real effect of the target, any reader
        for k in range(LAYER + 1, model.config.num_hidden_layers):
            diff = rec["value"][k, t] - abl["value"][k, t]
            for n in range(diff.numel()):
                measured[f"L{k}/N{n}"] = float(diff[n])
        per_context[name] = {"text": text, "fires_on": FIRES_ON[name], "position": t,
                             "value": round(value, 4), "swiglu_max_abs_err": swiglu_err,
                             "feeders": feed, "readers_top": reads[:OVERLAP],
                             "ablation_check": check}
        top_readers[name] = reads
        measured_all[name] = measured
        gate_moved[name] = rec["gate"][:, t] - abl["gate"][:, t]     # real change of every gate input
        print(f"\n[{name}] {text!r}: L{LAYER}/N{NEURON} = {value:+.3f} "
              f"(silu(gate) {float(F.silu(rec['gate'][LAYER, t, NEURON])):+.3f} x up "
              f"{float(rec['up'][LAYER, t, NEURON]):+.3f})")
        for lane in ("gate", "up"):
            f = feed[lane]
            print(f"  {lane:<4} {f['actual']:+.3f} = embedding {f['embedding']:+.3f} + attention "
                  f"{f['attention']:+.3f} + neurons {f['neurons']:+.3f}   (error {f['abs_err']:.1e})")
            print("       biggest neuron terms: " +
                  ", ".join(f"{x['from']} {x['term']:+.3f}" for x in f["top_neurons"][:TOP]))
        print("  readers, first-order direct effect on their value (via up | via gate):")
        for r, c in zip(reads[:TOP], check):
            print(f"    {r['reader']:<10} {r['da_first_order']:+.3f}  ({r['via_up']:+.3f} | "
                  f"{r['via_gate']:+.3f})  silu(gate) {r['silu_gate']:+.3f}   zeroed for real: "
                  f"{c['measured']:+.3f}")
    guard.tick("contexts")

    names = list(CONTEXTS)
    overlap = {}                  # every pair of contexts, so a third context can be added
    for a, b in combinations(names, 2):
        pair = {}
        for lane in ("gate", "up"):
            sa = {x["from"] for x in per_context[a]["feeders"][lane]["top_neurons"]}
            sb = {x["from"] for x in per_context[b]["feeders"][lane]["top_neurons"]}
            pair[f"feeders_{lane}"] = len(sa & sb)
        ra = {r["reader"] for r in top_readers[a][:OVERLAP]}
        rb = {r["reader"] for r in top_readers[b][:OVERLAP]}
        pair["readers"] = len(ra & rb)
        overlap[f"{a} | {b}"] = pair

    # Same reader, every context: how loud N arrives on its up lane, how open
    # its gate is, and how much of the effect goes through each lane. Readers
    # ranked by the weakest of their up signals.
    by = {c: {r["reader"]: r for r in top_readers[c]} for c in names}
    both = sorted(by[names[0]], key=lambda k: -min(abs(by[c][k]["dz_up"]) for c in names))[:TOP]

    def col(k, field):
        return [by[c][k][field] for c in names]

    shared = [{"reader": k, "dz_up": col(k, "dz_up"), "silu_gate": col(k, "silu_gate"),
               "da_first_order": col(k, "da_first_order"),
               "via_up": col(k, "via_up"), "via_gate": col(k, "via_gate"),
               "measured": [round(measured_all[c][k], 4) for c in names],
               "measured_dz_gate": [round(float(gate_moved[c][reader_id(k)]), 4) for c in names]}
              for k in both]

    def rel(c):
        return abs(c["estimate"] - c["measured"]) / max(abs(c["measured"]), 1e-6)

    checks = [c for ctx in per_context.values() for c in ctx["ablation_check"]]
    nxt = [rel(c) for c in checks if c["reader"].startswith(f"L{LAYER + 1}/")]
    far = [rel(c) for c in checks if not c["reader"].startswith(f"L{LAYER + 1}/")]
    same_sign = sum((c["estimate"] > 0) == (c["measured"] > 0) for c in checks)
    errs = [rel(c) for c in checks]
    print(f"\nshared, of the top {OVERLAP}:")
    for pair, o in overlap.items():
        print(f"  {pair}: gate feeders {o['feeders_gate']}, up feeders {o['feeders_up']}, "
              f"readers {o['readers']}")

    def cells(xs):
        return " | ".join(f"{x:+.3f}" for x in xs)

    print(f"readers that hear N on their up lane in every context ({' | '.join(names)}):")
    for s in shared:
        print(f"  {s['reader']:<10} up gets {cells(s['dz_up'])}   silu(gate) {cells(s['silu_gate'])}   "
              f"effect {cells(s['da_first_order'])}   zeroed for real {cells(s['measured'])}")
        print(f"  {'':<10} effect via up {cells(s['via_up'])}   via gate {cells(s['via_gate'])}   "
              f"N moves its gate input for real {cells(s['measured_dz_gate'])}")
    print(f"first-order estimate against zeroing, top 10 readers x {len(names)} contexts: median relative "
          f"error {median(errs):.0%}, worst {max(errs):.0%}; same sign {same_sign}/{len(checks)}")
    print(f"  readers in layer {LAYER + 1} (n={len(nxt)}): median {median(nxt):.0%}; "
          f"later layers (n={len(far)}): median {median(far):.0%}")

    out = {
        "model": MODEL_ID,
        "device": "cpu",
        "target": f"L{LAYER}/N{NEURON}",
        "mlp": gated,
        "exact": "gate and up pre-activations = sum over writes (embedding, attention per layer, "
                 "each earlier neuron) of v . write, v = row of gate_proj or up_proj * norm weight / rms(h), "
                 "rms at its actual value",
        "approximate": "readers: first order in the target's write, direct path only, each reader's norm "
                       "frozen; checked against zeroing the target at that token",
        "toy": toy_rows,
        "contexts": per_context,
        "overlap_of_top": OVERLAP,
        "overlap": overlap,
        "shared_up_readers": shared,
        "first_order_rel_error": {"median": round(median(errs), 3), "max": round(max(errs), 3),
                                  "same_sign": same_sign, "n": len(checks),
                                  "next_layer": {"n": len(nxt), "median": round(median(nxt), 3)},
                                  "later_layers": {"n": len(far), "median": round(median(far), 3)}},
    }
    out["guard"] = guard.report()
    write_results("wiring", out)
    return out


if __name__ == "__main__":
    run()
