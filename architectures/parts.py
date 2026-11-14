"""The parts a model has, derived from its record's own fields.

validate.py and build.py share this module. build.py writes the result into the explorer page,
where it drives the "Only models with a part" filter and each definition's list of models that
use it. Every rule reads a field; none reads free text (labels, notes, `what`).

RULES lists each rule once, in plain words; SCHEMA-CHANGES.md copies it. A rule that needs a
judgement (fine-grained experts, linear attention, local/global, dense prefix) is stated here and in
the glossary definition as this page's rule.
"""
from __future__ import annotations

PLACEMENT = {"pre": "pre_norm", "post": "post_norm", "sandwich": "sandwich_norm",
             "post_in_residual": "post_in_residual"}
RECURRENT = {"ssm_mamba", "rwkv", "mlstm", "kda", "gated_deltanet", "linear_attention", "short_conv"}
# Mixers their own papers call linear attention. RWKV and mLSTM keep a similar matrix state, but their
# papers present them as recurrent networks, so this page does not count them.
LINEAR_FAMILY = {"kda", "gated_deltanet"}
# Fine-grained experts: a router that picks at least FG_MIN_ROUTED of at least FG_MIN_EXPERTS experts.
FG_MIN_EXPERTS, FG_MIN_ROUTED = 64, 4
SENTINELS = {None, "", "none", "mixed"}

RULES = [
    ("embedding, residual_add, lm_head", "every record: every decoder-only model has them"),
    ("attention.kind, attention.mixers[].kind, layer_kinds.*.mixer", "each mixer kind named"),
    ("linear_attention", "a mixer its paper calls linear attention: gated_deltanet, kda (rwkv and mlstm count as recurrent only)"),
    ("short_conv", "attention.conv_kernel is set: the mixer runs a short causal convolution"),
    ("hybrid", "attention.kind is hybrid, or the record names two or more mixer kinds (attention.mixers, layer_kinds)"),
    ("sliding_window, chunked_attention", "attention.window_kind or layer_kinds.*.window_kind; a window with no kind counts as sliding"),
    ("local_global", "attention.local_global is set, or a windowed attention kind (no compression) repeats in stack.pattern beside a kind that reads the whole context"),
    ("sparse_topk, csa, ...", "attention.sparse.kind"),
    ("mla", "attention.mla is set"),
    ("kv_sharing", "attention.kv_shared_layers > 0 or any layer_kinds.*.kv_source"),
    ("qk_norm, l2_clip", "attention.qk_norm (rmsnorm counts as qk_norm)"),
    ("attention.other[]", "each id listed"),
    ("position scheme, scaling", "attention.position.scheme and .scaling, layer_kinds.*.position"),
    ("partial_rope", "a RoPE scheme with partial_fraction below 1 (top level or a layer kind)"),
    ("nope", "position scheme nope, or an attention (not recurrent) layer kind with position nope"),
    ("norm kind, placement", "norm.kind; norm.placement through pre_norm, post_norm, sandwich_norm, post_in_residual"),
    ("embedding_norm", "norm.embedding_norm is true"),
    ("ffn kind, activation", "ffn.kind (not moe or none), ffn.activation, layer_kinds.*.ffn"),
    ("moe, router", "moe is set"),
    ("shared_expert", "moe.shared > 0"),
    ("latent_moe", "moe.latent_dim is set"),
    ("dense_prefix", "moe.first_dense > 0"),
    ("fine_grained_experts", f"moe.experts >= {FG_MIN_EXPERTS} and moe.routed_per_token >= {FG_MIN_ROUTED}"),
    ("block_order.steps[].kind", "each step kind"),
    ("specials[].name", "each special"),
    ("vision_tower, audio_tower", "omitted_towers modality (video counts as vision)"),
    ("tokenizer.kind", "the tokenizer named"),
]


def fine_grained(moe: dict | None) -> bool:
    if not moe:
        return False
    e, k = moe.get("experts"), moe.get("routed_per_token")
    return e is not None and k is not None and e >= FG_MIN_EXPERTS and k >= FG_MIN_ROUTED


def derived_parts(rec: dict, gloss: set[str] | None = None) -> set[str]:
    """Glossary ids of the parts the record's fields show. With gloss, keep only glossary ids."""
    s: set[str] = set()

    def add(k):
        if k not in SENTINELS:
            s.add(k)

    a, norm, ffn, moe = rec["attention"], rec["norm"], rec["ffn"], rec.get("moe")
    pos = a.get("position") or {}
    lk = rec.get("layer_kinds") or {}

    s.update(("embedding", "residual_add", "lm_head"))
    add(a["kind"])
    mixers = {m["kind"] for m in a.get("mixers") or []}
    for k in lk.values():
        if k.get("mixer") not in SENTINELS:
            mixers.add(k["mixer"])
    for m in mixers:
        add(m)
    if a.get("conv_kernel"):
        add("short_conv")
    if mixers & LINEAR_FAMILY or a["kind"] in LINEAR_FAMILY:
        add("linear_attention")
    if a["kind"] == "hybrid" or len(mixers) >= 2:
        add("hybrid")

    def window_part(kind, window):
        if kind == "chunked":
            add("chunked_attention")
        elif kind == "sliding" or window:
            add("sliding_window")

    window_part(a.get("window_kind"), a.get("window"))
    for k in lk.values():
        window_part(k.get("window_kind"), k.get("window"))
    # Interleave: a windowed attention kind repeats in stack.pattern beside a kind that sees the whole
    # context. A compressed kind (compress_ratio) keeps a recent band but still reads the whole context.
    in_pattern = {run["type"] for run in rec["stack"]["pattern"]}
    attn = {n: k for n, k in lk.items() if k.get("mixer") not in SENTINELS and k["mixer"] not in RECURRENT}
    local = {n for n, k in attn.items() if (k.get("window") or k.get("window_kind")) and not k.get("compress_ratio")}
    if a.get("local_global") or (local & in_pattern and set(attn) - local):
        add("local_global")

    if a.get("sparse"):
        add(a["sparse"]["kind"])
    if a.get("mla"):
        add("mla")
    if a.get("kv_shared_layers") or any(k.get("kv_source") for k in lk.values()):
        add("kv_sharing")
    q = a.get("qk_norm")
    if q not in SENTINELS:
        add("qk_norm" if q == "rmsnorm" else q)
    for x in a.get("other") or []:
        add(x)

    add(pos.get("scheme"))
    add(pos.get("scaling"))
    rope = {"rope", "mrope", "lerope"}
    if pos.get("scheme") in rope and pos.get("partial_fraction") is not None and pos["partial_fraction"] < 1:
        add("partial_rope")
    if pos.get("scheme") == "nope":
        add("nope")
    for k in lk.values():
        p = k.get("position")
        attention_layer = k.get("mixer") not in RECURRENT and k.get("mixer") != "none"
        if p != "nope" or attention_layer:
            add(p)
        if p in rope and k.get("partial_fraction") is not None and k["partial_fraction"] < 1:
            add("partial_rope")

    add(norm["kind"])
    add(PLACEMENT.get(norm["placement"]))
    if norm.get("embedding_norm"):
        add("embedding_norm")

    if ffn["kind"] != "moe":
        add(ffn["kind"])
    add(ffn.get("activation"))
    for k in lk.values():
        add(k.get("ffn"))
    if moe:
        add("moe")
        add("router")
        if moe.get("shared"):
            add("shared_expert")
        if moe.get("latent_dim"):
            add("latent_moe")
        if moe.get("first_dense"):
            add("dense_prefix")
        if fine_grained(moe):
            add("fine_grained_experts")

    for st in rec["block_order"]["steps"]:
        add(st["kind"])
    for sp in rec["specials"]:
        add(sp["name"])
    for t in rec["omitted_towers"]:
        add("audio_tower" if t["modality"] == "audio" else "vision_tower")
    add((rec.get("tokenizer") or {}).get("kind"))

    return {k for k in s if gloss is None or k in gloss}

