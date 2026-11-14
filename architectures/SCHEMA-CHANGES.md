# Schema changes

## 1.2.0 (2026-11-14)

Every other change adds an optional field; one field, `teaches`, is removed. No record used it any more, so every 1.1.0 record without hand tags still passes the schema. `schema.json` now carries `"version": "1.2.0"`.

### Parts come from the fields

`parts.py` derives the set of glossary parts a record has from its own fields. `validate.py` and `build.py` both import it; `build.py` writes the sets into the page (`DATA.parts`), where they drive the "Only models with a part" filter, each definition's list of models that use the part, the lineage story and the guided paths. No rule reads free text.

`teaches` held hand tags; only 4 records named fine-grained experts. A tag the fields already show is redundant, and one they do not show contradicts them, so the field is gone: the schema forbids it, and a record that still carries it fails. A part with no field goes in `specials`.

| Part | Rule |
|---|---|
| embedding, residual_add, lm_head | every record |
| mixer kinds | `attention.kind`, `attention.mixers[].kind`, `layer_kinds.*.mixer` |
| linear_attention | a mixer its paper calls linear attention: gated_deltanet, kda; RWKV and mLSTM keep a similar matrix state but count as recurrent only |
| short_conv | `attention.conv_kernel` is set (review: Jamba and both Nemotrons gain it; their Mamba layers run the convolution) |
| hybrid | `attention.kind` is hybrid, or two or more mixer kinds |
| sliding_window, chunked_attention | `window_kind` at top level or on a layer kind; a window with no kind counts as sliding |
| local_global | `attention.local_global` is set, or a windowed attention kind (no `compress_ratio`) repeats in `stack.pattern` beside a kind that reads the whole context |
| sparse kind, mla, kv_sharing | `attention.sparse.kind`; `attention.mla`; `kv_shared_layers` or any `kv_source` |
| qk_norm, l2_clip | `attention.qk_norm` (rmsnorm counts as qk_norm) |
| extras | each id in `attention.other` |
| position | `position.scheme` and `scaling`, `layer_kinds.*.position`; partial_rope when a RoPE scheme has `partial_fraction` below 1 |
| nope | scheme nope, or an attention (not recurrent) layer kind with position nope |
| norm | `norm.kind`; `norm.placement` through pre_norm, post_norm, sandwich_norm, post_in_residual; embedding_norm when `norm.embedding_norm` is true |
| feed-forward | `ffn.kind` (not moe or none), `ffn.activation`, `layer_kinds.*.ffn` |
| moe, router | `moe` is set |
| shared_expert, latent_moe | `moe.shared` > 0; `moe.latent_dim` set |
| dense_prefix | `moe.first_dense` > 0 |
| fine_grained_experts | `moe.experts` >= 64 and `moe.routed_per_token` >= 4; the DeepSeekMoE paper (2401.06066) defines the idea, splitting each expert into m and routing to m times as many, but sets no threshold |
| steps, specials | each `block_order.steps[].kind` and `specials[].name` |
| towers, tokenizer | `omitted_towers` modality (video counts as vision); `tokenizer.kind` |

Four rules need a judgement; each glossary definition states its rule: fine_grained_experts, linear_attention, local_global, dense_prefix.

Counts that moved, page before and after:

| Part | Before | After | Why |
|---|---|---|---|
| fine_grained_experts | 4 | 26 | the count rule replaces hand tags |
| router | 1 | 30 | every MoE record |
| linear_attention | 1 | 7 | Gated DeltaNet and KDA records (review: RWKV-6 and xLSTM dropped, since their papers call them recurrent networks) |
| embedding_norm | 2 | 5 | read from `norm.embedding_norm` |
| nope | 15 | 10 | `nope_layers` text about recurrent or convolution layers no longer counts (LFM2.5, the four Qwen hybrids) |
| dense_prefix | 16 | 14 | Jamba and Llama 4 Maverick alternate dense and MoE layers; that is not a prefix |
| embedding, lm_head | 0 | 62 | every record |
| short_conv | 11 | 14 | review: read from `attention.conv_kernel`, so Jamba and both Nemotrons count |

### New fields

| Field | Meaning | Records |
|---|---|---|
| `attention.sparse.block_size` | tokens per key block when the indexer picks blocks; `top_k` then counts blocks | minimax-m3 (128, config `sparse_block_size`) |
| `layer_kinds.*.sparse` | this layer kind runs the top-k selection in `attention.sparse`; required on every kind when `attention.sparse` is set | deepseek-v3.2, deepseek-v4-pro, deepseek-v4.1-flash, ernie-5.5-flash, glm-5, kimi-k3.1-pro, minimax-m3, mistral-large-4 |
| `attention.conv_kernel` | width of the short causal convolution the mixer runs | 14: inkling, inkling-2-small, jamba-v0.1, kimi-k3, kimi-k3.1-pro, kimi-linear-48b-a3b, lfm2.5-1.2b (3), mamba-2.8b, both Nemotrons, the four Qwen hybrids; 4 unless noted |
| `moe.first_dense` | count of leading layers that keep a dense FFN; 0 when none; required on every MoE record | all 30 MoE records |
| `attn_res_block` | layers per block of block attention residuals | inkling-2-small (6), kimi-k3 (12), kimi-k3.1-pro (12) |
| `logit_softcap` | cap c on the final logits, c · tanh(z / c); set exactly when `specials` name logit_softcap | 8 records: 30, except grok-2.5 (50) and muse-spark-open (20) |

Each new value was already in the record, in a label, note or special, with its source; the page used to read it out of that text with a regular expression. The field now holds it, and `field_sources` names the source.

### Glossary

The four judgement rules above are stated in their definitions as this page's rules. Module 19 (section 5, mixture of experts) now sits in the module links of moe, router, shared_expert, fine_grained_experts and dense_prefix; module 18 (section 5, sliding windows) in those of sliding_window, local_global, chunked_attention and rolling_buffer_cache.

### Validation and build

- `validate.py` also fails on a MoE record without `moe.first_dense`, a layer kind without `sparse` when `attention.sparse` is set (or with `sparse: true` when it is not), and a `logit_softcap` or `attn_res_block` without its special.
- `build.py` stops when a record fails (exit 1, page untouched); `--lenient` leaves failing records out instead, for local work. `build.py --check` fails when a record fails or the page is stale; CI runs it.

## 1.1.0 (2026-11-14)

Every change adds an optional field or an allowed value. No field was removed or renamed, and every 1.0 record shape still validates. `schema.json` now carries `"version": "1.1.0"`.

### New fields

| Field | Meaning | Records that use it |
|---|---|---|
| `layer_kinds` | settings per layer kind, keyed by the type names in `stack`: `mixer`, `ffn`, `heads`, `kv_heads`, `head_dim`, `v_head_dim`, `window`, `window_kind`, `compress_ratio`, `position`, `theta`, `partial_fraction`, `kv_source`, `note`. Required when the stack has two or more types. The top-level `attention` fields still describe the main kind. | 35: deepseek-moe-16b, deepseek-v2, deepseek-v3, deepseek-v3.2, deepseek-v4-pro, deepseek-v4.1-flash, ernie-5.5-flash, gemma-2-9b, gemma-3-27b, gemma-4-31b, gemma-4-e4b, glm-4.5, glm-5, gpt-3-175b, gpt-oss-120b, inkling, inkling-2-small, jamba-v0.1, kimi-k2, kimi-k3, kimi-k3.1-pro, kimi-linear-48b-a3b, lfm2.5-1.2b, llama-4-maverick, minimax-m3, mistral-large-4, muse-spark-open, nemotron-3-nano-30b-a3b, nemotron-3-ultra-550b-a55b, olmo-3-7b, qwen3-next-80b-a3b, qwen3.5-397b-a17b, qwen3.6-35b-a3b, qwen3.8-27b, smollm3-3b |
| `layer_kinds.*.mixer = "none"` | a layer with no token mixer (Nemotron's MoE-only layers) | nemotron-3-nano-30b-a3b, nemotron-3-ultra-550b-a55b |
| `layer_kinds.*.ffn = "none"` | a layer with no feed-forward part (Nemotron's Mamba and attention layers) | the same two |
| `layer_kinds.*.compress_ratio` | past tokens pooled into one cached entry | deepseek-v4-pro, deepseek-v4.1-flash, ernie-5.5-flash |
| `layer_kinds.*.kv_source` | which layers reuse which layer's keys and values | deepseek-v4.1-flash, gemma-4-e4b, mistral-large-4 |
| `layer_kinds.*.v_head_dim` | value head size when it differs from query/key | qwen3-next-80b-a3b, qwen3.5-397b-a17b, qwen3.6-35b-a3b, qwen3.8-27b |
| `attention.attention_free` | `true`: no layer uses softmax attention; `kind` names the recurrent mixer | mamba-2.8b, rwkv-6-7b, xlstm-7b |
| `ffn.ffn_free` | `true`: blocks have no feed-forward part; `ffn.kind` is `"none"` | mamba-2.8b |
| `attention.window_kind` | `sliding` (last W tokens) or `chunked` (own block of W tokens) | sliding: 13 records with a window; chunked: llama-4-maverick |
| `attention.kv_shared_layers` | count of layers that reuse another layer's KV | deepseek-v4.1-flash (34), gemma-4-e4b (18), mistral-large-4 (9) |
| `attention.sparse.index_shared_layers` | count of layers that run no indexer and reuse another layer's top-k | deepseek-v4.1-flash (30), mistral-large-4 (28) |
| `moe.shared_expert_hidden_dim` | width of one shared expert, only when it differs from `expert_hidden_dim` | nemotron-3-nano-30b-a3b (3712), nemotron-3-ultra-550b-a55b (10240) |
| `moe.layer_step` | MoE on every n-th layer after the dense prefix | jamba-v0.1 (2), llama-4-maverick (2) |
| `params.effective` | the lab's "effective" size | gemma-4-e4b (4.5B; `active` is now the dense text count) |
| `params.active_prefill`, `params.active_decode` | active count per phase when they differ | deepseek-v4.1-flash (8B, 16B) |
| ref kind `config_mirror` | a gated config read through a public copy; `url` is the copy, `note` names the original | gemma-2-9b, gemma-3-27b, gemma-7b, llama-2-7b, llama-2-70b, llama-3-8b, llama-3.1-405b, llama-4-maverick, muse-spark-open |

No field was added for heads × head_dim ≠ d_model. It holds in many records (PaLM, Gemma, DeepSeek, Qwen hybrids, ERNIE ...); a drawing should take the attention width from `heads × head_dim` (or `layer_kinds`), never from `d_model`.

### Values the explorer should know

- `block_order.steps[].only_on` may name several stack types joined by `", "` (jamba-v0.1: `"mamba, attention"`; llama-4-maverick: `"chunked_moe, global_nope_moe"`), or a layer list starting with a digit (deepseek-v4.1-flash engram: `"2 and 15 (0-based 1 and 14)"`). Every other value is one stack type.
- New step kinds: `hca`, `csa`, `csa2`, `sliding_window` (deepseek-v4-pro, deepseek-v4.1-flash, ernie-5.5-flash), `layer_scalar` (gemma-4-31b, gemma-4-e4b). ERNIE's step `hybrid` became one `hca` step and one `mla` step with `only_on`; Kimi K3 and K3.1 Pro got separate `kda` and `mla` steps the same way.
- `attention.local_global` is now `null` for KDA or DeltaNet hybrids (kimi-linear-48b-a3b, kimi-k3, kimi-k3.1-pro, the four Qwen hybrids): the ratio there is between mixer kinds, which `attention.mixers` and `layer_kinds` already give.
- `norm.placement` keeps `pre`, `post`, `sandwich`, `post_in_residual`; they map to glossary ids `pre_norm`, `post_norm`, `sandwich_norm`, `post_in_residual`.
- Sentinels outside the glossary: `none` (attention.position.scheme, attention.qk_norm, ffn.kind, layer_kinds mixer/ffn/position) and `mixed` (attention.position.scheme).

### Glossary

68 terms before, 101 after. A merged id sits in the term's new `aliases` list; records use only the term's id.

| Term | Merged ids |
|---|---|
| `attn_softcap` (new) | `softcap_attn` |
| `relu2` (new) | `squared_relu`; also replaces `ffn.activation` null or `relu` in both Nemotrons and RWKV-6 |
| `embedding_scale` (new) | `embedding_multiplier` |
| `output_scale` (new) | `mup_output` |
| `head_norm` (new) | `multi_head_norm` |
| `csa` (new) | `csa_hca` |
| `mxfp4` (renamed "4-bit floats") | `fp4_experts`, `nvfp4`, `fp4_kv_cache` |
| `sparse_topk` | `hierarchical_sparse_indexer`, `msa` |
| `mtp` | `dspark` |
| `log_length_scaling` | `attn_temperature_len` (SGLang's kernel scales Grok's scores by log(pos)/log(1024)) |

Other new terms: `gate_softcap`, `no_biases`, `qkv_bias`, `k_eq_v`, `vha`, `multimax`, `residual_init_scaling`, `document_mask`, `z_loss`, `qk_clip`, `rolling_buffer_cache`, `grouped_output_projection`, `layer_scalar`, `residual_gate`, `mhc`, `causal_encoder_decoder`, `embedding_norm`, `zero_centered_rmsnorm`, `chunked_attention`, `kv_sharing`, `hca`, `csa2`, `token_shift`, `data_dependent_decay`, `mrope`, `lerope`, `hash_routing`.

Removed from records: the `untied_embeddings` special (gpt-j-6b, gpt-neox-20b; untied is the default, the fact moved to notes) and `model_surgery` (mistral-large-4; moved to notes).

### Validation

`validate.py` with no arguments checks every record; the old three-argument form still works. It now also fails on:

- an id outside the glossary in attention.kind, attention.mixers, attention.other, attention.sparse.kind, attention.position.scheme and scaling, attention.qk_norm, norm.kind, ffn.kind, ffn.activation, tokenizer.kind, block_order steps, specials, teaches and layer_kinds; a merged id names its replacement;
- a stack whose counts do not sum to `layers`;
- a multi-type stack without `layer_kinds`, or `layer_kinds` keys that differ from the stack types;
- an `only_on` that names no stack type;
- a ref without a web URL, a release date outside 2017 to today, a `parent_id` that is not a record;
- `model_list.json` name, org, date or open_weights that differ from the record.

At 1.1.0, `build.py` checked only step kinds and `teaches` against the glossary; since then it runs `validate.record_problems` too.

### model_list.json

`date_source` is gone; each record's `release_date.note` holds it. `name`, `org`, `date` and `open_weights` are kept for models that have no record yet (build.py reads them); for the 62 with records they now copy the record, and validate.py fails if they drift.
