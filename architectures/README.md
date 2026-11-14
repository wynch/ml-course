# Architectures

One JSON record per large language model: its shape, its block, and a source for
every number. The [LLM architecture explorer](../explorables/05b-architecture-explorer.html)
draws its diagrams, fact sheets and comparisons from these records.

| File | Holds |
|---|---|
| `schema.json` | the record format (JSON Schema 2020-12), version 1.1.0; changes are listed in `SCHEMA-CHANGES.md` |
| `glossary.json` | 101 part names (`gqa`, `rmsnorm`, `moe` ...), one plain line each, with the modules that teach them; a merged name sits in its term's `aliases` |
| `model_list.json` | the 62 models planned, with the best source for each |
| `records/<id>.json` | one record per model; the file name is the record's `id` |
| `validate.py` | the checks every record must pass: schema, a source for every field, glossary ids, stacks that sum to `layers`, `layer_kinds` that match the stack |
| `SCHEMA-CHANGES.md` | what changed between schema versions |
| `build.py` | validates the records and writes them into the explorer page |

## How a record is sourced

- Numbers come from the model's own files: `config.json`, the weights index
  (`model.safetensors.index.json`), the model card, the paper or the lab's post.
  A closed model enters only when a paper gives its shape.
- Each source sits once in `refs`, with its kind (`config`, `paper`,
  `model_card`, `official_post`, `code`, `weights_index`, `secondary`,
  `derived`). `field_sources` maps each field, or a prefix such as
  `attention.*`, to the refs it rests on. Every non-null field needs one.
- A number nobody published is `null`, with the reason in `unknowns`. Where
  sources disagree, `notes` says which one the record follows.
- `block_order.steps` lists one block's parts in order, using glossary ids.
  `block_order.basis` says what fixes that order: `code` or `paper` are firm;
  `tensor_names` means the parts are certain and the order inferred. The page
  shows an "order inferred" badge for it.
- A `secondary` ref is a summary, such as Sebastian Raschka's gallery data. When a
  core field rests on nothing else, the page shows a "secondary sources" badge; any
  fact sheet row that rests on nothing else says so, and the page footer lists them.
  At schema 1.1.0 that is eight release dates and one licence link, all from his gallery.
- A stack with two or more layer types needs `layer_kinds`: the mixer, feed-forward part,
  heads, window and positions of each type. The explorer draws one block per type from it.
- A gated config read through a public copy is a `config_mirror` ref; its `note` names the
  original file. The page labels it as a copy.
- Others' diagrams go in `figure_links`, with the author's name. They are linked,
  never copied.
- The loops drawn inside recurrent mixers (Mamba, RWKV, mLSTM, Gated DeltaNet, KDA) are
  each family's textbook update rule, taken from its paper, not from the record; the
  record's own notes print under the drawing.

## Add a model

1. Copy a record of a similar model to `records/<id>.json`; set `id` to the
   file name.
2. Fill every field from the sources above. Use glossary ids for kinds; if a
   part has no id, report it rather than editing `glossary.json`. Take attention
   width from heads × head size, never from `d_model`.
3. Set `parent_id` to the model it extends, so the explorer can walk the lineage.
4. Check it with `uv run --with jsonschema python -I architectures/validate.py`, then
   rebuild (below). `build.py` runs the same checks; a record that fails is left out of
   the page and listed with its problems.

## Rebuild

From `course/`:

| Command | Does |
|---|---|
| `uv run architectures/build.py` | checks every record and rewrites the data block of the explorer page |
| `uv run architectures/build.py --check` | exits 1 if the page is out of date; CI runs this |

`build.py` needs `jsonschema`; `uv` installs it from the script's header. The page
holds all its data inline and fetches nothing, so it works offline and in the
reader.
