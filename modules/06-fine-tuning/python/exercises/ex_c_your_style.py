"""Exercise (c) — teach the model YOUR voice with 20 examples.

Small-scale SFT can't add knowledge, but it's remarkably good at adopting a
*style*. Write 20 instruction/response pairs in a consistent, distinctive voice
(a pirate, a terse senior engineer, a Victorian butler, haiku-only — your call),
fine-tune on just those 20, and watch the model start answering in that voice on
prompts it has never seen.

Runtime on the M5: ~3-5 min (tiny dataset, ~60 steps).

Fill in the TODOs, then:  uv run python exercises/ex_c_your_style.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft_train import run_sft  # noqa: E402


def my_pairs() -> list[dict]:
    # TODO(you): write 20 (user, assistant) pairs in ONE consistent voice.
    # Return chat-format rows: {"messages": [{"role": "user", ...},
    #                                          {"role": "assistant", ...}]}
    pairs = [
        # ("How's the weather?", "Arr, the skies be clear, matey!"),
        # ... 20 of these ...
    ]
    return [
        {"messages": [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]}
        for (u, a) in pairs
    ]


def main():
    rows = my_pairs()
    assert len(rows) >= 20, "write at least 20 pairs!"
    ds = Dataset.from_list(rows)

    # TODO(you): fine-tune on your dataset. A tiny set overfits fast, so keep it
    # short: run_sft(run_name="ex-c-style", dataset=ds, max_steps=60,
    #                train_slice=len(rows), report_to="none").
    ...

    # TODO(you): load base + your adapter (see generate_compare.py for the
    # pattern) and generate on 3 HELD-OUT prompts to show the voice transfer.
    ...


if __name__ == "__main__":
    main()
