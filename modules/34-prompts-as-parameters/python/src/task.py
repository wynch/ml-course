"""The task, the splits and the fixed evaluator.

TREC question classification (Li and Roth, 2002), coarse labels: what kind
of thing does a question ask for? The prompt's free part, the instruction, is
the parameter; everything else here is frozen.

The evaluator never generates text. It reads the frozen model's next-token
scores at the start of the reply and picks the label whose first token
(lower or capital case) scores highest: a constrained, greedy, deterministic
answer. Same instruction, same examples, same accuracy, every time.
"""

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np

import common

DATASET_ID = "CogComp/trec"
DATASET_REVISION = "65752bf53af25bc935a0dce92fb5b6c930728450"   # refs/convert/parquet
# HF's coarse_label order.
LABELS = ["abbreviation", "entity", "description", "human", "location", "number"]
N_TRAIN_PER_LABEL = 10           # 60 training questions the optimisers see
N_HELDOUT_PER_LABEL = 50         # 300 held-out questions from the same file, never seen
SPLIT_SEED = 0

BASELINE = "Classify the question by the type of its answer."

TEMPLATE = ("{instruction}\n\nQuestion: {question}\n"
            "Reply with one word: abbreviation, entity, description, human, location or number.")


def _read(split: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET_ID, f"default/{split}/0000.parquet", repo_type="dataset",
                           revision=DATASET_REVISION)
    rows = pq.read_table(path).to_pylist()
    return [{"q": r["text"], "y": int(r["coarse_label"])} for r in rows]


@lru_cache(maxsize=1)
def splits() -> dict[str, list[dict]]:
    """train: 10 questions per label from TREC's train file, drawn once with SPLIT_SEED.
    heldout: 50 other questions per label from the same file: same source, same
    balance, so train minus heldout measures overfitting alone.
    test: all 500 questions of TREC's test file, in its own label mix.
    No optimiser ever sees heldout or test."""
    pool = _read("train")
    rng = random.Random(SPLIT_SEED)
    train, heldout = [], []
    for y in range(len(LABELS)):
        rows = rng.sample([r for r in pool if r["y"] == y], N_TRAIN_PER_LABEL + N_HELDOUT_PER_LABEL)
        train += rows[:N_TRAIN_PER_LABEL]
        heldout += rows[N_TRAIN_PER_LABEL:]
    rng.shuffle(train)
    rng.shuffle(heldout)
    return {"train": train, "heldout": heldout, "test": _read("test")}


class Evaluator:
    """The frozen model and its scoring rule. Loaded once per process."""

    def __init__(self, batch: int = 25):
        import mlx.core as mx
        from mlx_lm import load

        self.model, self.tok = load(common.TASK_ID, revision=common.TASK_REVISION)
        # Weights stay 4-bit; activations in float32. In bfloat16 a question's
        # scores moved by up to 2.25 nats with the batch it sat in; in float32
        # by up to 0.15, and no answer changed in either case.
        self.model.set_dtype(mx.float32)
        common.limit_cache()
        self.batch = batch
        self.label_ids = []
        for lab in LABELS:
            ids = {self.tok.encode(v, add_special_tokens=False)[0] for v in (lab, lab.capitalize())}
            self.label_ids.append(sorted(ids))
        firsts = [i for ids in self.label_ids for i in ids]
        assert len(firsts) == len(set(firsts)), "two labels share a first token"
        self.calls = 0          # examples scored: the budget unit

    def encode(self, instruction: str, question: str) -> list[int]:
        msgs = [{"role": "user", "content": TEMPLATE.format(instruction=instruction.strip(),
                                                             question=question)}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            enable_thinking=False)

    def label_logprobs(self, instruction: str, questions: list[str]) -> np.ndarray:
        """[n, 6] log-probabilities of each label's first token(s) at the reply start."""
        import mlx.core as mx

        out = []
        for i in range(0, len(questions), self.batch):
            seqs = [self.encode(instruction, q) for q in questions[i:i + self.batch]]
            lens = [len(s) for s in seqs]
            width = max(lens)
            pad = self.tok.pad_token_id or 0
            # Right padding: with a causal mask, tokens before the pad never see it.
            x = mx.array([s + [pad] * (width - len(s)) for s in seqs])
            # The head runs on the last real position only: logits for every
            # position would take 25 x 200 x 151,936 floats, about 3 GB.
            hidden = self.model.model(x)[mx.arange(len(seqs)), mx.array(lens) - 1]
            if self.model.args.tie_word_embeddings:
                last = self.model.model.embed_tokens.as_linear(hidden)
            else:
                last = self.model.lm_head(hidden)
            lp = last - mx.logsumexp(last, axis=-1, keepdims=True)
            lp = np.array(lp.astype(mx.float32))
            for row in lp:
                out.append([float(np.logaddexp.reduce(row[ids])) for ids in self.label_ids])
            self.calls += len(seqs)
        mx.clear_cache()
        return np.array(out)

    def score(self, instruction: str, rows: list[dict]) -> dict:
        lp = self.label_logprobs(instruction, [r["q"] for r in rows])
        pred = lp.argmax(axis=1)
        y = np.array([r["y"] for r in rows])
        right = (pred == y).astype(int)
        return {"acc": float(right.mean()), "right": right.tolist(), "pred": pred.tolist()}
