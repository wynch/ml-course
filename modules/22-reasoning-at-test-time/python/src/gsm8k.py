"""GSM8K: the problems, the 8-shot prompt, reading an answer, and the vote.

Every rule here was fixed before the first generation ran:

- problems: the test split, rows 0-199 in file order;
- the 8 worked examples: rows 0-7 of the train split, as chat turns, with the
  "<<...>>" calculator notes removed and the final "#### X" turned into
  "The answer is X.";
- an answer: the number after the last "the answer is", else the last number;
- a vote: the most common answer; outputs with no number do not vote; a tie
  goes to the answer that appeared first.
"""

from __future__ import annotations

import re

from common import DATASET_ID, DATASET_REVISION

N_PROBLEMS = 200
N_SHOTS = 8

_NUM = r"(?<![\d.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
NUM_RE = re.compile(_NUM)
ANS_RE = re.compile(r"answer is[:\s]*\**\s*\$?\s*(" + _NUM + ")", re.IGNORECASE)


def canon(s: str) -> str:
    """'1,250' -> '1250', '18.00' -> '18', '2.50' -> '2.5'."""
    v = float(s.replace(",", ""))
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return repr(round(v, 6))


def extract(text: str) -> str | None:
    hits = ANS_RE.findall(text)
    if hits:
        return canon(hits[-1])
    nums = NUM_RE.findall(text)
    if nums:
        return canon(nums[-1])
    return None


def gold(answer_field: str) -> str:
    """GSM8K's reference answer: the number after '####'."""
    return canon(answer_field.split("####")[-1].strip())


def same(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < 1e-6


def vote(answers: list[str | None]) -> str | None:
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for i, a in enumerate(answers):
        if a is None:
            continue
        counts[a] = counts.get(a, 0) + 1
        first.setdefault(a, i)
    if not counts:
        return None
    return max(counts, key=lambda a: (counts[a], -first[a]))


def cot_solution(answer: str) -> str:
    body, final = answer.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return f"{body}\nThe answer is {final.strip()}."


def load_split(split: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET_ID, f"main/{split}-00000-of-00001.parquet",
                           repo_type="dataset", revision=DATASET_REVISION)
    return pq.read_table(path).to_pylist()


def shots() -> list[dict]:
    return [{"question": r["question"], "cot": cot_solution(r["answer"]),
             "direct": f"The answer is {r['answer'].split('####')[-1].strip()}."}
            for r in load_split("train")[:N_SHOTS]]


def messages(examples: list[dict], kind: str, question: str | None) -> list[dict]:
    """Alternating user and assistant turns; kind is 'cot' or 'direct'."""
    m = []
    for s in examples:
        m.append({"role": "user", "content": f"Question: {s['question']}"})
        m.append({"role": "assistant", "content": s[kind]})
    if question is not None:
        m.append({"role": "user", "content": f"Question: {question}"})
    return m
