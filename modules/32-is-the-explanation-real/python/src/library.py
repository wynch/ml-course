"""Section 6, the library way: SAELens on GPT-2 small and on a random-weight GPT-2.

One recipe, two models. SAELens's ``LanguageModelSAETrainingRunner`` trains a
TopK SAE (6,144 latents, k = 32) on GPT-2 small's residual stream before
block 8 (``blocks.8.hook_resid_pre``, the layer module 12 read), on the same
TinyStories training stories as the rest of the module, tokenised by GPT-2.
The random model is TransformerLens's ``HookedTransformer`` built from the
same config with freshly initialised weights (TransformerLens's GPT-2 init:
normal, std 0.02; LayerNorm weights 1, biases 0).
Positions holding GPT-2's special token (BOS, which is also the story
separator) are left out of training and scoring.

A third model, added after the first results (not pre-registered), follows
Heap et al.'s main scheme: every weight matrix of GPT-2 redrawn from a Gaussian
with that matrix's trained mean and standard deviation, embeddings included
("matched"). Then the three SAEs, and the published SAE module 12 used
(``gpt2-small-res-jb``, 24,576 latents, trained on OpenWebText), are scored on
held-out stories with the module's detection score and token R2.

    uv run python src/library.py train trained    # about 3 minutes each on MPS
    uv run python src/library.py train random
    uv run python src/library.py train matched
    uv run python src/library.py eval             # results/library.json
"""

from __future__ import annotations

import argparse
import json

from common import RUNS, Guard, append_log, free, get_device, write_results

import numpy as np
import scipy.sparse as sp
import torch

import data
import scores

HOOK, LAYER = "blocks.8.hook_resid_pre", 8
D_SAE, K = 6144, 32
CTX = 128
TRAIN_TOKENS = 1_500_000
N_EVAL_SEQ = 1024                     # 131,072 held-out tokens, before dropping special-token positions
SEED = 0
PUBLISHED = ("gpt2-small-res-jb", "blocks.8.hook_resid_pre")


def load_gpt2(which: str, device):
    from transformer_lens import HookedTransformer

    # center_writing_weights: the processing the published SAE expects (module 12).
    base = HookedTransformer.from_pretrained_no_processing("gpt2", center_writing_weights=True, device=str(device))
    if which == "trained":
        return base.eval()
    if which == "matched":
        # Heap et al.'s "re-randomised including embeddings": every weight matrix redrawn from a Gaussian
        # with that matrix's own trained mean and standard deviation (CPU draw, so it repeats).
        g = torch.Generator().manual_seed(SEED)
        with torch.no_grad():
            for _, p in base.named_parameters():
                q = p.detach().float().cpu()
                p.copy_((torch.randn(q.shape, generator=g) * q.std() + q.mean()).to(p.device))
        return base.eval()
    # The loaded config says init_weights=False, so the constructor leaves the weights as raw memory;
    # init_weights() must be called. Built on the CPU so the draw repeats across runs and machines.
    cfg = base.cfg
    cfg.device = "cpu"
    cfg.initializer_range = 0.02                                       # GPT-2's own init std
    torch.manual_seed(SEED)
    rnd = HookedTransformer(cfg, tokenizer=base.tokenizer, move_to_device=False)
    rnd.init_weights()
    del base
    rnd.cfg.device = str(device)
    return rnd.to(device).eval()


def sequences(split: str) -> np.ndarray:
    """GPT-2 ids of the stories in rows of 128: BOS, then 127 tokens of the running text (story separators kept)."""
    tok = data.gpt2_tokenizer()
    st = data.stories()
    n_train = int(len(st) * 0.9)
    chunk = st[:n_train] if split == "train" else st[n_train:]
    ids = []
    for s in chunk:
        ids.extend(tok.encode(s))
        ids.append(tok.eos_token_id)
    ids = np.array(ids, dtype=np.int64)
    n = len(ids) // (CTX - 1)
    body = ids[: n * (CTX - 1)].reshape(n, CTX - 1)
    return np.concatenate([np.full((n, 1), tok.bos_token_id), body], 1)


def train(which: str) -> None:
    from datasets import Dataset
    from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, LoggingConfig
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

    guard = Guard(f"library.py train {which}")
    dev = get_device()
    model = load_gpt2(which, dev)
    rows = sequences("train")
    ds = Dataset.from_dict({"input_ids": rows.tolist()})
    out_dir = RUNS / f"gpt2_sae_{which}"
    cfg = LanguageModelSAERunnerConfig(
        sae=TopKTrainingSAEConfig(d_in=768, d_sae=D_SAE, k=K, dtype="float32", device=str(dev)),
        model_name="gpt2", hook_name=HOOK, dataset_path="", is_dataset_tokenized=True, streaming=False,
        context_size=CTX, prepend_bos=True, exclude_special_tokens=True,
        training_tokens=TRAIN_TOKENS, train_batch_size_tokens=4096, lr=3e-4, lr_warm_up_steps=50,
        n_batches_in_buffer=256, store_batch_size_prompts=16, device=str(dev), seed=SEED,
        logger=LoggingConfig(log_to_wandb=False), n_checkpoints=0, checkpoint_path=str(out_dir / "ckpt"),
        output_path=str(out_dir), save_final_checkpoint=False, verbose=False,
    )
    sae = LanguageModelSAETrainingRunner(cfg, override_dataset=ds, override_model=model).run()
    sae.save_model(str(out_dir / "final"))
    guard.tick("train")
    g = guard.report()
    (out_dir / "guard.json").write_text(json.dumps(g))
    append_log("library", {"train": which, **{k: g[k] for k in ("seconds", "peak_rss_gb", "peak_mps_driver_gb")}})


@torch.no_grad()
def encode_all(model, sae, rows: np.ndarray, device, bos: int):
    """Residuals before block 8 for every non-special position, the SAE's codes (sparse), and its fit."""
    codes, toks, se, ss, s1, n = [], [], 0.0, 0.0, None, 0
    for s in range(0, len(rows), 32):
        x = torch.from_numpy(rows[s : s + 32]).to(device)
        _, cache = model.run_with_cache(x, names_filter=HOOK, stop_at_layer=LAYER + 1)
        r = cache[HOOK]
        keep = (x != bos)
        r, t = r[keep], x[keep]
        z = sae.encode(r)
        rh = sae.decode(z)
        se += float(((r - rh) ** 2).sum())
        ss += float((r ** 2).sum())
        s1 = r.sum(0) if s1 is None else s1 + r.sum(0)
        n += len(r)
        codes.append(sp.csr_matrix(z.cpu().numpy()))
        toks.append(t.cpu().numpy())
    var = ss - float((s1 ** 2).sum()) / n
    return sp.vstack(codes).tocsc(), np.concatenate(toks), round(1 - se / var, 4)


def evaluate() -> None:
    from sae_lens import SAE

    guard = Guard("library.py eval")
    dev = get_device()
    rows = sequences("val")[:N_EVAL_SEQ]
    tok = data.gpt2_tokenizer()
    bos = tok.bos_token_id
    V = tok.vocab_size
    out = {"hook": HOOK, "d_sae": D_SAE, "k": K, "train_tokens": TRAIN_TOKENS, "eval_rows": int(len(rows)),
           "saes": {}, "browse": {}}
    for which, name in (("trained", "trained"), ("random", "random"), ("matched", "matched"), ("trained", "published")):
        model = load_gpt2(which, dev)
        if name == "published":
            sae = SAE.from_pretrained(*PUBLISHED, device=str(dev))
            if isinstance(sae, tuple):
                sae = sae[0]
        else:
            sae = SAE.load_from_disk(str(RUNS / f"gpt2_sae_{name}" / "final"), device=str(dev))
        sae.eval()
        Z, t, fve = encode_all(model, sae, rows, dev, bos)
        n_lat = Z.shape[1]
        freq = np.asarray((Z > 0).sum(0)).ravel() / Z.shape[0]
        det, r2, expl = scores.score_units(lambda i: Z[:, i].toarray().ravel(), t, V, n_lat)
        row = {"model": which, "n_latents": int(n_lat), "eval_tokens": int(Z.shape[0]), "fve": fve,
               "l0": round(float(Z.nnz / Z.shape[0]), 2), "dead": int((freq == 0).sum()),
               "scorable": int((~np.isnan(det)).sum()),
               "detection": scores.summary(det), "token_r2": scores.summary(r2)}
        if name != "published":
            row["guard_train"] = json.loads((RUNS / f"gpt2_sae_{name}" / "guard.json").read_text())
        out["saes"][name] = row
        rng = np.random.default_rng(SEED)
        pick = rng.permutation(np.flatnonzero(~np.isnan(det)))[:24]
        out["browse"][name] = [{"id": int(i), "det": round(float(det[i]), 3), "r2": round(float(r2[i]), 3),
                                "expl": [tok.decode([int(e)]) for e in expl[i]]} for i in pick]
        print(f"{name}: FVE {fve}, L0 {row['l0']}, detection {row['detection']['median']}, "
              f"token R2 {row['token_r2']['median']}")
        del model, sae, Z
        free()
        guard.tick(name)
    out["guard"] = guard.report()
    write_results("library", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "eval"])
    ap.add_argument("which", nargs="?", choices=["trained", "random", "matched"])
    a = ap.parse_args()
    train(a.which) if a.stage == "train" else evaluate()
