"""Section 2: the loop and the chunked form give the same outputs; the chunked one is faster.

For each rule: random inputs (unit keys and queries, write strengths in (0, 1),
decays in (0.74, 1)), the largest gap between the two forms; then again with
keys that are nearly alike, the case that defeats a careless chunked form. Then module 18's
selective recurrence against the "decay" rule. Then the time for one forward
and backward pass of each form, as the sequence grows, on the CPU and on MPS.

    uv run python src/forms.py --figures
"""

from __future__ import annotations

import argparse
import time

import common
import torch

import memory
from m18_selective import selective_matrix_form

B, H, D = 8, 4, 32
LENGTHS = [256, 512, 1024, 2048]
CHUNK = 64


def inputs(T, device, gen):
    q = memory.unit(torch.randn(B, H, T, D, generator=gen)).to(device)
    k = memory.unit(torch.randn(B, H, T, D, generator=gen)).to(device)
    v = torch.randn(B, H, T, D, generator=gen).to(device)
    beta = torch.rand(B, H, T, generator=gen).to(device)
    log_alpha = (-0.3 * torch.rand(B, H, T, generator=gen)).to(device)
    return q, k, v, beta, log_alpha


def agreement(alike: bool = False) -> dict:
    """Largest gap between the forms. With alike=True, keys and queries lean on one shared
    direction (cosine about 0.999 between any two) and write strengths sit in (0.9, 1)."""
    gen = torch.Generator().manual_seed(0)
    q, k, v, beta, la = inputs(512, "cpu", gen)
    if alike:
        base = torch.randn(B, H, 1, D, generator=gen)
        k = memory.unit(base + 0.02 * torch.randn(B, H, 512, D, generator=gen))
        q = memory.unit(base + 0.02 * torch.randn(B, H, 512, D, generator=gen))
        beta = 0.9 + 0.1 * beta
    out = {}
    for rule, (delta, gated) in memory.RULES.items():
        lg = la if gated else torch.zeros_like(la)
        o1, S1 = memory.recurrent(q, k, v, beta, lg, delta)
        o2, S2 = memory.chunked(q, k, v, beta, lg, delta, chunk=CHUNK)
        out[rule] = {"max_abs_diff_out": (o1 - o2).abs().max().item(),
                     "max_abs_diff_state": (S1 - S2).abs().max().item(),
                     "out_scale": o1.abs().max().item()}
    return out


def product_formula_inverse(A):
    """The first version's (I + A)^-1 = (I - A)(I + A^2)(I + A^4)...: exact in exact arithmetic, not in float32."""
    C = A.shape[-1]
    eye = torch.eye(C, dtype=A.dtype)
    P, Ap, j = eye - A, A, 2
    while j < C:
        Ap = Ap @ Ap
        P = P @ (eye + Ap)
        j *= 2
    return P


def inverse_errors() -> dict:
    """Largest error of each float32 inverse against a float64 solve, on random and on alike keys (C = 64)."""
    gen = torch.Generator().manual_seed(4)
    out = {}
    for name, spread in (("random_keys", None), ("alike_keys", 0.02)):
        if spread is None:
            k = memory.unit(torch.randn(B, H, CHUNK, D, generator=gen))
        else:
            k = memory.unit(torch.randn(B, H, 1, D, generator=gen) + spread * torch.randn(B, H, CHUNK, D, generator=gen))
        beta = 0.9 + 0.1 * torch.rand(B, H, CHUNK, generator=gen)
        A = (beta[..., None] * (k @ k.transpose(-1, -2))).tril(-1)
        eye = torch.eye(CHUNK, dtype=torch.float64)
        ref = torch.linalg.solve_triangular(eye + A.double(), eye.expand(B, H, CHUNK, CHUNK), upper=False,
                                            unitriangular=True)
        out[name] = {"block_form": (memory.unit_lower_inverse(A).double() - ref).abs().max().item(),
                     "product_formula": (product_formula_inverse(A).double() - ref).abs().max().item(),
                     "largest_entry": ref.abs().max().item()}
    return out


def vs_module18() -> float:
    """Module 18's matrix form against this module's decay rule, same inputs, in float64."""
    gen = torch.Generator().manual_seed(1)
    Bs, T, Hh, P, N = 2, 128, 4, 8, 16
    f64 = torch.float64
    xs = torch.randn(Bs, T, Hh, P, generator=gen, dtype=f64)
    Bm, Cm = torch.randn(Bs, T, N, generator=gen, dtype=f64), torch.randn(Bs, T, N, generator=gen, dtype=f64)
    dt = torch.rand(Bs, T, Hh, generator=gen, dtype=f64)
    log_a = -dt * torch.rand(Hh, generator=gen, dtype=f64) * 4
    ref = selective_matrix_form(xs, Bm, Cm, dt, log_a)                        # (B, T, H, P)
    q = Cm[:, None].expand(Bs, Hh, T, N)
    k = Bm[:, None].expand(Bs, Hh, T, N)
    v = xs.transpose(1, 2)                                                    # (B, H, T, P)
    o, _ = memory.recurrent(q, k, v, dt.transpose(1, 2), log_a.transpose(1, 2), delta=False)
    return (o.transpose(1, 2) - ref).abs().max().item()


def timing(device: str) -> dict:
    gen = torch.Generator().manual_seed(2)
    res = {}
    for T in LENGTHS:
        q, k, v, beta, la = inputs(T, device, gen)
        v.requires_grad_(True)
        row = {}
        for form in ("recurrent", "chunked"):
            def go():
                if form == "recurrent":
                    o, _ = memory.recurrent(q, k, v, beta, la, True)
                else:
                    o, _ = memory.chunked(q, k, v, beta, la, True, chunk=CHUNK)
                o.sum().backward()
            go()
            common.sync(device)
            reps = 3
            t0 = time.perf_counter()
            for _ in range(reps):
                go()
            common.sync(device)
            row[form] = round((time.perf_counter() - t0) / reps * 1000, 1)       # ms, forward + backward
        res[str(T)] = row
        print(f"{device} T={T}: loop {row['recurrent']} ms, chunked {row['chunked']} ms", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = common.Guard("forms")
    agree = agreement()
    alike = agreement(alike=True)
    for r, a in agree.items():
        print(f"{r:12s} loop vs chunked: {a['max_abs_diff_out']:.1e} (outputs up to {a['out_scale']:.1f})")
    for r, a in alike.items():
        print(f"{r:12s} alike keys: {a['max_abs_diff_out']:.1e} (outputs up to {a['out_scale']:.1f})")
    m18 = vs_module18()
    inv = inverse_errors()
    print("inverse errors:", inv)
    print(f"module 18 matrix form vs decay rule: {m18:.1e}")
    guard.tick("checks")
    times = {"cpu": timing("cpu")}
    guard.tick("cpu")
    if torch.backends.mps.is_available():
        times["mps"] = timing("mps")
        guard.tick("mps")
    payload = {"batch": B, "heads": H, "d_k": D, "chunk": CHUNK, "lengths": LENGTHS,
               "agreement": agree, "agreement_alike_keys": alike, "module18_vs_decay_max_abs_diff": m18,
               "inverse_errors": inv,
               "ms_forward_backward_gated_delta": times, "guard": guard.report()}
    common.write_results("forms", payload)
    if args.figures:
        import figures

        figures.fig_forms()


if __name__ == "__main__":
    main()
