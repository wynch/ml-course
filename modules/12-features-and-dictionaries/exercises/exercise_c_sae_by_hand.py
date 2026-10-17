"""Exercise (c): load GPT-2's layer-8 SAE by hand, without SAELens.

The release ``gpt2-small-res-jb`` is four tensors in one safetensors file.
Write ``encode`` and ``decode`` from them. This SAE subtracts the decoder bias
from the input before encoding.

Predict before you run: x is the decoder bias plus 40 times latent 1986's
decoder row. Which latent comes out on top, and does it come out at 40?
Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_c_sae_by_hand.py
    uv run pytest tests/test_solutions.py -k by_hand -v

The download (151 MB) happens once; ``src/gpt2_sae.py`` already fetched it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
FILE = "blocks.8.hook_resid_pre/sae_weights.safetensors"
LATENT = 1986                     # fires on " so" in "She was so happy"


def load_sae() -> dict[str, torch.Tensor]:
    """W_enc (768, 24576), b_enc (24576,), W_dec (24576, 768), b_dec (768,)."""
    return load_file(hf_hub_download(REPO, FILE))


def encode(sae: dict, x: torch.Tensor) -> torch.Tensor:
    """ReLU((x - b_dec) W_enc + b_enc): this SAE subtracts the decoder bias first."""
    # TODO(you): ReLU((x - b_dec) W_enc + b_enc).
    ...


def decode(sae: dict, f: torch.Tensor) -> torch.Tensor:
    # TODO(you): f W_dec + b_dec.
    ...


def main() -> None:
    sae = load_sae()
    x = sae["b_dec"] + 40 * sae["W_dec"][LATENT]      # a residual that is "latent 1986, strongly"
    f = encode(sae, x)
    v, i = f.topk(3)
    print("top latents:", [(int(a), round(float(b), 1)) for a, b in zip(i, v)])


if __name__ == "__main__":
    main()
