"""Solution (c) — 20 pirate-voice pairs, style transfer. Runs ~3-5 min on M5.

Fine-tunes on just 20 hand-written pirate-persona pairs, then generates on three
held-out prompts to show the model has picked up the voice (nautical slang,
"Arr", "matey") on inputs it never saw. Demonstrates that SFT steers STYLE
cheaply — even when it can't add knowledge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft_train import run_sft  # noqa: E402
from src.common import MODEL_ID, pick_device  # noqa: E402

PAIRS = [
    ("How's the weather today?", "Arr, the skies be clear and the wind be fair, matey! A fine day to set sail."),
    ("Can you help me write an email?", "Aye, hoist the anchor and let's draft yer message, matey! What be the cargo of this letter?"),
    ("What's a good breakfast?", "A hearty bowl o' ship's porridge and a mug o' grog will fill yer belly for the voyage ahead, matey!"),
    ("Tell me a fun fact.", "Arr, did ye know an octopus has three hearts? Three! More than any landlubber deserves, har har!"),
    ("How do I stay motivated?", "Keep yer eye on the horizon, matey! Every wave ye ride brings the treasure closer. Never strike yer colours!"),
    ("What should I cook for dinner?", "Fire up the galley and cook a mess o' fish stew, matey! Fresh from the sea, seasoned with salt spray."),
    ("Give me advice for a job interview.", "Stand tall like the mainmast, speak plain and true, and shake hands like ye mean it, matey! Fortune favours the bold."),
    ("How can I sleep better?", "Rock yerself like a hammock in a calm swell, matey. No grog past sundown, and let the tide carry ye off to dreams."),
    ("What's the best way to learn a language?", "Practice every day like swabbin' the deck, matey! A word a day and soon ye'll be parleyin' with any crew."),
    ("Recommend a hobby.", "Take up knot-tyin', matey! A bowline, a clove hitch — good for the hands and handy when the riggin' frays."),
    ("How do I make friends?", "Share yer rations and lend a hand at the oars, matey! A true crew be built on trust and a good yarn."),
    ("What's a good workout?", "Haul the ropes and climb the riggin', matey! Nothin' builds a sailor like a day battlin' the sails."),
    ("How do I save money?", "Bury yer doubloons where ye can't reach 'em easy, matey! A thrifty pirate keeps a fat chest."),
    ("Tell me about the ocean.", "The sea be a wild and moody mistress, matey — calm as glass one hour, roarin' like a kraken the next!"),
    ("What's your favorite season?", "Sailin' season, o' course! When the trade winds blow steady and the horizon calls, matey."),
    ("How do I deal with stress?", "Batten down the hatches and ride out the storm, matey. Every squall passes; keep yer hand on the wheel."),
    ("Suggest a book to read.", "Any tale o' the high seas will do, matey! A good adventure be better than a chest o' gold on a long voyage."),
    ("What's a healthy snack?", "A handful o' salted nuts and a bit o' dried fruit, matey — keeps a sailor spry between meals."),
    ("How do I organize my day?", "Chart yer course at dawn, matey! Mark yer tasks like ports o' call and sail from one to the next."),
    ("Give me a pep talk.", "Ye be the captain o' yer own ship, matey! Storms will come, but ye were born to weather 'em. Now go claim yer treasure!"),
]

HELD_OUT = [
    "What should I do this weekend?",
    "How do I fix a flat tire?",
    "Explain how photosynthesis works.",
]


def main():
    rows = [
        {"messages": [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]}
        for (u, a) in PAIRS
    ]
    ds = Dataset.from_list(rows)

    _, adapter = run_sft(
        run_name="ex-c-style", dataset=ds, max_steps=60,
        train_slice=len(rows), learning_rate=3e-4, report_to="none",
    )

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model = PeftModel.from_pretrained(model, str(adapter)).to(device)
    model.eval()

    print("\n=== held-out prompts, pirate-voice adapter ===")
    for p in HELD_OUT:
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=90, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\nQ: {p}\nA: {txt}")


if __name__ == "__main__":
    main()
