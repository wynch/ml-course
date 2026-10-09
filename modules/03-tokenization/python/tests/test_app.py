"""The Gradio playground must construct and launch headlessly, then close.

We never leave a server running: launch with prevent_thread_lock=True and
close in a finally block.
"""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import app as playground  # noqa: E402


def test_tokenize_produces_spans():
    toks = playground.build_tokenizers()
    for which in ("your-BPE", "HF-trained", "SmolLM3"):
        spans = playground.tokenize("Hello, world! 42", which, toks)
        assert len(spans) > 0
        assert all(isinstance(s, tuple) and len(s) == 2 for s in spans)


def test_app_launches_and_closes():
    demo = playground.build_demo()
    try:
        demo.launch(
            prevent_thread_lock=True,  # never block the test on the server loop
            show_error=True,
            quiet=True,
        )
        assert demo.local_url is not None
    finally:
        demo.close()
