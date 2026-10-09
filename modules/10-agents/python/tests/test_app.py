"""Offline test for the Gradio playground.

Building the Blocks does NOT load the model (the agent is created lazily on the
first message), so we can verify UI construction and even a real launch/close
cycle without any weights.
"""

import gradio as gr

import app


def test_build_demo_constructs():
    demo = app.build_demo()
    assert isinstance(demo, gr.Blocks)


def test_launch_and_close():
    # Spec: verify construction + launch(prevent_thread_lock=True) then close.
    # Never leave a server running.
    demo = app.build_demo()
    demo.launch(prevent_thread_lock=True, show_error=True)
    try:
        assert demo.local_url is not None
    finally:
        demo.close()
