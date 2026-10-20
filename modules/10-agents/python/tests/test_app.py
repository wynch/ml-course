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


class _StubAgent:
    """Stands in for the CodeAgent: no model, one fake step."""

    class _Step:
        step_number = 1
        code_action = "final_answer('42')"
        observations = "42"

    class _Memory:
        steps = []

    def __init__(self):
        self.memory = self._Memory()
        self.memory.steps = [self._Step()]

    def run(self, message):
        return "42"


def test_respond_returns_messages_the_chatbot_accepts(monkeypatch):
    monkeypatch.setattr(app, "_AGENT", _StubAgent())
    history, cleared, trace = app.respond("What is 6 * 7?", [])
    assert cleared == ""
    assert history == [
        {"role": "user", "content": "What is 6 * 7?"},
        {"role": "assistant", "content": "42"},
    ]
    assert "Step 1" in trace
    # The Chatbot must accept this history as its value.
    gr.Chatbot().postprocess(history)
