"""Offline tests for the hand-rolled ReAct loop.

We monkeypatch the model so no weights are needed: a scripted "model" emits a
Thought+Action, then a Final Answer. This tests the *loop mechanics* — parsing,
tool dispatch, observation injection, termination — deterministically and fast.
"""

import handrolled
import model_local


def test_parser_action_and_final():
    m = handrolled._ACTION_RE.search("Thought: go\nAction: calculator[2 + 2]\n")
    assert m.group(1) == "calculator" and m.group(2).strip() == "2 + 2"
    f = handrolled._FINAL_RE.search("Thought: done\nFinal Answer: 42")
    assert f.group(1).strip() == "42"


def test_loop_runs_tool_then_answers(monkeypatch):
    # Scripted model: first call issues an Action, second call gives the answer.
    calls = {"n": 0}

    def fake_generate(messages, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Thought: I will add.\nAction: calculator[2 + 2]"
        # The observation (4) should be in the injected context by now.
        assert any("Observation: 4" in m["content"] for m in messages)
        return "Thought: got it.\nFinal Answer: 4"

    monkeypatch.setattr(model_local, "generate", fake_generate)
    tr = handrolled.run_react("add two and two", max_steps=4)
    assert tr.ok and tr.answer == "4"
    kinds = [s.kind for s in tr.steps]
    assert "action" in kinds and "observation" in kinds and "answer" in kinds
    assert tr.n_model_calls == 2


def test_unknown_tool_is_reported(monkeypatch):
    def fake_generate(messages, **kw):
        # Always ask for a tool that doesn't exist -> loop reports error, no crash.
        return "Action: nope[x]"

    monkeypatch.setattr(model_local, "generate", fake_generate)
    tr = handrolled.run_react("x", max_steps=2)
    obs = [s.content for s in tr.steps if s.kind == "observation"]
    assert obs and "unknown tool" in obs[0]


def test_trace_to_text_roundtrip():
    tr = handrolled.Trace(task="t")
    tr.add("thought", "thinking")
    tr.add("action", "2+2", tool="calculator")
    tr.add("observation", "4", tool="calculator")
    tr.add("answer", "4")
    tr.answer, tr.ok = "4", True
    txt = handrolled.trace_to_text(tr)
    assert "Action: calculator[2+2]" in txt and "Final Answer: 4" in txt
