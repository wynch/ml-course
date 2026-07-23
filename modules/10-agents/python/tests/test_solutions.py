"""Offline tests for the exercise solutions (pure-Python parts only)."""

import solution_a_unit_tool as sa


def test_unit_converter_km_mi():
    out = sa.unit_converter.forward(10, "km", "mi")
    assert "6.214" in out or "6.21" in out


def test_unit_converter_c_f():
    out = sa.unit_converter.forward(100, "c", "f")
    assert "212" in out


def test_unit_converter_kg_lb():
    out = sa.unit_converter.forward(1, "kg", "lb")
    assert "2.205" in out or "2.20" in out


def test_unit_converter_unsupported():
    out = sa.unit_converter.forward(1, "km", "kg")
    assert out.startswith("Error")


def test_memory_agent_carries_state(monkeypatch):
    # Verify the MEMORY mechanism without a model: fake run_react to echo the
    # injected prompt so we can assert the recap is present on the 2nd call.
    import solution_c_memory as sc

    seen_prompts = []

    class FakeTrace:
        def __init__(self, ans):
            self.answer = ans

    def fake_run_react(task, system_prompt=None, **kw):
        seen_prompts.append(system_prompt or "")
        return FakeTrace("42" if "6 * 7" in task else "420")

    monkeypatch.setattr(sc, "run_react", fake_run_react)
    agent = sc.MemoryAgent()
    agent.run("Use the calculator to compute 6 * 7.")
    agent.run("Use the calculator to multiply that result by 10.")
    # First prompt has no recap; second prompt recalls the first answer.
    assert "Conversation so far" not in seen_prompts[0]
    assert "Conversation so far" in seen_prompts[1] and "42" in seen_prompts[1]
