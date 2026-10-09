"""Offline tests for the three tools. No model, fast."""

import tools


def test_calculator_basic():
    assert tools.calculator("2 ** 10 + sqrt(144)") == "1036"
    assert tools.calculator("(5 + 7) * 3") == "36"
    assert tools.calculator("10 / 4") == "2.5"


def test_calculator_is_safe():
    # An injection attempt must NOT execute — it returns an error string.
    out = tools.calculator("__import__('os').system('echo hi')")
    assert out.startswith("Error")
    out2 = tools.calculator("open('/etc/passwd').read()")
    assert out2.startswith("Error")


def test_calculator_div_zero():
    assert tools.calculator("1/0") == "Error: division by zero"


def test_kb_lookup_exact_and_normalized():
    assert tools.kb_lookup("speed of light") == "299792458 meters per second"
    # hyphens/underscores/case are normalized
    assert tools.kb_lookup("Seconds-In-A-Day") == "86400"
    assert tools.kb_lookup("seconds_in_a_day") == "86400"


def test_kb_lookup_miss_suggests():
    out = tools.kb_lookup("earth radius")  # near "earth radius km"
    assert "Did you mean" in out and "earth radius km" in out


def test_list_module_files():
    top = tools.list_module_files("")
    assert "python/" in top and "figures/" in top and "knowledge_base.json" in top
    # path traversal is blocked
    assert tools.list_module_files("../..").startswith("Error")
