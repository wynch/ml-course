# mlcourse-10-agents (python project)

The runnable code for **Module 10 — Agents**. See the module walkthrough one
level up: [`../README.md`](../README.md).

```bash
uv sync
uv run python src/handrolled.py    # hand-rolled ReAct loop (local SmolLM2)
uv run python src/smol_way.py      # the smolagents CodeAgent, same tools
uv run python src/eval_lab.py --n 10   # stochastic eval mini-lab
uv run python src/figures.py       # regenerate every figure
uv run python app.py               # Gradio agent playground
uv run pytest -q                   # offline-safe tests (mock the LLM)
```

Layout: `src/` (loop, tools, model loader, smolagents agent, eval, figures),
`tests/` (offline), `app.py` (Gradio). Tools and the knowledge base live in
`src/tools.py` and `../knowledge_base.json`.
