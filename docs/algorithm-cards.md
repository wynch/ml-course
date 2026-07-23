# Algorithm cards

An **algorithm card** is this course's signature teaching device: the *same*
algorithm, presented **side by side in two languages**, so that the language
difference itself teaches you the algorithm.

- **Python** — dynamic, concise, expressive. It hides memory management, lets
  shapes flex at runtime, and reads close to the math. Great for seeing the
  *idea* with the least ceremony.
- **Zig** — static, explicit, unforgiving. Every allocation is written out,
  ownership is visible, and there is no garbage collector or hidden reshape to
  paper over what is really happening. Great for seeing the *machine*.

Put them next to each other and the gaps light up. Where Python writes `x @ y`,
Zig makes you loop, allocate the result buffer, and decide who frees it. Where
Python grows a list, Zig makes you size it. The Python version tells you *what*
the algorithm computes; the Zig version tells you *what the computer must do* to
compute it. Neither is "the real one" — together they are.

## How to read a card

Each card is a two-column walkthrough with running commentary:

- **Left: Python.** The reference implementation. Read it first to grasp the
  shape of the algorithm.
- **Right: Zig.** The same steps, but with allocation, iteration, and ownership
  made explicit. Read it second and map each line back to the Python.
- **Commentary** in between calls out exactly what Python hid — a temporary
  buffer, a broadcast, an in-place update, a growth of a container — and why it
  matters for performance and correctness.

The goal is not to make you a Zig programmer. It is to make the abstractions in
your Python code **stop being magic**. Once you have hand-allocated the
scratch buffer for a backward pass, `loss.backward()` never looks the same
again.

## The cards this course ships

| Card | Module | Algorithm | What Zig reveals |
|------|--------|-----------|------------------|
| **Backward pass** | [01 · autograd](../modules/01-autograd) | Reverse-mode autodiff over a scalar graph | The topological order, the gradient accumulation, and who owns each node. |
| **Blocked matmul** | [02 · neural-networks](../modules/02-neural-networks) | Cache-blocked matrix multiplication | Memory layout, tiling, and why the naive triple loop is slow. |
| **BPE training loop** | [03 · tokenization](../modules/03-tokenization) | Byte-Pair Encoding merge learning | Pair counting, the merge table, and the cost of growing vocabularies. |
| **Transformer inference loop** | [07 · inference-internals](../modules/07-inference-internals) | Autoregressive decode with a KV cache | Buffer reuse across steps, the cache's exact shape, and where the memory goes. |

Each card lives with its module — the Python side under `python/`, the Zig side
under `zig/` — and the module README walks you through both. Read the module
first; the card is the moment the two implementations are laid against each
other.
