# The backward pass, side by side

Reverse-mode automatic differentiation ("backprop") is the *same* algorithm in
every framework. This module implements it twice — once in Python, once in Zig —
so you can see the algorithm underneath the language. The Python version is the
short poem; the Zig version is the same poem with every word's ownership,
lifetime, and type spelled out. Reading them together is the fastest way to
understand what an autograd engine actually *is*.

There are three moving parts:

1. **The node structure** — what one scalar remembers about how it was made.
2. **Chain-rule accumulation** — how a node pushes gradient to its inputs.
3. **The topological sort + reverse walk** — the order that makes it all correct.

---

## 1. The node structure

**Python** — a node is an object. How to differentiate it is a *closure*
(`_backward`) captured at construction time. The closure holds references to the
parent objects; Python's garbage collector keeps everything alive.

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None      # a closure, filled in by each op
        self._prev = set(_children)         # references to input nodes
        self._op = _op
```

**Zig** — there are no GC'd closures, so a node cannot carry a captured function.
Instead it stores its operation *as data* (an `Op` enum) plus **indices** to its
children. All nodes live in one flat `ArrayList` owned by the `Engine`; a
"Value" is just a `usize` index into that list.

```zig
pub const Op = enum { leaf, add, mul, powc, expf, tanhf, reluf };

pub const Node = struct {
    data: f64,
    grad: f64 = 0,
    op: Op = .leaf,     // the operation, stored as a tag instead of a closure
    a: usize = 0,       // index of first input (not a pointer — an index)
    b: usize = 0,       // index of second input
    aux: f64 = 0,       // a constant the op needs (the exponent, for powc)
    nkids: u2 = 0,      // how many of a,b are real children
};

pub const V = usize;    // a handle to a node is its position in the list
```

**What the compiled language makes explicit:** Python hides *where nodes live*
and *how backward is represented*. Zig forces both into the open — nodes live in
one contiguous array you can point at, and "how to differentiate this op" becomes
a value (`Op`) you can print, compare, and `switch` on. The closure Python
attaches per node is replaced by one shared `switch` statement (see §2). That is
strictly less memory per node and no per-node function allocation.

---

## 2. Chain-rule accumulation

Every operation contributes a local derivative. The chain rule says: to get the
gradient at an input, multiply the *local* derivative by the gradient that has
already arrived at the output (`out.grad`), and **accumulate** (`+=`) because a
node can feed several consumers.

**Python** — the local rule is baked into each op's closure when the forward op
runs. Here is multiplication (the product rule):

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad  += other.data * out.grad   # d(a*b)/da = b
        other.grad += self.data  * out.grad   # d(a*b)/db = a
    out._backward = _backward
    return out
```

**Zig** — the forward op only records the node; the local rule lives in one
central `switch` that runs during backward. Same product rule, addressed by index:

```zig
pub fn mul(self: *Engine, a: V, b: V) !V {
    const d = self.nodes.items[a].data * self.nodes.items[b].data;
    return self.push(.{ .data = d, .op = .mul, .a = a, .b = b, .nkids = 2 });
}

// ... later, inside backwardNode(i):
.mul => {
    const g = out.grad;
    self.nodes.items[out.a].grad += self.nodes.items[out.b].data * g; // d/da = b
    self.nodes.items[out.b].grad += self.nodes.items[out.a].data * g; // d/db = a
},
```

Line up the two `+=` pairs: they are identical arithmetic. The only difference is
that Python writes `self.grad` / `other.grad` and Zig writes
`self.nodes.items[out.a].grad` — a name versus an array index. Every op in this
module (`add`, `powc`, `expf`, `tanhf`, `reluf`) follows the same shape.

**What the compiled language makes explicit:** *aliasing and mutation*. In Zig
you can see that backward is nothing but in-place `+=` into a shared array, so a
node feeding two consumers naturally sums their contributions. Python's `+=`
does the same thing, but the closure indirection hides that it is all one array
of accumulators.

---

## 3. Topological sort + reverse walk

Gradient must flow from the output back toward the leaves, and a node may only
be processed once **all** of its consumers have contributed. The fix is a
topological sort: order nodes so every child comes before its parents, then walk
it in reverse.

**Python** — recursion builds the order; a `set` tracks visited nodes; the GC
handles all the temporary structures.

```python
def backward(self):
    topo, visited = [], set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)         # child appended before parent

    build(self)
    self.grad = 1.0               # seed: d(out)/d(out) = 1
    for node in reversed(topo):   # parents before children
        node._backward()
```

**Zig** — the same recursion, but you *allocate* the `visited` array and the
`topo` list, and you *free* them (here via `defer`). Visited-tracking is a plain
`[]bool` indexed by node id instead of a hash set.

```zig
pub fn backward(self: *Engine, root: V) !void {
    const n = self.nodes.items.len;
    for (self.nodes.items) |*node| node.grad = 0;      // start clean

    const visited = try self.alloc.alloc(bool, n);
    defer self.alloc.free(visited);
    @memset(visited, false);

    var topo: std.ArrayList(V) = .empty;
    defer topo.deinit(self.alloc);
    try self.buildTopo(root, visited, &topo);          // child pushed before parent

    self.nodes.items[root].grad = 1.0;                 // seed: d(out)/d(out) = 1

    var k: usize = topo.items.len;
    while (k > 0) {                                     // parents before children
        k -= 1;
        self.backwardNode(topo.items[k]);
    }
}

fn buildTopo(self: *Engine, v: V, visited: []bool, topo: *std.ArrayList(V)) !void {
    if (visited[v]) return;
    visited[v] = true;
    const node = self.nodes.items[v];
    if (node.nkids >= 1) try self.buildTopo(node.a, visited, topo);
    if (node.nkids >= 2) try self.buildTopo(node.b, visited, topo);
    try topo.append(self.alloc, v);                    // child appended before parent
}
```

**What the compiled language makes explicit:** *allocation and ownership*. The
Python version silently allocates a list and a set that the GC later reclaims.
The Zig version names the allocator, states that `visited` and `topo` are
temporary (`defer`-freed), and — crucially — reuses one big node arena across
training steps via `clearRetainingCapacity()`. Backprop's memory cost (one
gradient accumulator and one topo slot per node) is invisible in Python and
right there in Zig.

---

## The one idea to keep

Backprop is **message passing on a DAG**: each node receives `out.grad` from
above, multiplies by its own local derivative, and adds the result into its
inputs' accumulators — processed in reverse topological order so every message
has arrived before a node forwards it on. Python expresses this with closures
and a garbage collector; Zig expresses it with an enum tag, an index-addressed
array, and an explicit allocator. **Same algorithm. The compiler just refuses to
let you look away from the bookkeeping.**
