//! A scalar autograd engine in Zig — the same algorithm as python/src/micrograd,
//! but with the machinery the compiler forces us to make explicit.
//!
//! Where Python stores a Python *closure* on every node (`_backward`), Zig has no
//! garbage-collected closures, so instead each node stores its operation as an
//! `Op` enum plus indices to its children. The backward pass is one `switch` over
//! that enum. Nodes live in a single flat `ArrayList` owned by the `Engine`; a
//! "Value" is just a `usize` index into that list. Rebuilding the graph each step
//! is `clearRetainingCapacity()` — the arena/list is reused, nothing leaks.

const std = @import("std");

/// The operation that produced a node. `.leaf` nodes are inputs/parameters.
pub const Op = enum { leaf, add, mul, powc, expf, tanhf, reluf };

/// One node = one scalar. `a`/`b` index its inputs; `aux` holds a constant
/// (the exponent for `powc`). `nkids` says how many of a,b are real children.
pub const Node = struct {
    data: f64,
    grad: f64 = 0,
    op: Op = .leaf,
    a: usize = 0,
    b: usize = 0,
    aux: f64 = 0,
    nkids: u2 = 0,
};

/// A handle to a node is just its index. Reads nicer than a bare `usize`.
pub const V = usize;

pub const Engine = struct {
    nodes: std.ArrayList(Node),
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator) Engine {
        return .{ .nodes = .empty, .alloc = alloc };
    }

    /// Drop every node but keep the backing memory — call at the start of a step.
    pub fn reset(self: *Engine) void {
        self.nodes.clearRetainingCapacity();
    }

    fn push(self: *Engine, node: Node) !V {
        const idx = self.nodes.items.len;
        try self.nodes.append(self.alloc, node);
        return idx;
    }

    // -------------------------------------------------------------- forward ops
    pub fn value(self: *Engine, x: f64) !V {
        return self.push(.{ .data = x, .op = .leaf });
    }

    pub fn add(self: *Engine, a: V, b: V) !V {
        const d = self.nodes.items[a].data + self.nodes.items[b].data;
        return self.push(.{ .data = d, .op = .add, .a = a, .b = b, .nkids = 2 });
    }

    pub fn mul(self: *Engine, a: V, b: V) !V {
        const d = self.nodes.items[a].data * self.nodes.items[b].data;
        return self.push(.{ .data = d, .op = .mul, .a = a, .b = b, .nkids = 2 });
    }

    /// x ** exponent, exponent is a compile-time-unknown constant f64.
    pub fn powc(self: *Engine, a: V, exponent: f64) !V {
        const d = std.math.pow(f64, self.nodes.items[a].data, exponent);
        return self.push(.{ .data = d, .op = .powc, .a = a, .aux = exponent, .nkids = 1 });
    }

    pub fn expf(self: *Engine, a: V) !V {
        const d = std.math.exp(self.nodes.items[a].data);
        return self.push(.{ .data = d, .op = .expf, .a = a, .nkids = 1 });
    }

    pub fn tanhf(self: *Engine, a: V) !V {
        const d = std.math.tanh(self.nodes.items[a].data);
        return self.push(.{ .data = d, .op = .tanhf, .a = a, .nkids = 1 });
    }

    pub fn reluf(self: *Engine, a: V) !V {
        const x = self.nodes.items[a].data;
        return self.push(.{ .data = if (x < 0) 0 else x, .op = .reluf, .a = a, .nkids = 1 });
    }

    // ------------------------------------------------------ convenience reads
    pub fn data(self: *Engine, i: V) f64 {
        return self.nodes.items[i].data;
    }
    pub fn grad(self: *Engine, i: V) f64 {
        return self.nodes.items[i].grad;
    }

    // -------------------------------------------------------------- backward
    /// Reverse-mode autodiff. Builds a topological order (children before
    /// parents), seeds d(root)/d(root)=1, then walks parents-before-children
    /// applying the chain rule per op.
    pub fn backward(self: *Engine, root: V) !void {
        const n = self.nodes.items.len;
        // zero all grads (fresh tapes start at 0, but be explicit & safe)
        for (self.nodes.items) |*node| node.grad = 0;

        const visited = try self.alloc.alloc(bool, n);
        defer self.alloc.free(visited);
        @memset(visited, false);

        var topo: std.ArrayList(V) = .empty;
        defer topo.deinit(self.alloc);
        try self.buildTopo(root, visited, &topo);

        self.nodes.items[root].grad = 1.0;

        // reverse topo => parents before children
        var k: usize = topo.items.len;
        while (k > 0) {
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
        try topo.append(self.alloc, v); // child appended before parent
    }

    /// Push `out.grad` into the grads of out's children — the chain rule, once.
    fn backwardNode(self: *Engine, i: V) void {
        const out = self.nodes.items[i];
        const g = out.grad;
        switch (out.op) {
            .leaf => {},
            .add => {
                // d(a+b)/da = 1, d(a+b)/db = 1
                self.nodes.items[out.a].grad += g;
                self.nodes.items[out.b].grad += g;
            },
            .mul => {
                // product rule
                self.nodes.items[out.a].grad += self.nodes.items[out.b].data * g;
                self.nodes.items[out.b].grad += self.nodes.items[out.a].data * g;
            },
            .powc => {
                // d(x**n)/dx = n * x**(n-1)
                const x = self.nodes.items[out.a].data;
                self.nodes.items[out.a].grad += out.aux * std.math.pow(f64, x, out.aux - 1) * g;
            },
            .expf => {
                // d(e**x)/dx = e**x = out.data
                self.nodes.items[out.a].grad += out.data * g;
            },
            .tanhf => {
                // d(tanh(x))/dx = 1 - tanh(x)**2
                self.nodes.items[out.a].grad += (1.0 - out.data * out.data) * g;
            },
            .reluf => {
                // gradient flows only where the input was positive
                self.nodes.items[out.a].grad += (if (out.data > 0) g else 0);
            },
        }
    }
};
