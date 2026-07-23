//! A tiny MLP over the scalar Engine. Parameters (`params`) persist across
//! training steps; the *graph* is rebuilt every step. Each step we call
//! `materialize` to create fresh leaf nodes seeded from the persistent params,
//! run forward/backward, then read grads back out to update `params`.

const std = @import("std");
const eng = @import("engine.zig");
const Engine = eng.Engine;
const V = eng.V;

pub const MLP = struct {
    sizes: []const usize, // e.g. { 2, 8, 1 }
    params: []f64, // persistent weights+biases, laid out layer/neuron/[w..,b]
    grads: []f64, // grads from the last backward, same layout
    pnodes: []V, // node index of each param for the current step
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, sizes: []const usize, rng: std.Random) !MLP {
        var np: usize = 0;
        var li: usize = 1;
        while (li < sizes.len) : (li += 1) np += sizes[li] * (sizes[li - 1] + 1);

        const params = try alloc.alloc(f64, np);
        const grads = try alloc.alloc(f64, np);
        const pnodes = try alloc.alloc(V, np);

        // init: weights ~ uniform(-1, 1), biases = 0, in layout order
        var pi: usize = 0;
        li = 1;
        while (li < sizes.len) : (li += 1) {
            const nin = sizes[li - 1];
            var j: usize = 0;
            while (j < sizes[li]) : (j += 1) {
                var w: usize = 0;
                while (w < nin) : (w += 1) {
                    params[pi] = rng.float(f64) * 2.0 - 1.0;
                    pi += 1;
                }
                params[pi] = 0.0; // bias
                pi += 1;
            }
        }
        return .{ .sizes = sizes, .params = params, .grads = grads, .pnodes = pnodes, .alloc = alloc };
    }

    pub fn nparams(self: *const MLP) usize {
        return self.params.len;
    }

    /// Create a fresh leaf node for every parameter from its current value.
    pub fn materialize(self: *MLP, e: *Engine) !void {
        for (self.params, 0..) |p, i| self.pnodes[i] = try e.value(p);
    }

    /// Build the forward graph for one input; return the output node index.
    /// `input` are node indices (length == sizes[0]). Hidden layers use tanh,
    /// the output layer is linear. `scratch` supplies temporary index buffers.
    pub fn forward(self: *MLP, e: *Engine, scratch: std.mem.Allocator, input: []const V) !V {
        var prev = try scratch.dupe(V, input);
        var pi: usize = 0;
        var li: usize = 1;
        while (li < self.sizes.len) : (li += 1) {
            const nin = self.sizes[li - 1];
            const nout = self.sizes[li];
            const is_last = (li == self.sizes.len - 1);
            const out = try scratch.alloc(V, nout);
            var j: usize = 0;
            while (j < nout) : (j += 1) {
                // layout per neuron: nin weights then 1 bias
                var acc = self.pnodes[pi + nin]; // start from the bias node
                var w: usize = 0;
                while (w < nin) : (w += 1) {
                    const term = try e.mul(self.pnodes[pi + w], prev[w]);
                    acc = try e.add(acc, term);
                }
                pi += nin + 1;
                out[j] = if (is_last) acc else try e.tanhf(acc);
            }
            prev = out;
        }
        return prev[0];
    }

    /// After backward(), copy each parameter's grad out of the tape.
    pub fn readGrads(self: *MLP, e: *Engine) void {
        for (self.pnodes, 0..) |node, i| self.grads[i] = e.grad(node);
    }

    /// One SGD step: params -= lr * grad.
    pub fn sgdStep(self: *MLP, lr: f64) void {
        for (self.params, self.grads) |*p, g| p.* -= lr * g;
    }
};
