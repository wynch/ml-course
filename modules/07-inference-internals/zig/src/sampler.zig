//! Turning a logit vector into the next token id: temperature scaling, then
//! optional top-k and/or top-p (nucleus) truncation, then a multinomial draw.
//!
//! This is the only stochastic part of inference. Everything upstream is
//! deterministic arithmetic; here we roll the dice.

const std = @import("std");
const gpt = @import("gpt.zig");

pub const Config = struct {
    temperature: f32 = 0.8,
    top_k: usize = 0, // 0 = disabled
    top_p: f32 = 0.0, // 0 = disabled (nucleus sampling)
};

pub const Sampler = struct {
    rng: std.Random.DefaultPrng,
    cfg: Config,
    // scratch: (prob, id) pairs, sized to vocab
    order: []usize,

    pub fn init(arena: std.mem.Allocator, seed: u64, cfg: Config, vocab: usize) !Sampler {
        return .{
            .rng = std.Random.DefaultPrng.init(seed),
            .cfg = cfg,
            .order = try arena.alloc(usize, vocab),
        };
    }

    /// Sample the next token id from `logits` (consumed/modified in place).
    pub fn sample(self: *Sampler, logits: []f32) usize {
        const temp = @max(self.cfg.temperature, 1e-8);

        // greedy shortcut when temperature -> 0
        if (self.cfg.temperature <= 1e-6) return argmax(logits);

        for (logits) |*v| v.* /= temp;

        // top-k: keep only the k highest logits, mask the rest to -inf
        if (self.cfg.top_k > 0 and self.cfg.top_k < logits.len) {
            const kth = kthLargest(self.order, logits, self.cfg.top_k);
            for (logits) |*v| {
                if (v.* < kth) v.* = -std.math.inf(f32);
            }
        }

        gpt.softmax(logits); // logits are now probabilities

        // top-p / nucleus: keep the smallest set of tokens whose probability
        // mass reaches p, zero the rest, renormalize.
        if (self.cfg.top_p > 0.0 and self.cfg.top_p < 1.0) {
            applyTopP(self.order, logits, self.cfg.top_p);
        }

        return self.multinomial(logits);
    }

    fn multinomial(self: *Sampler, probs: []const f32) usize {
        const r = self.rng.random().float(f32);
        var acc: f32 = 0;
        for (probs, 0..) |p, i| {
            acc += p;
            if (r < acc) return i;
        }
        return probs.len - 1; // fp slack: fall back to last
    }
};

fn argmax(x: []const f32) usize {
    var best: usize = 0;
    for (x, 0..) |v, i| {
        if (v > x[best]) best = i;
    }
    _ = &x;
    return best;
}

/// Value of the k-th largest logit (used as the top-k cutoff). O(n*k), which is
/// nothing for vocab=65.
fn kthLargest(order: []usize, x: []const f32, k: usize) f32 {
    for (order, 0..) |*o, i| o.* = i;
    // partial selection sort of the top k by logit
    for (0..k) |i| {
        var best = i;
        for (i + 1..order.len) |j| {
            if (x[order[j]] > x[order[best]]) best = j;
        }
        const t = order[i];
        order[i] = order[best];
        order[best] = t;
    }
    return x[order[k - 1]];
}

/// Nucleus (top-p) truncation on a probability vector, in place: sort by
/// probability descending, walk until the cumulative mass reaches p, zero every
/// token past that point, then renormalize.
pub fn applyTopP(order: []usize, probs: []f32, p: f32) void {
    for (order, 0..) |*o, i| o.* = i;
    // sort ids by probability, descending (vocab is tiny — insertion sort is fine)
    std.sort.insertion(usize, order, probs, struct {
        fn lessThan(ctx: []f32, a: usize, b: usize) bool {
            return ctx[a] > ctx[b]; // "less than" in sort order == higher prob first
        }
    }.lessThan);

    var cum: f32 = 0;
    var cutoff: usize = order.len; // index (into sorted order) past which we cut
    for (order, 0..) |id, i| {
        cum += probs[id];
        if (cum >= p) {
            cutoff = i + 1; // keep through this token
            break;
        }
    }
    // zero everything past the nucleus
    for (order[cutoff..]) |id| probs[id] = 0;
    // renormalize the survivors
    var sum: f32 = 0;
    for (probs) |v| sum += v;
    if (sum > 0) {
        const inv = 1.0 / sum;
        for (probs) |*v| v.* *= inv;
    }
}
