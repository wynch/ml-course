//! Exercise C — SOLUTION. Perplexity from dumped logits. Same math as the
//! engine's cmdPerplexity.
//!
//! Run:  zig test ex_c_perplexity.zig

const std = @import("std");

pub fn perplexity(logits: []const f32, targets: []const usize, n_pos: usize, vocab: usize) f64 {
    var nll_sum: f64 = 0;
    for (0..n_pos) |i| {
        const row = logits[i * vocab ..][0..vocab];
        // stable log_softmax: subtract the max before exp()
        var maxl: f32 = -std.math.inf(f32);
        for (row) |v| maxl = @max(maxl, v);
        var denom: f32 = 0;
        for (row) |v| denom += @exp(v - maxl);
        const logp = (row[targets[i]] - maxl) - @log(denom);
        nll_sum += -@as(f64, logp);
    }
    const mean_nll = nll_sum / @as(f64, @floatFromInt(n_pos));
    return @exp(mean_nll);
}

test "perplexity matches the numpy reference" {
    const logits = [_]f32{
        2.0,  1.0, 0.1, -1.0,
        0.5,  0.5, 2.0, 0.0,
        -1.0, 3.0, 0.0, 1.0,
    };
    const targets = [_]usize{ 0, 2, 1 };
    const ppl = perplexity(&logits, &targets, 3, 4);
    try std.testing.expectApproxEqAbs(@as(f64, 1.439520), ppl, 1e-4);
}
