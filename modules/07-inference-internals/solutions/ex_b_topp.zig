//! Exercise B — SOLUTION. Top-p (nucleus) sampling. This is the same routine as
//! zig/src/sampler.zig::applyTopP.
//!
//! Run:  zig test ex_b_topp.zig

const std = @import("std");

pub fn applyTopP(order: []usize, probs: []f32, p: f32) void {
    // 1. sort ids by probability, descending (vocab is tiny — insertion sort)
    for (order, 0..) |*o, i| o.* = i;
    std.sort.insertion(usize, order, probs, struct {
        fn lessThan(ctx: []f32, a: usize, b: usize) bool {
            return ctx[a] > ctx[b]; // higher probability first
        }
    }.lessThan);

    // 2. walk until cumulative mass reaches p; that index ends the nucleus
    var cum: f32 = 0;
    var cutoff: usize = order.len;
    for (order, 0..) |id, i| {
        cum += probs[id];
        if (cum >= p) {
            cutoff = i + 1;
            break;
        }
    }
    // 3. drop everything past the nucleus, then renormalize the survivors
    for (order[cutoff..]) |id| probs[id] = 0;
    var sum: f32 = 0;
    for (probs) |v| sum += v;
    if (sum > 0) {
        const inv = 1.0 / sum;
        for (probs) |*v| v.* *= inv;
    }
}

test "top-p keeps the smallest set reaching mass p, and renormalizes" {
    var probs = [_]f32{ 0.5, 0.25, 0.15, 0.07, 0.03 };
    var order: [5]usize = undefined;
    applyTopP(&order, &probs, 0.9);

    try std.testing.expect(probs[3] == 0.0);
    try std.testing.expect(probs[4] == 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5 / 0.9), probs[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25 / 0.9), probs[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.15 / 0.9), probs[2], 1e-5);
    var sum: f32 = 0;
    for (probs) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}
