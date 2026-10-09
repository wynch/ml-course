//! Exercise B — implement top-p (nucleus) sampling.
//!
//! Top-k keeps a *fixed* number of candidates. Top-p keeps the *smallest set of
//! tokens whose probabilities add up to at least p* — an adaptive cutoff that
//! is wide when the model is unsure and narrow when it is confident.
//!
//! Given a probability vector (already softmaxed), nucleus filtering is:
//!   1. sort token ids by probability, descending;
//!   2. walk down the sorted list, accumulating probability mass, until the
//!      running sum first reaches p — keep every token up to and including that
//!      one, drop the rest (set their probability to 0);
//!   3. renormalize the survivors so they sum to 1 again.
//!
//! `order` is scratch space of length == probs.len (holds the sorted ids).
//!
//! The test below FAILS until you implement `applyTopP`. Once it passes, the
//! same logic lives in the real engine (zig/src/sampler.zig) behind `--top-p`.
//!
//! Run:  zig test ex_b_topp.zig
//! Solution:  ../solutions/ex_b_topp.zig

const std = @import("std");

/// TODO(you): nucleus (top-p) truncation of `probs`, in place.
/// After this returns, tokens outside the nucleus must be 0 and the surviving
/// probabilities must sum to 1.
pub fn applyTopP(order: []usize, probs: []f32, p: f32) void {
    _ = order;
    _ = probs;
    _ = p;
    // Replace this stub.
}

test "top-p keeps the smallest set reaching mass p, and renormalizes" {
    // probs sorted for clarity: 0.5, 0.25, 0.15, 0.07, 0.03 (sum = 1)
    var probs = [_]f32{ 0.5, 0.25, 0.15, 0.07, 0.03 };
    var order: [5]usize = undefined;

    // p = 0.9 -> cumulative 0.5, 0.75, 0.90 reached at the third token, so the
    // nucleus is {0, 1, 2}; tokens 3 and 4 are dropped.
    applyTopP(&order, &probs, 0.9);

    try std.testing.expect(probs[3] == 0.0);
    try std.testing.expect(probs[4] == 0.0);
    try std.testing.expect(probs[0] > 0 and probs[1] > 0 and probs[2] > 0);

    // survivors renormalize: 0.5/0.9, 0.25/0.9, 0.15/0.9
    try std.testing.expectApproxEqAbs(@as(f32, 0.5 / 0.9), probs[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25 / 0.9), probs[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.15 / 0.9), probs[2], 1e-5);

    var sum: f32 = 0;
    for (probs) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}
