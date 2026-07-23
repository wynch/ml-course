//! Exercise A — implement the LayerNorm kernel in Zig.
//!
//! LayerNorm is the first kernel every transformer position passes through. It
//! normalizes a feature vector to zero mean / unit variance, then rescales it
//! with a learned gamma (weight) and shifts it with a learned beta (bias):
//!
//!     mean = mean(x)                          over the feature axis
//!     var  = mean((x - mean)^2)               BIASED variance (divide by N)
//!     y    = (x - mean) / sqrt(var + eps) * gamma + beta
//!
//! The test below checks your kernel against reference values dumped from
//! PyTorch's `torch.nn.LayerNorm` (eps = 1e-5). It FAILS until you implement
//! `layerNorm`.
//!
//! Run:  zig test ex_a_layernorm.zig
//! Solution:  ../solutions/ex_a_layernorm.zig

const std = @import("std");

/// TODO(you): implement LayerNorm over the whole slice `x`, writing the result
/// into `y`. `gamma` and `beta` have the same length as `x`. Use the BIASED
/// variance (divide the sum of squared deviations by N, not N-1) to match
/// torch.nn.LayerNorm.
pub fn layerNorm(y: []f32, x: []const f32, gamma: []const f32, beta: []const f32, eps: f32) void {
    _ = x;
    _ = gamma;
    _ = beta;
    _ = eps;
    // Replace this stub. Right now it writes zeros, so the test fails.
    for (y) |*v| v.* = 0;
}

test "layernorm matches PyTorch reference" {
    // Reference values produced by torch.nn.LayerNorm(8, eps=1e-5). See
    // python/scripts/ (the same math the engine uses in gpt.zig::layerNorm).
    const x = [_]f32{ 3.381100, -0.931900, 0.065600, 0.815000, -1.577800, 0.004100, -0.001800, -3.509400 };
    const g = [_]f32{ 1.089600, 1.151900, 0.940500, 0.783000, 0.894100, 1.204800, 0.856700, 0.976100 };
    const b = [_]f32{ 0.172500, -0.190000, 0.040200, 0.180100, -0.107900, 0.019400, 0.163700, -0.146700 };
    const expected = [_]f32{ 2.292734, -0.633570, 0.185057, 0.617824, -0.764307, 0.164920, 0.264444, -1.882290 };

    var y: [8]f32 = undefined;
    layerNorm(&y, &x, &g, &b, 1e-5);

    for (y, expected) |got, want| {
        try std.testing.expectApproxEqAbs(want, got, 1e-4);
    }
}
