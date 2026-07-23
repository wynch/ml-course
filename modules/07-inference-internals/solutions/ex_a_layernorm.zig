//! Exercise A — SOLUTION. LayerNorm over the feature axis.
//!
//! Run:  zig test ex_a_layernorm.zig

const std = @import("std");

pub fn layerNorm(y: []f32, x: []const f32, gamma: []const f32, beta: []const f32, eps: f32) void {
    const n = x.len;
    // 1. mean over the feature axis
    var mean: f32 = 0;
    for (x) |v| mean += v;
    mean /= @floatFromInt(n);
    // 2. biased variance (divide by N) — this is what torch.nn.LayerNorm uses
    var variance: f32 = 0;
    for (x) |v| {
        const d = v - mean;
        variance += d * d;
    }
    variance /= @floatFromInt(n);
    // 3. normalize, then scale (gamma) and shift (beta)
    const inv = 1.0 / @sqrt(variance + eps);
    for (0..n) |i| y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
}

test "layernorm matches PyTorch reference" {
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
