//! Train the Zig perceptron on the same 200 points as the Python lane and
//! check, at runtime, that the two agree exactly.
//!
//! The dataset and the Python lane's answers both live in `data.zig`, written
//! by `python/scripts/gen_zig_data.py`. Because the algorithm has no learning
//! rate, no shuffling and starts from w = 0, the two languages should not merely
//! be close — they should produce the identical integer mistake count and the
//! same weights to the last bit that f64 arithmetic allows.
//!
//! Run:  zig run main.zig      (or: zig build run)

const std = @import("std");
const data = @import("data.zig");
const perc = @import("perceptron.zig");

/// Weights are sums of the same f64 inputs in the same order, so they should be
/// bit-identical; this leaves room for a different libm only.
const TOL: f64 = 1e-12;

pub fn main() !void {
    // std.Io is how 0.16 reaches the outside world: an Io implementation, then
    // a File writer over a buffer you own.
    var threaded: std.Io.Threaded = .init(std.heap.smp_allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buf);
    const out = &stdout_writer.interface;

    const gpa = std.heap.smp_allocator;
    const points = &data.points;

    const t0 = std.Io.Timestamp.now(io, .awake);
    var run = try perc.train(gpa, points, 100);
    defer run.deinit(gpa);
    const t1 = std.Io.Timestamp.now(io, .awake);

    const R = perc.radius(points);
    const m = perc.maxMargin(points, 20000);
    const bound = (R / m.gamma) * (R / m.gamma);
    const t2 = std.Io.Timestamp.now(io, .awake);

    const train_us = @as(f64, @floatFromInt(t0.durationTo(t1).nanoseconds)) / 1000.0;
    const margin_us = @as(f64, @floatFromInt(t1.durationTo(t2).nanoseconds)) / 1000.0;

    try out.print("Zig perceptron — {d} points, seed 1958, margin >= 0.35\n\n", .{points.len});
    try out.print("  final w    [{d:.9}, {d:.9}, {d:.9}]\n", .{ run.w[0], run.w[1], run.w[2] });
    try out.print("  mistakes   {d}   epochs {d}   converged {}\n", .{ run.mistakes, run.epochs, run.converged });
    try out.print("  accuracy   {d:.1}%\n", .{perc.accuracy(run, points) * 100.0});
    try out.print("  snapshots  {d} weight vectors kept\n\n", .{run.snapshots.items.len});
    try out.print("  R          {d:.9}\n", .{R});
    try out.print("  gamma      {d:.9}  (hull {d:.9})\n", .{ m.gamma, m.gamma_hull });
    try out.print("  bound      (R/gamma)^2 = {d:.4}\n", .{bound});
    try out.print("  used       {d:.2}% of the guarantee\n\n", .{@as(f64, @floatFromInt(run.mistakes)) / bound * 100.0});
    try out.print("  timing     train {d:.1} us, margin solve {d:.1} us\n\n", .{ train_us, margin_us });

    // ---- parity with the Python lane -------------------------------------
    const py = data.python;
    const dw = @max(@max(@abs(run.w[0] - py.w[0]), @abs(run.w[1] - py.w[1])), @abs(run.w[2] - py.w[2]));
    const dR = @abs(R - py.radius);
    const dg = @abs(m.gamma - py.gamma);
    const db = @abs(bound - py.bound);

    try out.print("parity vs python/scripts/train_perceptron.py\n", .{});
    try out.print("  mistakes   zig {d} == python {d}   {s}\n", .{ run.mistakes, py.mistakes, if (run.mistakes == py.mistakes) "ok" else "MISMATCH" });
    try out.print("  epochs     zig {d} == python {d}   {s}\n", .{ run.epochs, py.epochs, if (run.epochs == py.epochs) "ok" else "MISMATCH" });
    try out.print("  max |dw|   {e:.3}\n", .{dw});
    try out.print("  |dR|       {e:.3}\n", .{dR});
    try out.print("  |dgamma|   {e:.3}\n", .{dg});
    try out.print("  |dbound|   {e:.3}\n", .{db});
    try out.flush();

    if (run.mistakes != py.mistakes or run.epochs != py.epochs) return error.LaneMismatch;
    if (dw > TOL or dR > TOL or dg > TOL or db > 1e-9) return error.LaneMismatch;

    try out.print("\nboth lanes agree.\n", .{});
    try out.flush();
}

test "the two lanes agree on the mistake count" {
    const gpa = std.testing.allocator;
    var run = try perc.train(gpa, &data.points, 100);
    defer run.deinit(gpa);
    try std.testing.expect(run.converged);
    try std.testing.expectEqual(data.python.mistakes, run.mistakes);
    try std.testing.expectEqual(@as(f64, 1.0), perc.accuracy(run, &data.points));
}

test "the mistake count respects Novikoff's bound" {
    const gpa = std.testing.allocator;
    var run = try perc.train(gpa, &data.points, 100);
    defer run.deinit(gpa);
    const bound = perc.novikoffBound(&data.points);
    try std.testing.expect(@as(f64, @floatFromInt(run.mistakes)) <= bound);
}

test "XOR keeps the perceptron cycling forever" {
    const gpa = std.testing.allocator;
    const xor = [_]data.Point{
        .{ .x1 = 0, .x2 = 0, .y = -1 },
        .{ .x1 = 0, .x2 = 1, .y = 1 },
        .{ .x1 = 1, .x2 = 0, .y = 1 },
        .{ .x1 = 1, .x2 = 1, .y = -1 },
    };
    var run = try perc.train(gpa, &xor, 50);
    defer run.deinit(gpa);
    try std.testing.expect(!run.converged);
    try std.testing.expectEqual(@as(usize, 200), run.mistakes); // 4 every epoch
    // and w is back where it started, because the sum of y·x over XOR is zero
    try std.testing.expectEqual(@as(f64, 0), run.w[0] + run.w[1] + run.w[2]);
}
