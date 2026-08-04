//! The perceptron in Zig 0.16 — same algorithm as python/src/perceptron,
//! with the arithmetic and the allocations in plain sight.
//!
//! A point becomes a 3-vector the moment it enters the algorithm: `[x1, x2, 1]`.
//! That third coordinate is the bias, and once you accept it the learning rule
//! has no special cases left — `w += y * x`, three multiply-adds.
//!
//! Nothing here allocates except the snapshot history, which uses the 0.16
//! `ArrayList` shape: the list is created with `.empty` and every method that
//! might grow it takes the allocator as its first argument, so the list itself
//! stores no allocator and costs three words.

const std = @import("std");
const data = @import("data.zig");

/// An augmented point: [x1, x2, 1].
pub const Vec3 = [3]f64;

pub fn dot(a: Vec3, b: Vec3) f64 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

pub fn norm(a: Vec3) f64 {
    return @sqrt(dot(a, a));
}

/// `[x1, x2, 1]` — the bias is just another feature, permanently switched on.
pub fn augment(p: data.Point) Vec3 {
    return .{ p.x1, p.x2, 1.0 };
}

pub const Run = struct {
    w: Vec3 = .{ 0, 0, 0 },
    mistakes: usize = 0,
    epochs: usize = 0,
    converged: bool = false,
    /// w after every update — `.empty` costs nothing until the first append.
    snapshots: std.ArrayList(Vec3) = .empty,

    pub fn deinit(self: *Run, gpa: std.mem.Allocator) void {
        self.snapshots.deinit(gpa);
    }

    pub fn predict(self: Run, x: Vec3) f64 {
        return if (dot(x, self.w) > 0) 1.0 else -1.0;
    }
};

/// Cycle over the points in order; update on every mistake; stop after a clean
/// pass. Deterministic: no shuffling, no learning rate, w starts at zero.
pub fn train(
    gpa: std.mem.Allocator,
    points: []const data.Point,
    max_epochs: usize,
) !Run {
    var run: Run = .{};
    try run.snapshots.append(gpa, run.w);

    var epoch: usize = 0;
    while (epoch < max_epochs) : (epoch += 1) {
        var epoch_mistakes: usize = 0;
        for (points) |p| {
            const x = augment(p);
            if (p.y * dot(x, run.w) <= 0) { // a tie counts as a mistake
                run.w[0] += p.y * x[0]; // the entire learning rule
                run.w[1] += p.y * x[1];
                run.w[2] += p.y * x[2];
                run.mistakes += 1;
                epoch_mistakes += 1;
                try run.snapshots.append(gpa, run.w);
            }
        }
        run.epochs = epoch + 1;
        if (epoch_mistakes == 0) {
            run.converged = true;
            break;
        }
    }
    return run;
}

pub fn accuracy(run: Run, points: []const data.Point) f64 {
    var correct: usize = 0;
    for (points) |p| {
        if (run.predict(augment(p)) == p.y) correct += 1;
    }
    return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(points.len));
}

/// R = max‖x‖ over the augmented data.
pub fn radius(points: []const data.Point) f64 {
    var best: f64 = 0;
    for (points) |p| {
        const n = norm(augment(p));
        if (n > best) best = n;
    }
    return best;
}

pub const Margin = struct { gamma: f64, gamma_hull: f64, u: Vec3 };

/// Frank–Wolfe on the convex hull of {y·x}: the distance from the origin to
/// that hull *is* the largest margin any unit separator achieves. Allocation
/// free — the iterate is three floats.
pub fn maxMargin(points: []const data.Point, iters: usize) Margin {
    // start at the shortest y·x
    var p: Vec3 = .{ 0, 0, 0 };
    var best: f64 = std.math.inf(f64);
    for (points) |pt| {
        const x = augment(pt);
        const z: Vec3 = .{ pt.y * x[0], pt.y * x[1], pt.y * x[2] };
        const n2 = dot(z, z);
        if (n2 < best) {
            best = n2;
            p = z;
        }
    }

    var it: usize = 0;
    while (it < iters) : (it += 1) {
        // the hull vertex that leans furthest away from the current point
        var jbest: f64 = std.math.inf(f64);
        var zj: Vec3 = .{ 0, 0, 0 };
        for (points) |pt| {
            const x = augment(pt);
            const z: Vec3 = .{ pt.y * x[0], pt.y * x[1], pt.y * x[2] };
            const s = dot(z, p);
            if (s < jbest) {
                jbest = s;
                zj = z;
            }
        }
        const d: Vec3 = .{ zj[0] - p[0], zj[1] - p[1], zj[2] - p[2] };
        const dd = dot(d, d);
        if (dd < 1e-12) break;
        var step = -dot(p, d) / dd; // exact line search
        if (step <= 0) break;
        if (step > 1) step = 1;
        p = .{ p[0] + step * d[0], p[1] + step * d[1], p[2] + step * d[2] };
    }

    const hull = norm(p);
    const u: Vec3 = .{ p[0] / hull, p[1] / hull, p[2] / hull };
    var achieved: f64 = std.math.inf(f64);
    for (points) |pt| {
        const m = pt.y * dot(augment(pt), u);
        if (m < achieved) achieved = m;
    }
    return .{ .gamma = achieved, .gamma_hull = hull, .u = u };
}

/// Novikoff: a perceptron on separable data makes at most (R/γ)² mistakes.
pub fn novikoffBound(points: []const data.Point) f64 {
    const R = radius(points);
    const m = maxMargin(points, 20000);
    return (R / m.gamma) * (R / m.gamma);
}
