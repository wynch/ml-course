//! Matmul performance sidebar for module 02.
//!
//! The forward pass of our 784 -> 256 MLP is dominated by one matrix multiply:
//! X(M, K) @ W(K, N) with M = batch, K = 784, N = 256. Here we implement that
//! same multiply three ways and time them, to show *why* a naive triple loop
//! leaves most of the machine on the table — and why real frameworks call into
//! hand-tuned BLAS kernels instead of a textbook loop.
//!
//!   1. naive  (i, j, k order): the inner loop strides down a column of B,
//!              one cache line per element — terrible locality.
//!   2. ikj    (reordered):     the inner loop walks rows of B and C
//!              contiguously; the compiler can vectorize it.
//!   3. blocked (tiled ikj):    process cache-sized tiles so operands stay hot
//!              in L1/L2 across reuse.
//!
//! Build & run:  zig build run

const std = @import("std");

const M: usize = 256; // batch rows
const K: usize = 784; // input features (28*28)
const N: usize = 256; // hidden units
const BS: usize = 64; // block size for the tiled variant
const REPEATS: usize = 20;

fn zero(c: []f32) void {
    @memset(c, 0);
}

/// Naive i,j,k. Inner loop accumulates over k; B access b[k*N + j] jumps by N
/// floats each step — cache-hostile.
fn matmulNaive(a: []const f32, b: []const f32, c: []f32) void {
    zero(c);
    for (0..M) |i| {
        for (0..N) |j| {
            var acc: f32 = 0;
            for (0..K) |k| {
                acc += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = acc;
        }
    }
}

/// Reordered i,k,j. Inner loop over j touches b[k*N + j] and c[i*N + j]
/// contiguously; auto-vectorizes well.
fn matmulIKJ(a: []const f32, b: []const f32, c: []f32) void {
    zero(c);
    for (0..M) |i| {
        for (0..K) |k| {
            const aik = a[i * K + k];
            const brow = b[k * N ..][0..N];
            const crow = c[i * N ..][0..N];
            for (0..N) |j| {
                crow[j] += aik * brow[j];
            }
        }
    }
}

/// Blocked / tiled ikj. Same math, but operands are reused while hot in cache.
fn matmulBlocked(a: []const f32, b: []const f32, c: []f32) void {
    zero(c);
    var ii: usize = 0;
    while (ii < M) : (ii += BS) {
        const i_end = @min(ii + BS, M);
        var kk: usize = 0;
        while (kk < K) : (kk += BS) {
            const k_end = @min(kk + BS, K);
            var jj: usize = 0;
            while (jj < N) : (jj += BS) {
                const j_end = @min(jj + BS, N);
                for (ii..i_end) |i| {
                    for (kk..k_end) |k| {
                        const aik = a[i * K + k];
                        for (jj..j_end) |j| {
                            c[i * N + j] += aik * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

fn maxAbsDiff(x: []const f32, y: []const f32) f32 {
    var m: f32 = 0;
    for (x, y) |xi, yi| {
        const d = @abs(xi - yi);
        if (d > m) m = d;
    }
    return m;
}

/// Time `f` over REPEATS runs, return best (minimum) nanoseconds.
fn bench(io: std.Io, f: *const fn ([]const f32, []const f32, []f32) void, a: []const f32, b: []const f32, c: []f32) u64 {
    // warm up
    f(a, b, c);
    var best: u64 = std.math.maxInt(u64);
    var r: usize = 0;
    while (r < REPEATS) : (r += 1) {
        const t0 = std.Io.Clock.now(.awake, io);
        f(a, b, c);
        const t1 = std.Io.Clock.now(.awake, io);
        const ns: u64 = @intCast(t1.nanoseconds - t0.nanoseconds);
        if (ns < best) best = ns;
        std.mem.doNotOptimizeAway(c[0]);
    }
    return best;
}

pub fn main() !void {
    const alloc = std.heap.page_allocator;

    var threaded: std.Io.Threaded = .init(alloc, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const a = try alloc.alloc(f32, M * K);
    const b = try alloc.alloc(f32, K * N);
    const c = try alloc.alloc(f32, M * N);
    const c_ref = try alloc.alloc(f32, M * N);
    defer alloc.free(a);
    defer alloc.free(b);
    defer alloc.free(c);
    defer alloc.free(c_ref);

    var prng = std.Random.DefaultPrng.init(0);
    const rand = prng.random();
    for (a) |*x| x.* = rand.floatNorm(f32);
    for (b) |*x| x.* = rand.floatNorm(f32);

    // reference result for correctness check
    matmulIKJ(a, b, c_ref);

    const flops: f64 = 2.0 * @as(f64, M) * @as(f64, N) * @as(f64, K);

    const Row = struct { name: []const u8, ns: u64, diff: f32 };
    var rows: [3]Row = undefined;

    const ns_naive = bench(io, matmulNaive, a, b, c);
    rows[0] = .{ .name = "naive (ijk)", .ns = ns_naive, .diff = maxAbsDiff(c, c_ref) };

    const ns_ikj = bench(io, matmulIKJ, a, b, c);
    rows[1] = .{ .name = "reordered (ikj)", .ns = ns_ikj, .diff = maxAbsDiff(c, c_ref) };

    const ns_blk = bench(io, matmulBlocked, a, b, c);
    rows[2] = .{ .name = "blocked (tiled)", .ns = ns_blk, .diff = maxAbsDiff(c, c_ref) };

    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    const out = &w.interface;

    try out.print("\nMatmul benchmark  C({d}x{d}) = A({d}x{d}) * B({d}x{d})\n", .{ M, N, M, K, K, N });
    try out.print("best of {d} runs, ReleaseFast, single thread\n\n", .{REPEATS});
    try out.print("{s:<18} {s:>10} {s:>12} {s:>10} {s:>12}\n", .{ "variant", "time (ms)", "GFLOP/s", "speedup", "max|diff|" });
    try out.print("{s}\n", .{"-" ** 66});
    const base_ns: f64 = @floatFromInt(rows[0].ns);
    for (rows) |row| {
        const ms: f64 = @as(f64, @floatFromInt(row.ns)) / 1.0e6;
        const gflops: f64 = flops / @as(f64, @floatFromInt(row.ns));
        const speedup: f64 = base_ns / @as(f64, @floatFromInt(row.ns));
        try out.print("{s:<18} {d:>10.3} {d:>12.2} {d:>9.2}x {e:>12.1}\n", .{ row.name, ms, gflops, speedup, row.diff });
    }
    try out.print("\n", .{});
    try out.flush();
}
