//! Byte-level BPE trainer in Zig 0.16 — the *same* algorithm as
//! ../../python/src/bpe.py, deliberately written to produce a byte-for-byte
//! identical merge list. The Python test suite runs this binary and diffs the
//! two merge lists (see python/tests/test_cross_language.py).
//!
//! Usage:  bpe <corpus_path> <num_merges> <out_merges_path>
//! Output: <out_merges_path> holds one "a b" line per merge (token ids), and
//!         a one-line timing summary is printed to stderr.
//!
//! The tie-break rule matches Python exactly: pick the pair with the highest
//! count; on ties pick the lexicographically smallest (a, b). Because that
//! ordering is total and independent of hash-map iteration order, both
//! languages converge on the same sequence of merges.

const std = @import("std");
const Io = std.Io;

/// A pair of token ids, packed into a u64 so it can key an AutoHashMap.
/// ids stay far below 2^32 here (base 256 + a few hundred merges), so
/// `a << 32 | b` is a lossless encoding we can unpack when we need a, b back.
fn packPair(a: u32, b: u32) u64 {
    return (@as(u64, a) << 32) | @as(u64, b);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    // smp_allocator is Zig 0.16's fast general-purpose allocator (no init/deinit).
    const alloc = std.heap.smp_allocator;

    // ---- parse args -----------------------------------------------------
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 4) {
        std.debug.print("usage: bpe <corpus> <num_merges> <out_merges>\n", .{});
        return error.BadArgs;
    }
    const corpus_path = args[1];
    const num_merges = try std.fmt.parseInt(usize, args[2], 10);
    const out_path = args[3];

    const cwd = Io.Dir.cwd();

    // ---- read the corpus as raw bytes -----------------------------------
    const data = try cwd.readFileAlloc(io, corpus_path, alloc, .limited(64 * 1024 * 1024));
    defer alloc.free(data);

    // Working sequence of token ids. Start from the raw bytes promoted to u32.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(alloc);
    try ids.ensureTotalCapacity(alloc, data.len);
    for (data) |byte| ids.appendAssumeCapacity(@as(u32, byte));

    // Learned merges, as (a, b) pairs. Merge k mints token id 256 + k.
    var merges = std.ArrayList([2]u32).empty;
    defer merges.deinit(alloc);

    const t_start = Io.Timestamp.now(io, .boot);

    // ---- the training loop (mirror of bpe.py) --------------------------
    var k: usize = 0;
    while (k < num_merges) : (k += 1) {
        // 1. count every adjacent pair
        var counts = std.AutoHashMap(u64, u32).init(alloc);
        defer counts.deinit();
        var i: usize = 0;
        while (i + 1 < ids.items.len) : (i += 1) {
            const key = packPair(ids.items[i], ids.items[i + 1]);
            const gop = try counts.getOrPut(key);
            if (gop.found_existing) gop.value_ptr.* += 1 else gop.value_ptr.* = 1;
        }
        if (counts.count() == 0) break;

        // 2. pick the best pair: max count, ties -> smallest (a, b)
        var best_count: u32 = 0;
        var best_a: u32 = 0;
        var best_b: u32 = 0;
        var have_best = false;
        var it = counts.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            const count = entry.value_ptr.*;
            const a: u32 = @intCast(key >> 32);
            const b: u32 = @intCast(key & 0xffff_ffff);
            const better = !have_best or
                count > best_count or
                (count == best_count and (a < best_a or (a == best_a and b < best_b)));
            if (better) {
                have_best = true;
                best_count = count;
                best_a = a;
                best_b = b;
            }
        }

        // 3. apply the merge in place: pair -> new id, greedy, non-overlapping
        const new_id: u32 = @intCast(256 + k);
        var out = std.ArrayList(u32).empty;
        try out.ensureTotalCapacity(alloc, ids.items.len);
        var j: usize = 0;
        const n = ids.items.len;
        while (j < n) {
            if (j + 1 < n and ids.items[j] == best_a and ids.items[j + 1] == best_b) {
                out.appendAssumeCapacity(new_id);
                j += 2;
            } else {
                out.appendAssumeCapacity(ids.items[j]);
                j += 1;
            }
        }
        ids.deinit(alloc);
        ids = out;

        try merges.append(alloc, .{ best_a, best_b });
    }

    const elapsed_ns = t_start.durationTo(Io.Timestamp.now(io, .boot)).nanoseconds;

    // ---- write the merge list -------------------------------------------
    var text = std.ArrayList(u8).empty;
    defer text.deinit(alloc);
    var buf: [64]u8 = undefined;
    for (merges.items) |m| {
        const line = try std.fmt.bufPrint(&buf, "{d} {d}\n", .{ m[0], m[1] });
        try text.appendSlice(alloc, line);
    }
    try cwd.writeFile(io, .{ .sub_path = out_path, .data = text.items });

    const secs = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    std.debug.print(
        "zig: trained {d} merges on {d} bytes in {d:.3}s -> {s}\n",
        .{ merges.items.len, data.len, secs, out_path },
    );
}
