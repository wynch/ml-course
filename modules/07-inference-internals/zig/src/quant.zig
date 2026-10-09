//! int8 symmetric quantization — the same idea GGUF/llama.cpp call `Q8_0`.
//!
//! For each weight *row* we keep one f32 scale and store the row as signed
//! 8-bit integers:
//!
//!     scale = max(|row|) / 127
//!     q[i]  = round(row[i] / scale)          in [-127, 127]
//!     row[i] ≈ q[i] * scale                   (dequantization)
//!
//! "Per row" means: for a 2-D weight `[out, in]` each of the `out` output rows
//! gets its own scale (this is what gives Q8 its good fidelity); for a 1-D
//! tensor the whole vector shares one scale. Storage drops from 4 bytes/weight
//! to ~1 byte/weight — the 2.48 MB f32 blob becomes ~0.62 MB.
//!
//! Our on-disk Q8 format is deliberately trivial (this is a teaching artifact,
//! not GGUF): a 4-byte magic `TGQ8`, then, for every tensor *in manifest
//! order*, `nrows` little-endian f32 scales followed by `count` int8 values.
//! The loader knows each tensor's shape from the same config manifest, so no
//! per-tensor header is needed.

const std = @import("std");
const load = @import("load.zig");

pub const MAGIC = "TGQ8";

fn rowsAndLen(info: load.TensorInfo) struct { nrows: usize, rlen: usize } {
    if (info.shape.len == 2) {
        const nrows: usize = @intCast(info.shape[0]);
        return .{ .nrows = nrows, .rlen = info.count / nrows };
    }
    return .{ .nrows = 1, .rlen = info.count };
}

/// Quantize the whole f32 blob to our Q8 byte format. Caller owns the result.
pub fn quantizeBlob(alloc: std.mem.Allocator, manifest: []load.TensorInfo, blob: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    try out.appendSlice(alloc, MAGIC);

    for (manifest) |info| {
        const rl = rowsAndLen(info);
        const src = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, blob[info.offset .. info.offset + info.count * 4]));
        for (0..rl.nrows) |r| {
            const row = src[r * rl.rlen ..][0..rl.rlen];
            var amax: f32 = 0;
            for (row) |v| amax = @max(amax, @abs(v));
            const scale: f32 = if (amax == 0) 1.0 else amax / 127.0;
            // one f32 scale, then this row's int8 values
            const sbytes: [4]u8 = @bitCast(scale);
            try out.appendSlice(alloc, &sbytes);
            for (row) |v| {
                const q = std.math.clamp(@round(v / scale), -127.0, 127.0);
                const qi: i8 = @intFromFloat(q);
                try out.append(alloc, @bitCast(qi));
            }
        }
    }
    return out.toOwnedSlice(alloc);
}

/// Reconstruct a full f32 blob (identical byte layout to the original, using
/// the manifest offsets) from Q8 bytes. `gpt` then loads it exactly as if it
/// were the f32 file — the compute path is unchanged, only the weights carry
/// quantization error.
pub fn dequantizeToBlob(alloc: std.mem.Allocator, manifest: []load.TensorInfo, q8: []const u8) ![]u8 {
    if (!std.mem.eql(u8, q8[0..4], MAGIC)) return error.BadMagic;

    // total size = highest (offset + count*4) across tensors
    var total: usize = 0;
    for (manifest) |info| total = @max(total, info.offset + info.count * 4);
    const blob = try alloc.alloc(u8, total);

    var cur: usize = 4; // past the magic
    for (manifest) |info| {
        const rl = rowsAndLen(info);
        const dst = std.mem.bytesAsSlice(f32, @as([]align(1) u8, blob[info.offset .. info.offset + info.count * 4]));
        for (0..rl.nrows) |r| {
            const scale: f32 = @bitCast(q8[cur..][0..4].*);
            cur += 4;
            const row = dst[r * rl.rlen ..][0..rl.rlen];
            for (0..rl.rlen) |i| {
                const qi: i8 = @bitCast(q8[cur]);
                cur += 1;
                row[i] = @as(f32, @floatFromInt(qi)) * scale;
            }
        }
    }
    return blob;
}
