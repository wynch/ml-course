//! Loading the model off disk: parse `tiny_gpt_config.json`, then pull each
//! named tensor out of the flat `tiny_gpt_weights.bin` blob (or its int8 Q8
//! variant) into f32 arrays laid out exactly how `gpt.zig` wants them.
//!
//! There is no custom binary parser here for the f32 file — the config's tensor
//! manifest (name, byte offset, element count) is the index, and Apple silicon
//! is little-endian, so a tensor is just a `@memcpy` of `count*4` bytes.

const std = @import("std");
const gpt = @import("gpt.zig");
const quant = @import("quant.zig");

/// One entry of the config's `"tensors"` manifest.
pub const TensorInfo = struct {
    name: []const u8,
    shape: []i64,
    offset: usize,
    count: usize,
};

/// The parts of `tiny_gpt_config.json` we care about. `ignore_unknown_fields`
/// lets us skip the descriptive strings (arch, dtype, ...).
pub const RawConfig = struct {
    vocab_size: usize,
    block_size: usize,
    n_layer: usize,
    n_head: usize,
    d_model: usize,
    d_head: usize,
    d_ff: usize,
    layer_norm_eps: f64,
    tensors: []TensorInfo,
};

pub const Loaded = struct {
    cfg: gpt.Config,
    w: gpt.Weights,
};

/// Parse the config JSON into a `gpt.Config` plus the raw manifest (kept alive
/// on the arena for the caller to index tensors by name).
pub fn parseConfig(arena: std.mem.Allocator, json_bytes: []const u8) !std.json.Parsed(RawConfig) {
    return std.json.parseFromSlice(RawConfig, arena, json_bytes, .{ .ignore_unknown_fields = true });
}

pub fn toConfig(raw: RawConfig) gpt.Config {
    return .{
        .vocab_size = raw.vocab_size,
        .block_size = raw.block_size,
        .n_layer = raw.n_layer,
        .n_head = raw.n_head,
        .d_model = raw.d_model,
        .d_head = raw.d_head,
        .d_ff = raw.d_ff,
        .eps = @floatCast(raw.layer_norm_eps),
    };
}

fn find(manifest: []TensorInfo, name: []const u8) !TensorInfo {
    for (manifest) |t| {
        if (std.mem.eql(u8, t.name, name)) return t;
    }
    std.debug.print("tensor not found in manifest: {s}\n", .{name});
    return error.TensorNotFound;
}

/// Copy one tensor's `count` little-endian f32 values out of the raw blob.
fn tensor(alloc: std.mem.Allocator, blob: []const u8, info: TensorInfo) ![]f32 {
    const out = try alloc.alloc(f32, info.count);
    const bytes = blob[info.offset .. info.offset + info.count * 4];
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

/// Load all 40 tensors from the raw f32 blob into a `gpt.Weights`.
pub fn loadWeightsF32(alloc: std.mem.Allocator, cfg: gpt.Config, manifest: []TensorInfo, blob: []const u8) !gpt.Weights {
    var buf: [64]u8 = undefined;
    const layers = try alloc.alloc(gpt.Layer, cfg.n_layer);
    for (0..cfg.n_layer) |i| {
        const p = struct {
            fn name(b: []u8, layer: usize, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(b, "blocks.{d}.{s}", .{ layer, suffix }) catch unreachable;
            }
        };
        layers[i] = .{
            .ln1_w = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "ln1.weight"))),
            .ln1_b = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "ln1.bias"))),
            .qkv_w = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "attn.qkv.weight"))),
            .qkv_b = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "attn.qkv.bias"))),
            .proj_w = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "attn.proj.weight"))),
            .proj_b = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "attn.proj.bias"))),
            .ln2_w = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "ln2.weight"))),
            .ln2_b = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "ln2.bias"))),
            .fc_w = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "mlp.fc.weight"))),
            .fc_b = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "mlp.fc.bias"))),
            .mproj_w = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "mlp.proj.weight"))),
            .mproj_b = try tensor(alloc, blob, try find(manifest, p.name(&buf, i, "mlp.proj.bias"))),
        };
    }
    return .{
        .wte = try tensor(alloc, blob, try find(manifest, "wte.weight")),
        .wpe = try tensor(alloc, blob, try find(manifest, "wpe.weight")),
        .layers = layers,
        .ln_f_w = try tensor(alloc, blob, try find(manifest, "ln_f.weight")),
        .ln_f_b = try tensor(alloc, blob, try find(manifest, "ln_f.bias")),
    };
}

// ------------------------------------------------------------- tokenizer -----

/// The char vocabulary from `tokenizer_chars.json`. Token id == index.
pub const Tokenizer = struct {
    chars: [][]const u8, // each is 1 UTF-8 char (e.g. "\n", " ", "A")

    pub fn encode(self: Tokenizer, alloc: std.mem.Allocator, text: []const u8) ![]usize {
        var ids = std.ArrayList(usize).empty;
        // char-level: match each input byte to a single-byte vocab entry.
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            const c = text[i];
            var found: ?usize = null;
            for (self.chars, 0..) |ch, id| {
                if (ch.len == 1 and ch[0] == c) {
                    found = id;
                    break;
                }
            }
            if (found) |id| {
                try ids.append(alloc, id);
            } else {
                // unknown char: skip it (keeps the demo robust to odd prompts)
            }
        }
        return ids.toOwnedSlice(alloc);
    }

    pub fn decodeInto(self: Tokenizer, list: *std.ArrayList(u8), alloc: std.mem.Allocator, id: usize) !void {
        try list.appendSlice(alloc, self.chars[id]);
    }
};

pub fn parseTokenizer(arena: std.mem.Allocator, json_bytes: []const u8) !Tokenizer {
    const Wrap = struct { chars: [][]const u8 };
    const parsed = try std.json.parseFromSlice(Wrap, arena, json_bytes, .{});
    return .{ .chars = parsed.value.chars };
}
