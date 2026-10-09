//! The tiny-GPT forward pass, in plain Zig — no framework, no autograd, no
//! hidden anything. Every float that comes out of `logits()` is accounted for
//! by the arithmetic in this file.
//!
//! The model is the char-level GPT trained in module 04 and exported to
//! `artifacts/tiny_gpt_weights.bin`. Its architecture (see EXPORT_FORMAT.md):
//!
//!     x = wte[token] + wpe[pos]
//!     for each block:
//!         x = x + attn(layernorm(x))          causal multi-head self-attention
//!         x = x + mlp (layernorm(x))          Linear -> GELU -> Linear
//!     x = layernorm_final(x)
//!     logits = x @ wte.weight^T               LM head is TIED to wte
//!
//! ## One primitive, two drivers
//!
//! The whole engine is built on a single function: `step(token, pos)`, which
//! pushes one token through the network and returns its logits, updating a
//! per-layer key/value cache as it goes. From that one primitive we get both
//! generation strategies for free:
//!
//!   * KV cache ON  — call `step` once per token, never resetting the cache.
//!     Each step is O(seq_len). Total generation is O(n^2). This is what real
//!     inference engines do (the "decode" loop).
//!
//!   * KV cache OFF — reset the cache and replay the whole sequence through
//!     `step` for every new token. Each step is O(seq_len^2). Total is O(n^3).
//!     This is the naive "recompute everything" baseline.
//!
//! Because the causal attention math is identical either way, both drivers —
//! and the PyTorch reference in python/ — produce the *same* logits. That is
//! exactly what python/scripts/parity.py checks (max abs diff < 1e-3).

const std = @import("std");

/// Model dimensions, parsed from `tiny_gpt_config.json`. Nothing is hardcoded;
/// the config is the single source of truth so a different checkpoint just
/// works.
pub const Config = struct {
    vocab_size: usize,
    block_size: usize, // maximum context length (128 for this checkpoint)
    n_layer: usize,
    n_head: usize,
    d_model: usize,
    d_head: usize,
    d_ff: usize,
    eps: f32,
};

/// The weights of one transformer block. Linear weights use PyTorch's
/// `[out_features, in_features]` row-major layout, so `y[o] = sum_i x[i]*W[o*in+i] + b[o]`.
pub const Layer = struct {
    ln1_w: []f32,
    ln1_b: []f32,
    qkv_w: []f32, // [3*d_model, d_model]
    qkv_b: []f32, // [3*d_model]
    proj_w: []f32, // [d_model, d_model]
    proj_b: []f32,
    ln2_w: []f32,
    ln2_b: []f32,
    fc_w: []f32, // [d_ff, d_model]
    fc_b: []f32,
    mproj_w: []f32, // [d_model, d_ff]
    mproj_b: []f32,
};

/// All model weights, already dequantized to f32 if we loaded a Q8 file.
pub const Weights = struct {
    wte: []f32, // [vocab_size, d_model] — also the (tied) LM head
    wpe: []f32, // [block_size, d_model]
    layers: []Layer,
    ln_f_w: []f32,
    ln_f_b: []f32,
};

/// Reused scratch buffers so the hot decode loop performs *zero* allocations
/// per token. Allocated once from an arena; sized from the config.
pub const Scratch = struct {
    x: []f32, // residual stream        [d_model]
    xn: []f32, // layernorm output       [d_model]
    qkv: []f32, // q|k|v packed           [3*d_model]
    y: []f32, // attention output       [d_model]
    ff: []f32, // mlp hidden             [d_ff]
    scores: []f32, // attention scores       [block_size]
    logits: []f32, // output logits          [vocab_size]
};

pub const Model = struct {
    cfg: Config,
    w: Weights,
    s: Scratch,
    // KV cache: kcache[layer] holds K for every position, laid out as
    // [block_size * d_model]; head h of position p is the slice
    // [p*d_model + h*d_head .. + d_head]. vcache is the same for V.
    kcache: [][]f32,
    vcache: [][]f32,

    pub fn init(arena: std.mem.Allocator, cfg: Config, w: Weights) !Model {
        const dm = cfg.d_model;
        const s = Scratch{
            .x = try arena.alloc(f32, dm),
            .xn = try arena.alloc(f32, dm),
            .qkv = try arena.alloc(f32, 3 * dm),
            .y = try arena.alloc(f32, dm),
            .ff = try arena.alloc(f32, cfg.d_ff),
            .scores = try arena.alloc(f32, cfg.block_size),
            .logits = try arena.alloc(f32, cfg.vocab_size),
        };
        const kcache = try arena.alloc([]f32, cfg.n_layer);
        const vcache = try arena.alloc([]f32, cfg.n_layer);
        for (0..cfg.n_layer) |l| {
            kcache[l] = try arena.alloc(f32, cfg.block_size * dm);
            vcache[l] = try arena.alloc(f32, cfg.block_size * dm);
        }
        return .{ .cfg = cfg, .w = w, .s = s, .kcache = kcache, .vcache = vcache };
    }

    /// Forget every cached key/value. Call before starting a fresh sequence,
    /// or before every step when running the KV-cache-OFF baseline.
    pub fn resetCache(self: *Model) void {
        // Nothing to zero: positions are written before they are read, and we
        // only ever read cache slots [0 .. pos]. Reset is a no-op on the data;
        // the *driver* controls how many positions are considered live by the
        // `pos` it passes to `step`. We keep the method for intent/clarity.
        _ = self;
    }

    /// Push a single token through the network at absolute position `pos`,
    /// writing this token's K/V into the cache and returning its logits.
    /// `pos` must be < block_size. Returns a slice into scratch (valid until
    /// the next call).
    pub fn step(self: *Model, token: usize, pos: usize) []f32 {
        const cfg = self.cfg;
        const dm = cfg.d_model;
        const dh = cfg.d_head;
        const s = self.s;

        // --- embedding: token + position -------------------------------------
        const tok_row = self.w.wte[token * dm ..][0..dm];
        const pos_row = self.w.wpe[pos * dm ..][0..dm];
        for (0..dm) |i| s.x[i] = tok_row[i] + pos_row[i];

        // --- transformer blocks ---------------------------------------------
        for (self.w.layers, 0..) |layer, l| {
            // pre-norm attention
            layerNorm(s.xn, s.x, layer.ln1_w, layer.ln1_b, cfg.eps);
            self.attention(layer, l, pos);
            // residual: x = x + proj(attn)  (proj applied inside attention -> s.y)
            for (0..dm) |i| s.x[i] += s.y[i];

            // pre-norm mlp
            layerNorm(s.xn, s.x, layer.ln2_w, layer.ln2_b, cfg.eps);
            // fc: [d_ff, d_model] -> GELU -> proj: [d_model, d_ff]
            linear(s.ff, s.xn, layer.fc_w, layer.fc_b, cfg.d_ff, dm);
            for (s.ff) |*v| v.* = gelu(v.*);
            linear(s.y, s.ff, layer.mproj_w, layer.mproj_b, dm, cfg.d_ff);
            for (0..dm) |i| s.x[i] += s.y[i];
        }

        // --- final layernorm + tied LM head ---------------------------------
        layerNorm(s.xn, s.x, self.w.ln_f_w, self.w.ln_f_b, cfg.eps);
        // logits[v] = xn . wte[v]      (tied head: reuse the token embedding)
        for (0..cfg.vocab_size) |v| {
            const row = self.w.wte[v * dm ..][0..dm];
            var acc: f32 = 0;
            for (0..dm) |i| acc += s.xn[i] * row[i];
            s.logits[v] = acc;
        }
        _ = dh;
        return s.logits;
    }

    /// Causal multi-head self-attention for the token currently in `s.xn`,
    /// at absolute position `pos`, using and updating layer `l`'s KV cache.
    /// Result (after the output projection) is left in `s.y`.
    fn attention(self: *Model, layer: Layer, l: usize, pos: usize) void {
        const cfg = self.cfg;
        const dm = cfg.d_model;
        const dh = cfg.d_head;
        const s = self.s;

        // project x_norm into packed [Q | K | V], each of width d_model
        linear(s.qkv, s.xn, layer.qkv_w, layer.qkv_b, 3 * dm, dm);
        const q = s.qkv[0..dm];
        const k = s.qkv[dm .. 2 * dm];
        const vv = s.qkv[2 * dm .. 3 * dm];

        // store this token's K and V into the cache at position `pos`
        const kdst = self.kcache[l][pos * dm ..][0..dm];
        const vdst = self.vcache[l][pos * dm ..][0..dm];
        @memcpy(kdst, k);
        @memcpy(vdst, vv);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dh)));

        // each head attends independently over positions 0..=pos
        for (0..cfg.n_head) |h| {
            const off = h * dh;
            const qh = q[off..][0..dh];

            // scores[j] = (q_h . k_h[j]) / sqrt(d_head), for j in 0..=pos
            for (0..pos + 1) |j| {
                const kh = self.kcache[l][j * dm + off ..][0..dh];
                var dot: f32 = 0;
                for (0..dh) |d| dot += qh[d] * kh[d];
                s.scores[j] = dot * scale;
            }
            softmax(s.scores[0 .. pos + 1]);

            // head output = sum_j weights[j] * v_h[j]
            const yh = s.y[off..][0..dh];
            for (0..dh) |d| yh[d] = 0;
            for (0..pos + 1) |j| {
                const wj = s.scores[j];
                const vh = self.vcache[l][j * dm + off ..][0..dh];
                for (0..dh) |d| yh[d] += wj * vh[d];
            }
        }

        // output projection: reuse s.xn as scratch, then leave result in s.y
        // (proj maps d_model -> d_model; we need s.y as both input and output,
        //  so route through s.xn which is free at this point)
        @memcpy(s.xn[0..dm], s.y[0..dm]);
        linear(s.y, s.xn, layer.proj_w, layer.proj_b, dm, dm);
    }
};

// ------------------------------------------------------------------ kernels --

/// y = x @ W^T + b, where W is [out_dim, in_dim] row-major (PyTorch layout).
pub fn linear(y: []f32, x: []const f32, w: []const f32, b: []const f32, out_dim: usize, in_dim: usize) void {
    for (0..out_dim) |o| {
        const row = w[o * in_dim ..][0..in_dim];
        var acc: f32 = b[o];
        for (0..in_dim) |i| acc += x[i] * row[i];
        y[o] = acc;
    }
}

/// LayerNorm over the feature axis: y = (x-mean)/sqrt(var+eps) * gamma + beta.
/// Variance is the biased estimate (divide by N), matching torch.nn.LayerNorm.
pub fn layerNorm(y: []f32, x: []const f32, gamma: []const f32, beta: []const f32, eps: f32) void {
    const n = x.len;
    var mean: f32 = 0;
    for (x) |v| mean += v;
    mean /= @floatFromInt(n);
    var variance: f32 = 0;
    for (x) |v| {
        const d = v - mean;
        variance += d * d;
    }
    variance /= @floatFromInt(n);
    const inv = 1.0 / @sqrt(variance + eps);
    for (0..n) |i| y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
}

/// In-place softmax with the standard max-subtraction for numerical stability.
pub fn softmax(x: []f32) void {
    var m: f32 = -std.math.inf(f32);
    for (x) |v| m = @max(m, v);
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - m);
        sum += v.*;
    }
    const inv = 1.0 / sum;
    for (x) |*v| v.* *= inv;
}

/// Exact (erf-form) GELU, matching PyTorch's default `F.gelu`:
///     gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
/// We compute erf in f64 for accuracy, then round back to f32.
pub fn gelu(v: f32) f32 {
    const x: f64 = v;
    const inv_sqrt2: f64 = 0.7071067811865476;
    return @floatCast(0.5 * x * (1.0 + erf(x * inv_sqrt2)));
}

/// erf via Abramowitz & Stegun 7.1.26 (|error| <= 1.5e-7) — accurate enough
/// that the resulting logits match PyTorch to well under the 1e-3 tolerance.
fn erf(x: f64) f64 {
    const a1: f64 = 0.254829592;
    const a2: f64 = -0.284496736;
    const a3: f64 = 1.421413741;
    const a4: f64 = -1.453152027;
    const a5: f64 = 1.061405429;
    const p: f64 = 0.3275911;
    const sign: f64 = if (x < 0) -1.0 else 1.0;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + p * ax);
    const poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    const y = 1.0 - poly * @exp(-ax * ax);
    return sign * y;
}

// ------------------------------------------------------------------- tests --

const testing = std.testing;

test "layerNorm matches PyTorch reference (eps=1e-5)" {
    const x = [_]f32{ 3.381100, -0.931900, 0.065600, 0.815000, -1.577800, 0.004100, -0.001800, -3.509400 };
    const g = [_]f32{ 1.089600, 1.151900, 0.940500, 0.783000, 0.894100, 1.204800, 0.856700, 0.976100 };
    const b = [_]f32{ 0.172500, -0.190000, 0.040200, 0.180100, -0.107900, 0.019400, 0.163700, -0.146700 };
    const want = [_]f32{ 2.292734, -0.633570, 0.185057, 0.617824, -0.764307, 0.164920, 0.264444, -1.882290 };
    var y: [8]f32 = undefined;
    layerNorm(&y, &x, &g, &b, 1e-5);
    for (y, want) |got, w| try testing.expectApproxEqAbs(w, got, 1e-4);
}

test "softmax sums to one and is monotone in the input" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 0.0 };
    softmax(&x);
    var sum: f32 = 0;
    for (x) |v| sum += v;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
    try testing.expect(x[2] > x[1] and x[1] > x[0]);
}

test "gelu matches known values" {
    // exact-erf GELU: gelu(0)=0, gelu(1)~=0.8413, gelu(-1)~=-0.1587
    try testing.expectApproxEqAbs(@as(f32, 0.0), gelu(0.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.8413447), gelu(1.0), 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -0.1586553), gelu(-1.0), 1e-4);
}

test "erf approximation is accurate" {
    try testing.expectApproxEqAbs(@as(f64, 0.5204998778), erf(0.5), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.8427007929), erf(1.0), 1e-6);
}
