//! Exercise C — the perplexity plumbing.
//!
//! Perplexity measures how surprised the model is by real text: you feed it the
//! true characters and read off the probability it assigned to each *actual*
//! next character. Lower = better.
//!
//!     for each position i:
//!         logp_i = log_softmax(logits_i)[ true_next_token_i ]
//!     mean_nll   = mean(-logp_i)
//!     perplexity = exp(mean_nll)
//!
//! This is exactly the loop inside the engine's `perplexity` subcommand
//! (zig/src/main.zig::cmdPerplexity) — the part that turns dumped logits into a
//! score. Implement it here, then the provided Python harness
//! (ex_c_perplexity.py) runs the full engine on a held-out Shakespeare slice
//! for both the f32 and int8 weights and compares.
//!
//! The test FAILS until you implement `perplexity`.
//!
//! Run:  zig test ex_c_perplexity.zig
//! Solution:  ../solutions/ex_c_perplexity.zig

const std = @import("std");

/// TODO(you): given `n_pos` logit rows (each `vocab` long, row-major in
/// `logits`) and the true next-token id for each row in `targets`, return the
/// perplexity = exp(mean negative log-likelihood).
///
/// Compute log_softmax stably: subtract the row max before exponentiating.
pub fn perplexity(logits: []const f32, targets: []const usize, n_pos: usize, vocab: usize) f64 {
    _ = logits;
    _ = targets;
    _ = n_pos;
    _ = vocab;
    return 0; // replace this
}

test "perplexity matches the numpy reference" {
    // 3 positions, vocab 4 (values match python/scripts reference computation)
    const logits = [_]f32{
        2.0,  1.0, 0.1, -1.0,
        0.5,  0.5, 2.0, 0.0,
        -1.0, 3.0, 0.0, 1.0,
    };
    const targets = [_]usize{ 0, 2, 1 };
    const ppl = perplexity(&logits, &targets, 3, 4);
    try std.testing.expectApproxEqAbs(@as(f64, 1.439520), ppl, 1e-4);
}
