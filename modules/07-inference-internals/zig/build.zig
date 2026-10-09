const std = @import("std");

// Build the `tiny-gpt` inference engine. ReleaseFast is the default so a plain
//   zig build
// gives you a fast binary at zig-out/bin/tiny-gpt. Then, e.g.:
//   zig-out/bin/tiny-gpt generate --prompt "ROMEO:" --tokens 200 --temperature 0.8
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    const exe = b.addExecutable(.{
        .name = "tiny-gpt",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run tiny-gpt (pass args after --)");
    run_step.dependOn(&run_cmd.step);

    // Unit tests for the kernels (LayerNorm, softmax, GELU, quant round-trip).
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gpt.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
