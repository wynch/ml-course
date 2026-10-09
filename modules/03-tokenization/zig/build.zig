const std = @import("std");

// Minimal build: one executable, `bpe`. Build with
//   zig build -Doptimize=ReleaseFast
// then run zig-out/bin/bpe <corpus> <num_merges> <out_merges.txt>
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "bpe",
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

    const run_step = b.step("run", "Run the BPE trainer");
    run_step.dependOn(&run_cmd.step);
}
