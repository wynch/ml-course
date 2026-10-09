const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // Default to ReleaseFast: a matmul benchmark in Debug measures the debug
    // checks, not the code. Override with `-Doptimize=Debug` if you want.
    const optimize = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Optimization mode (defaults to ReleaseFast for meaningful timings)",
    ) orelse .ReleaseFast;

    const exe = b.addExecutable(.{
        .name = "matmul_bench",
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

    const run_step = b.step("run", "Run the matmul benchmark");
    run_step.dependOn(&run_cmd.step);
}
