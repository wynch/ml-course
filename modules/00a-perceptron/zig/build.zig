const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{ .name = "perceptron", .root_module = root });
    b.installArtifact(exe);

    // `zig build run` trains the perceptron and checks parity with Python.
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Train the perceptron and check lane parity");
    run_step.dependOn(&run_cmd.step);

    // `zig build test` runs the assertions in main.zig.
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run the Zig lane's tests");
    test_step.dependOn(&run_tests.step);
}
