const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The inference library. Pure computation: no file or network I/O, except
    // model.zig's loader, which takes an std.Io from the caller.
    const lib = b.addModule("naptrap", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // CLI: naptrap detect clip.wav "hey computer"
    const exe = b.addExecutable(.{
        .name = "naptrap",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "naptrap", .module = lib }},
        }),
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run the naptrap CLI").dependOn(&run.step);

    // Parity check against tensors dumped from the PyTorch reference
    // (tools/dump_fixtures.py).
    const parity = b.addExecutable(.{
        .name = "naptrap-parity",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/parity.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "naptrap", .module = lib }},
        }),
    });
    b.installArtifact(parity);

    const run_parity = b.addRunArtifact(parity);
    if (b.args) |args| run_parity.addArgs(args);
    b.step("parity", "Compare every stage against PyTorch fixtures").dependOn(&run_parity.step);

    // Unit tests (no model files needed).
    const tests = b.addTest(.{ .root_module = lib });
    b.step("test", "Run unit tests").dependOn(&b.addRunArtifact(tests).step);
}
