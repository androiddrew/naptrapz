const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The inference library. Pure computation: no file or network I/O, except
    // model.zig's loader, which takes an std.Io from the caller. Exported
    // modules leave `optimize` unset so they follow the importing module.
    const lib = b.addModule("naptrap", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    var bundle_source = b.path("src/unbundled.zig");
    if (b.option([]const u8, "embed-models", "Embed phonmatchnet/, google/, g2p/ from this directory (relative to source root or absolute)")) |root| {
        const files = b.addWriteFiles();
        inline for (.{ "phonmatchnet", "google", "g2p" }) |part| {
            inline for (.{ "config.json", "model.safetensors" }) |name| {
                const path = b.pathResolve(&.{ b.build_root.path orelse ".", root, part, name });
                _ = files.addCopyFile(.{ .cwd_relative = path }, part ++ "/" ++ name);
            }
        }
        // All generated sources/assets live in the build cache, not src/.
        bundle_source = files.add("bundled.zig",
            \\const std = @import("std");
            \\const naptrap = @import("naptrap");
            \\pub const available = true;
            \\/// Load the embedded models; the result owns memory from `gpa`.
            \\pub fn load(gpa: std.mem.Allocator) !naptrap.Model {
            \\    return naptrap.Model.fromBytes(gpa,
            \\        .{ .config = @embedFile("phonmatchnet/config.json"), .weights = @embedFile("phonmatchnet/model.safetensors") },
            \\        .{ .config = @embedFile("google/config.json"), .weights = @embedFile("google/model.safetensors") },
            \\        .{ .config = @embedFile("g2p/config.json"), .weights = @embedFile("g2p/model.safetensors") });
            \\}
            \\
        );
    }
    const bundled = b.addModule("naptrap_bundled", .{
        .root_source_file = bundle_source,
        .target = target,
        .imports = &.{.{ .name = "naptrap", .module = lib }},
    });

    // CLI: naptrap detect clip.wav "hey computer"
    const exe = b.addExecutable(.{
        .name = "naptrap",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{ .{ .name = "naptrap", .module = lib }, .{ .name = "naptrap_bundled", .module = bundled } },
        }),
    });
    b.installArtifact(exe);

    // Native ABI for C/C++, ctypes, Swift, Rust, etc. Both variants expose the
    // same versioned header; no Zig allocator or error union crosses the ABI.
    const c_module = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
        .pic = true,
        .link_libc = true, // allocations use the host's malloc
        .imports = &.{ .{ .name = "naptrap", .module = lib }, .{ .name = "naptrap_bundled", .module = bundled } },
    });
    // On Windows the DLL's import library is naptrap.lib, so the static
    // archive needs a different name. wasm has no stable shared libraries.
    const static_name = if (target.result.os.tag == .windows) "naptrap_static" else "naptrap";
    const static = b.addLibrary(.{ .name = static_name, .linkage = .static, .root_module = c_module });
    static.bundle_compiler_rt = true;
    // Zig's self-hosted Debug backend omits .note.GNU-stack, which makes linkers
    // give the host program an executable stack. LLVM always emits it.
    static.use_llvm = true;
    // Don't export Zig's UBSan runtime symbols into C hosts.
    static.bundle_ubsan_rt = false;
    static.installHeader(b.path("include/naptrap.h"), "naptrap.h");
    b.installArtifact(static);
    if (!target.result.cpu.arch.isWasm()) {
        // The soname's major version tracks NAPTRAP_ABI_VERSION.
        const shared = b.addLibrary(.{
            .name = "naptrap",
            .linkage = .dynamic,
            .root_module = c_module,
            .version = .{ .major = 1, .minor = 0, .patch = 0 },
        });
        shared.bundle_ubsan_rt = false; // keep the export table to naptrap_*
        b.installArtifact(shared);
    }

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
            .imports = &.{ .{ .name = "naptrap", .module = lib }, .{ .name = "naptrap_bundled", .module = bundled } },
        }),
    });
    if (b.option(bool, "install-parity", "Also install the naptrap-parity developer tool") orelse false)
        b.installArtifact(parity);

    const run_parity = b.addRunArtifact(parity);
    if (b.args) |args| run_parity.addArgs(args);
    b.step("parity", "Compare every stage against PyTorch fixtures").dependOn(&run_parity.step);

    // `zig build test` runs the unit tests and a C ABI smoke test. With
    // -Dtest-models=DIR it also runs end-to-end tests against real models.
    const test_step = b.step("test", "Run unit, model and C ABI tests");
    const check_step = b.step("check", "Compile every artifact and test without installing or running");
    check_step.dependOn(&exe.step);
    check_step.dependOn(&parity.step);
    check_step.dependOn(&static.step);
    const models_dir: ?[]const u8 = if (b.option([]const u8, "test-models", "Directory with phonmatchnet/, google/ and g2p/ for end-to-end tests")) |dir|
        b.pathResolve(&.{ b.build_root.path orelse ".", dir })
    else
        null;

    const unit_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    const test_options = b.addOptions();
    test_options.addOption([]const u8, "models_dir", models_dir orelse "");
    const model_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/model_test.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{ .{ .name = "naptrap", .module = lib }, .{ .name = "test_options", .module = test_options.createModule() } },
    }) });
    const abi_smoke = b.addExecutable(.{
        .name = "abi_smoke",
        .root_module = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true }),
    });
    abi_smoke.root_module.addCSourceFile(.{ .file = b.path("tests/abi_smoke.c"), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" } });
    abi_smoke.root_module.addIncludePath(b.path("include"));
    abi_smoke.root_module.linkLibrary(static);
    const run_abi_smoke = b.addRunArtifact(abi_smoke);
    if (models_dir) |dir| run_abi_smoke.addArg(dir);
    run_abi_smoke.expectExitCode(0);

    const docs = b.addInstallDirectory(.{
        .source_dir = unit_tests.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    b.step("docs", "Generate API documentation into <prefix>/docs").dependOn(&docs.step);

    for ([_]*std.Build.Step.Compile{ unit_tests, model_tests, abi_smoke }) |artifact| {
        check_step.dependOn(&artifact.step);
    }
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);
    test_step.dependOn(&b.addRunArtifact(model_tests).step);
    test_step.dependOn(&run_abi_smoke.step);
}
