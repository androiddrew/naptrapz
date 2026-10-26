//! naptrap-parity <fixtures.safetensors> [--models DIR]
//!
//! Runs each stage on the fixture's clip and keyword and compares it with the
//! tensors tools/dump_fixtures.py saved from the PyTorch reference. Stages
//! downstream of the first get the reference inputs too, so one broken stage
//! doesn't hide the others.

const std = @import("std");
const naptrap = @import("naptrap");
const bundled = @import("naptrap_bundled");

const Check = struct {
    name: []const u8,
    max_abs: f32,
    tolerance: f32,
    index: usize = 0,
    reference: f32 = 0,
    actual: f32 = 0,
    linear_floor: f64 = 0,
    roundoff_bins: usize = 0,
    passed: bool = true,
};

const Metric = enum { absolute, power_log10, magnitude_log };

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const out = &stdout_writer.interface;
    const code = run(io, arena, args, out) catch |err| blk: {
        std.log.err("{s}", .{@errorName(err)});
        break :blk @as(u8, 2);
    };
    try out.flush();
    return code;
}

fn run(io: std.Io, arena: std.mem.Allocator, args: []const [:0]const u8, out: *std.Io.Writer) !u8 {
    if (args.len < 2) {
        try out.writeAll("usage: naptrap-parity <fixtures.safetensors> [--models DIR | --embedded] [--strict-frontends]\n");
        return 2;
    }
    var models: []const u8 = "models";
    var models_explicit = false;
    var force_embedded = false;
    var strict_frontends = false;
    var arg: usize = 2;
    while (arg < args.len) : (arg += 1) {
        if (std.mem.eql(u8, args[arg], "--models") and arg + 1 < args.len) {
            arg += 1;
            models = args[arg];
            models_explicit = true;
        } else if (std.mem.eql(u8, args[arg], "--embedded")) {
            force_embedded = true;
        } else if (std.mem.eql(u8, args[arg], "--strict-frontends")) {
            strict_frontends = true;
        } else return error.InvalidArguments;
    }

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[1], arena, .limited(64 << 20));
    const fx = try naptrap.safetensors.File.parse(arena, bytes);
    // Validate every reference before using its dimensions or feeding it to a
    // kernel. A corrupt oracle must never be reported as a successful match.
    try validateFixture(fx);
    const wave = (try fx.get("wave")).data;
    const keyword = fx.metadataString("keyword") orelse return error.FixtureMissingKeyword;
    const normalized = fx.metadataString("normalized") orelse return error.FixtureMissingNormalized;
    const ours = try naptrap.text.normalize(arena, keyword);
    if (!std.mem.eql(u8, normalized, ours)) {
        std.log.err("normalized keyword: reference \"{s}\", actual \"{s}\"", .{ normalized, ours });
        return error.NormalizationMismatch;
    }

    if (models_explicit and force_embedded) return error.ConflictingModelOptions;
    const model = if (force_embedded or (bundled.available and !models_explicit)) try bundled.load(arena) else try naptrap.Model.load(
        io,
        arena,
        try std.fmt.allocPrint(arena, "{s}/phonmatchnet", .{models}),
        try std.fmt.allocPrint(arena, "{s}/google", .{models}),
        try std.fmt.allocPrint(arena, "{s}/g2p", .{models}),
    );
    var checks: std.ArrayList(Check) = .empty;

    // 1. Text features.
    const kw = try model.keyword(arena, keyword);
    try checks.append(arena, try compare(fx, "text", kw.data, &.{ kw.rows, 256 }, 1e-4, .absolute));

    // 2. Keyword log-mel.
    const mel = try model.network.keywordMel(arena, wave);
    try checks.append(arena, try compare(fx, "keyword_mel", mel, &.{ mel.len / 40, 40 }, 1e-3, if (strict_frontends) .absolute else .power_log10));

    // 3. Google log-mel, then the conv stack on the reference log-mel.
    const google_mel = try model.google.logMel(arena, wave);
    try checks.append(arena, try compare(fx, "google_log_mel", google_mel, &.{ google_mel.len / 32, 32 }, 1e-3, if (strict_frontends) .absolute else .magnitude_log));
    const ref_google_mel = try fx.get("google_log_mel");
    const emb = try model.google.encodeLogMel(arena, ref_google_mel.data, ref_google_mel.shape[0]);
    try checks.append(arena, try compare(fx, "google", emb.data, &.{ emb.rows, 96 }, 1e-3, .absolute));

    // 4. Network on the reference features, then end to end.
    const ref_mel = try fx.get("keyword_mel");
    const ref_google = try fx.get("google");
    const ref_text = try fx.get("text");
    const logit = try model.network.forward(arena, ref_mel.data, ref_mel.shape[0], ref_google.data, ref_google.shape[0], ref_text.data, ref_text.shape[0]);
    try checks.append(arena, try compare(fx, "logit", &.{logit}, &.{1}, 1e-3, .absolute));

    const far = model.thresholds[0].far;
    const p = try model.detect(arena, wave, kw, far);
    try checks.append(arena, try compare(fx, "logit", &.{p.logit}, &.{1}, 5e-3, .absolute));
    checks.items[checks.items.len - 1].name = "logit (end to end)";

    var failed = false;
    try out.print("keyword: \"{s}\" -> \"{s}\", {d} samples\n\n", .{ keyword, kw.text, wave.len });
    for (checks.items) |c| {
        const ok = c.passed;
        failed = failed or !ok;
        try out.print("{s:<20} max |diff| {e:>10.3}  tol {e:.0}", .{ c.name, c.max_abs, c.tolerance });
        if (c.linear_floor > 0) try out.print(" + linear {e:.0}", .{c.linear_floor});
        try out.print("  {s}\n", .{if (ok) "ok" else "FAIL"});
        try out.print("  worst flat index {d}: reference {e:.7}, actual {e:.7}\n", .{ c.index, c.reference, c.actual });
        if (c.roundoff_bins > 0) try out.print("  {d} bins within the linear-domain roundoff floor\n", .{c.roundoff_bins});
    }
    return if (failed) 1 else 0;
}

fn referenceTensor(fx: naptrap.safetensors.File, name: []const u8, shape: []const usize) !void {
    const tensor = try fx.get(name);
    if (!std.mem.eql(usize, tensor.shape, shape)) {
        std.log.err("{s}: reference shape {any}, expected {any}", .{ name, tensor.shape, shape });
        return error.FixtureShapeMismatch;
    }
    for (tensor.data, 0..) |value, index| {
        if (!std.math.isFinite(value)) {
            std.log.err("{s}: nonfinite reference at flat index {d}", .{ name, index });
            return error.NonfiniteReference;
        }
    }
}

fn validateFixture(fx: naptrap.safetensors.File) !void {
    const wave = try fx.get("wave");
    const n = wave.data.len;
    if (n < 8000 or n > 32000) return error.FixtureClipLengthOutOfRange;
    try referenceTensor(fx, "wave", &.{n});
    for (wave.data) |sample| if (@abs(sample) > 1) return error.FixtureInvalidSamples;
    try referenceTensor(fx, "keyword_mel", &.{ (n + 159) / 160, 40 });
    try referenceTensor(fx, "google_log_mel", &.{ naptrap.features.googleFrames(n), 32 });
    try referenceTensor(fx, "google", &.{ n / 1280, 96 });
    const text = try fx.get("text");
    if (text.shape.len != 2 or text.shape[0] < 1 or text.shape[0] > 128) return error.FixtureShapeMismatch;
    try referenceTensor(fx, "text", &.{ text.shape[0], 256 });
    try referenceTensor(fx, "logit", &.{1});
}

fn compare(fx: naptrap.safetensors.File, name: []const u8, ours: []const f32, shape: []const usize, tolerance: f32, metric: Metric) !Check {
    const ref = try fx.get(name);
    if (!std.mem.eql(usize, ref.shape, shape) or ref.data.len != ours.len) {
        std.log.err("{s}: actual shape {any}, reference has {any}", .{ name, shape, ref.shape });
        return error.ShapeMismatch;
    }
    // Near-empty spectral bins magnify fp32 FFT roundoff after logarithms.
    // Keep the original log tolerance and allow only small absolute differences
    // in linear mel energy. Verified against a float64 FFT of fp32-windowed
    // inputs; see README. Strict mode retains the original log-only criterion.
    var check = Check{ .name = name, .max_abs = 0, .tolerance = tolerance, .linear_floor = switch (metric) {
        .absolute => 0,
        .power_log10 => 5e-9,
        .magnitude_log => 1e-5,
    } };
    for (ref.data, ours, 0..) |a, b, index| {
        if (!std.math.isFinite(a) or !std.math.isFinite(b)) {
            std.log.err("{s}: nonfinite value at flat index {d}: reference {e}, actual {e}", .{ name, index, a, b });
            return error.NonfiniteComparison;
        }
        const diff = @abs(a - b);
        if (diff > tolerance) {
            const linear_diff = switch (metric) {
                .absolute => std.math.inf(f64),
                .power_log10 => @abs(std.math.pow(f64, 10, a) - std.math.pow(f64, 10, b)),
                .magnitude_log => @abs(@exp(@as(f64, a)) - @exp(@as(f64, b))),
            };
            if (std.math.isFinite(linear_diff) and linear_diff <= check.linear_floor) {
                check.roundoff_bins += 1;
            } else check.passed = false;
        }
        if (index == 0 or diff > check.max_abs) {
            check.max_abs = diff;
            check.index = index;
            check.reference = a;
            check.actual = b;
        }
    }
    return check;
}
