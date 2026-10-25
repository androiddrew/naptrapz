//! naptrap-parity <fixtures.safetensors> [--models DIR]
//!
//! Runs each stage on the fixture's clip and keyword and compares it with the
//! tensors tools/dump_fixtures.py saved from the PyTorch reference. Stages
//! downstream of the first get the reference inputs too, so one broken stage
//! doesn't hide the others.

const std = @import("std");
const naptrap = @import("naptrap");

const Check = struct {
    name: []const u8,
    max_abs: f32,
    tolerance: f32,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const out = &stdout_writer.interface;
    defer out.flush() catch {};

    if (args.len != 2 and !(args.len == 4 and std.mem.eql(u8, args[2], "--models"))) {
        try out.writeAll("usage: naptrap-parity <fixtures.safetensors> [--models DIR]\n");
        std.process.exit(2);
    }
    const models: []const u8 = if (args.len == 4) args[3] else "models";

    const model = try naptrap.Model.load(
        io,
        arena,
        try std.fmt.allocPrint(arena, "{s}/phonmatchnet", .{models}),
        try std.fmt.allocPrint(arena, "{s}/google", .{models}),
        try std.fmt.allocPrint(arena, "{s}/g2p", .{models}),
    );
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[1], arena, .limited(64 << 20));
    const fx = try naptrap.safetensors.File.parse(arena, bytes);

    const wave = (try fx.get("wave")).data;
    const keyword = fx.metadataString("keyword") orelse return error.FixtureMissingKeyword;
    var checks: std.ArrayList(Check) = .empty;

    // 1. Text features.
    const kw = try model.keyword(arena, keyword);
    try checks.append(arena, try compare(fx, "text", kw.data, 1e-4));

    // 2. Keyword log-mel.
    const mel = try model.network.keywordMel(arena, model.dft, wave);
    try checks.append(arena, try compare(fx, "keyword_mel", mel, 1e-3));

    // 3. Google log-mel, then the conv stack on the reference log-mel.
    const google_mel = try model.google.logMel(arena, model.dft, wave);
    try checks.append(arena, try compare(fx, "google_log_mel", google_mel, 1e-3));
    const ref_google_mel = try fx.get("google_log_mel");
    const emb = try model.google.encodeLogMel(arena, ref_google_mel.data, ref_google_mel.shape[0]);
    try checks.append(arena, try compare(fx, "google", emb.data, 1e-3));

    // 4. Network on the reference features, then end to end.
    const ref_mel = try fx.get("keyword_mel");
    const ref_google = try fx.get("google");
    const ref_text = try fx.get("text");
    const logit = try model.network.forward(arena, ref_mel.data, ref_mel.shape[0], ref_google.data, ref_google.shape[0], ref_text.data, ref_text.shape[0]);
    try checks.append(arena, try compare(fx, "logit", &.{logit}, 1e-3));

    const far = model.thresholds[0].far;
    const p = try model.detect(arena, wave, kw, far);
    try checks.append(arena, try compare(fx, "logit", &.{p.logit}, 5e-3));
    checks.items[checks.items.len - 1].name = "logit (end to end)";

    var failed = false;
    try out.print("keyword: \"{s}\" -> \"{s}\", {d} samples\n\n", .{ keyword, kw.text, wave.len });
    for (checks.items) |c| {
        const ok = c.max_abs <= c.tolerance;
        failed = failed or !ok;
        try out.print("{s:<20} max |diff| {e:>10.3}  tol {e:.0}  {s}\n", .{ c.name, c.max_abs, c.tolerance, if (ok) "ok" else "FAIL" });
    }
    out.flush() catch {};
    std.process.exit(if (failed) 1 else 0);
}

fn compare(fx: naptrap.safetensors.File, name: []const u8, ours: []const f32, tolerance: f32) !Check {
    const ref = (try fx.get(name)).data;
    if (ref.len != ours.len) {
        std.log.err("{s}: {d} values, reference has {d}", .{ name, ours.len, ref.len });
        return .{ .name = name, .max_abs = std.math.inf(f32), .tolerance = tolerance };
    }
    var max_abs: f32 = 0;
    for (ref, ours) |a, b| max_abs = @max(max_abs, @abs(a - b));
    return .{ .name = name, .max_abs = max_abs, .tolerance = tolerance };
}
