//! naptrap detect <clip.wav> <keyword> [--far 0.01] [--models DIR]
//!                [--model-dir DIR] [--google-dir DIR] [--g2p-dir DIR]
//!
//! Prints the prediction as JSON. Exit code: 0 accepted, 1 rejected, 2 error
//! (the same contract as the Python CLI).

const std = @import("std");
const naptrap = @import("naptrap");

const usage =
    \\usage: naptrap detect <clip.wav> <keyword> [--far 0.01] [--models DIR]
    \\                      [--model-dir DIR] [--google-dir DIR] [--g2p-dir DIR]
    \\
    \\--models DIR holds phonmatchnet/, google/ and g2p/ (default: models).
    \\
;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const code: u8 = run(io, arena, args, stdout) catch |err| blk: {
        std.log.err("{s}", .{@errorName(err)});
        break :blk 2;
    };
    stdout.flush() catch {};
    std.process.exit(code);
}

fn run(io: std.Io, arena: std.mem.Allocator, args: []const [:0]const u8, out: *std.Io.Writer) !u8 {
    var positional: [3][]const u8 = undefined;
    var count: usize = 0;
    var far: f64 = 0.01;
    var models: []const u8 = "models";
    var model_dir: ?[]const u8 = null;
    var google_dir: ?[]const u8 = null;
    var g2p_dir: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg: []const u8 = args[i];
        if (std.mem.startsWith(u8, arg, "--")) {
            if (i + 1 >= args.len) return usageError(out);
            i += 1;
            const value: []const u8 = args[i];
            if (std.mem.eql(u8, arg, "--far")) {
                far = std.fmt.parseFloat(f64, value) catch return usageError(out);
            } else if (std.mem.eql(u8, arg, "--models")) {
                models = value;
            } else if (std.mem.eql(u8, arg, "--model-dir")) {
                model_dir = value;
            } else if (std.mem.eql(u8, arg, "--google-dir")) {
                google_dir = value;
            } else if (std.mem.eql(u8, arg, "--g2p-dir")) {
                g2p_dir = value;
            } else return usageError(out);
        } else {
            if (count == positional.len) return usageError(out);
            positional[count] = arg;
            count += 1;
        }
    }
    if (count != 3 or !std.mem.eql(u8, positional[0], "detect")) return usageError(out);

    const model = try naptrap.Model.load(
        io,
        arena,
        model_dir orelse try std.fmt.allocPrint(arena, "{s}/phonmatchnet", .{models}),
        google_dir orelse try std.fmt.allocPrint(arena, "{s}/google", .{models}),
        g2p_dir orelse try std.fmt.allocPrint(arena, "{s}/g2p", .{models}),
    );

    const wav_bytes = try std.Io.Dir.cwd().readFileAlloc(io, positional[1], arena, .limited(16 << 20));
    const wave = try naptrap.wav.decode(arena, wav_bytes);
    const kw = try model.keyword(arena, positional[2]);
    const p = try model.detect(arena, wave, kw, far);

    try out.print(
        "{{\"model\":\"phonmatchnet\",\"task\":\"C-KWS\",\"keyword\":\"{s}\",\"score\":{d},\"threshold\":{d}," ++
            "\"accepted\":{s},\"target_far\":{d},\"details\":{{\"keyword_logit\":{d}}}}}\n",
        .{ kw.text, p.score, p.threshold, if (p.accepted) "true" else "false", p.target_far, p.logit },
    );
    return if (p.accepted) 0 else 1;
}

fn usageError(out: *std.Io.Writer) u8 {
    out.writeAll(usage) catch {};
    return 2;
}
