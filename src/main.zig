//! naptrap detect <clip.wav> <keyword> [--far 0.01] [--models DIR]
//!                [--model-dir DIR] [--google-dir DIR] [--g2p-dir DIR]
//!
//! Prints the prediction as JSON. Exit code: 0 accepted, 1 rejected, 2 error
//! (the same contract as the Python CLI).

const std = @import("std");
const naptrap = @import("naptrap");
const bundled = @import("naptrap_bundled");

const usage =
    \\usage: naptrap detect <clip.wav> <keyword> [--far 0.01] [--models DIR]
    \\                      [--model-dir DIR] [--google-dir DIR] [--g2p-dir DIR]
    \\       naptrap listen <keyword> [--models DIR] [--far 0.01]
    \\                      [--window-ms 1500] [--hop-ms 250] [--trigger-windows 2]
    \\                      [--release-windows 2] [--cooldown-ms 1000] [--all-windows]
    \\
    \\--models DIR holds phonmatchnet/, google/ and g2p/ (default: models).
    \\listen reads raw mono 16 kHz signed 16-bit little-endian PCM from stdin.
    \\Bundled builds use embedded models by default. Explicit model paths override them.
    \\--embedded requires embedded models and cannot be combined with model paths.
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
    var models_explicit = false;
    var force_embedded = false;
    var model_dir: ?[]const u8 = null;
    var google_dir: ?[]const u8 = null;
    var g2p_dir: ?[]const u8 = null;
    var config: naptrap.stream.Config = .{};
    var all_windows = false;
    var listen_options = false;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg: []const u8 = args[i];
        if (std.mem.eql(u8, arg, "--help")) {
            try out.writeAll(usage);
            return 0;
        }
        if (std.mem.eql(u8, arg, "--all-windows")) {
            all_windows = true;
            listen_options = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--embedded")) {
            force_embedded = true;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--")) {
            if (i + 1 >= args.len) return usageError(out);
            i += 1;
            const value: []const u8 = args[i];
            if (std.mem.eql(u8, arg, "--far")) {
                far = std.fmt.parseFloat(f64, value) catch return usageError(out);
            } else if (std.mem.eql(u8, arg, "--models")) {
                models = value;
                models_explicit = true;
            } else if (std.mem.eql(u8, arg, "--model-dir")) {
                model_dir = value;
                models_explicit = true;
            } else if (std.mem.eql(u8, arg, "--google-dir")) {
                google_dir = value;
                models_explicit = true;
            } else if (std.mem.eql(u8, arg, "--g2p-dir")) {
                g2p_dir = value;
                models_explicit = true;
            } else if (std.mem.eql(u8, arg, "--window-ms")) {
                config.window_samples = try milliseconds(value);
                listen_options = true;
            } else if (std.mem.eql(u8, arg, "--hop-ms")) {
                config.hop_samples = try milliseconds(value);
                listen_options = true;
            } else if (std.mem.eql(u8, arg, "--cooldown-ms")) {
                config.cooldown_samples = try milliseconds(value);
                listen_options = true;
            } else if (std.mem.eql(u8, arg, "--trigger-windows")) {
                config.trigger_windows = try std.fmt.parseInt(usize, value, 10);
                listen_options = true;
            } else if (std.mem.eql(u8, arg, "--release-windows")) {
                config.release_windows = try std.fmt.parseInt(usize, value, 10);
                listen_options = true;
            } else return usageError(out);
        } else {
            if (count == positional.len) return usageError(out);
            positional[count] = arg;
            count += 1;
        }
    }
    if (count == 0) return usageError(out);
    const listening = std.mem.eql(u8, positional[0], "listen");
    if (listening) {
        if (count != 2) return usageError(out);
        config.far = far;
        try config.validate();
    } else if (count != 3 or !std.mem.eql(u8, positional[0], "detect") or listen_options) return usageError(out);

    if (models_explicit and force_embedded) return error.ConflictingModelOptions;
    const model = if (force_embedded or (bundled.available and !models_explicit)) try bundled.load(arena) else try naptrap.Model.load(
        io,
        arena,
        model_dir orelse try std.fmt.allocPrint(arena, "{s}/phonmatchnet", .{models}),
        google_dir orelse try std.fmt.allocPrint(arena, "{s}/google", .{models}),
        g2p_dir orelse try std.fmt.allocPrint(arena, "{s}/g2p", .{models}),
    );

    if (listening) {
        const keyword = try model.keyword(arena, positional[1]);
        return listen(io, &model, keyword, config, all_windows, out);
    }

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

fn milliseconds(text: []const u8) !usize {
    const value = try std.fmt.parseInt(usize, text, 10);
    return std.math.mul(usize, value, 16);
}

const ListenerOutput = struct {
    writer: *std.Io.Writer,
    keyword: []const u8,
    all_windows: bool,

    fn emit(self: *ListenerOutput, event: naptrap.stream.Event) !void {
        if (!event.triggered and !self.all_windows) return;
        try std.json.Stringify.value(.{
            .event = if (event.triggered) @as([]const u8, "wake") else "window",
            .keyword = self.keyword,
            .start_sample = event.start_sample,
            .end_sample = event.end_sample,
            .start_seconds = @as(f64, @floatFromInt(event.start_sample)) / 16000,
            .end_seconds = @as(f64, @floatFromInt(event.end_sample)) / 16000,
            .score = event.prediction.score,
            .threshold = event.prediction.threshold,
            .accepted = event.prediction.accepted,
            .triggered = event.triggered,
            .target_far = event.prediction.target_far,
            .keyword_logit = event.prediction.logit,
        }, .{}, self.writer);
        try self.writer.writeByte('\n');
        try self.writer.flush();
    }
};

fn listen(io: std.Io, model: *const naptrap.Model, keyword: naptrap.Keyword, config: naptrap.stream.Config, all_windows: bool, out: *std.Io.Writer) !u8 {
    var stream = try naptrap.Stream.init(std.heap.page_allocator, model, keyword, config);
    defer stream.deinit();
    var output = ListenerOutput{ .writer = out, .keyword = keyword.text, .all_windows = all_windows };
    var buffer: [4096]u8 = undefined;
    var input = std.Io.File.stdin().readerStreaming(io, &buffer);
    var bytes: [1024]u8 = undefined;
    var samples: [512]f32 = undefined;
    while (true) {
        const count = try input.interface.readSliceShort(&bytes);
        if (count == 0) break;
        if (count % 2 != 0) return error.TruncatedPcmSample;
        for (samples[0 .. count / 2], 0..) |*sample, index| {
            sample.* = @as(f32, @floatFromInt(std.mem.readInt(i16, bytes[index * 2 ..][0..2], .little))) / 32768;
        }
        try stream.push(samples[0 .. count / 2], &output, ListenerOutput.emit);
    }
    return 0;
}

fn usageError(out: *std.Io.Writer) u8 {
    out.writeAll(usage) catch {};
    return 2;
}
