//! naptrap detect <clip.wav> <keyword> [options]
//! naptrap listen <keyword> [options]
//!
//! `detect` scores one mono 16 kHz 16-bit PCM WAV and prints the prediction as
//! JSON. Exit code: 0 accepted, 1 rejected, 2 error (the same contract as the
//! Python CLI). `listen` reads raw PCM from stdin and prints one JSON line per
//! wake event (or per window with --all-windows); it exits 0 at end of input.
//! Usage errors go to stderr.

const std = @import("std");
const naptrap = @import("naptrap");
const bundled = @import("naptrap_bundled");

const usage =
    \\usage: naptrap detect <clip.wav> <keyword> [--far 0.01] [model options]
    \\       naptrap listen <keyword> [--far 0.01] [model options]
    \\                      [--window-ms 1500] [--hop-ms 250] [--trigger-windows 2]
    \\                      [--release-windows 2] [--cooldown-ms 1000] [--all-windows]
    \\model options: [--models DIR] [--model-dir DIR] [--google-dir DIR] [--g2p-dir DIR]
    \\               [--embedded]
    \\
    \\--models DIR holds phonmatchnet/, google/ and g2p/ (default: models).
    \\listen reads raw mono 16 kHz signed 16-bit little-endian PCM from stdin.
    \\Bundled builds use embedded models by default. Explicit model paths override them.
    \\--embedded requires embedded models and cannot be combined with model paths.
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    var stderr_buffer: [1024]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &stderr_writer.interface;

    const code: u8 = run(io, arena, init.gpa, args, stdout, stderr) catch |err| blk: {
        std.log.err("{s}", .{@errorName(err)});
        break :blk 2;
    };
    stdout.flush() catch {};
    stderr.flush() catch {};
    return code;
}

fn run(io: std.Io, arena: std.mem.Allocator, gpa: std.mem.Allocator, args: []const [:0]const u8, out: *std.Io.Writer, err_out: *std.Io.Writer) !u8 {
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
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
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
            if (i + 1 >= args.len) return usageError(err_out, "{s} needs a value", .{arg});
            i += 1;
            const value: []const u8 = args[i];
            const invalid = "invalid value for {s}: '{s}'";
            if (std.mem.eql(u8, arg, "--far")) {
                far = std.fmt.parseFloat(f64, value) catch return usageError(err_out, invalid, .{ arg, value });
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
                config.window_samples = milliseconds(value) catch return usageError(err_out, invalid, .{ arg, value });
                listen_options = true;
            } else if (std.mem.eql(u8, arg, "--hop-ms")) {
                config.hop_samples = milliseconds(value) catch return usageError(err_out, invalid, .{ arg, value });
                listen_options = true;
            } else if (std.mem.eql(u8, arg, "--cooldown-ms")) {
                config.cooldown_samples = milliseconds(value) catch return usageError(err_out, invalid, .{ arg, value });
                listen_options = true;
            } else if (std.mem.eql(u8, arg, "--trigger-windows")) {
                config.trigger_windows = std.fmt.parseInt(usize, value, 10) catch return usageError(err_out, invalid, .{ arg, value });
                listen_options = true;
            } else if (std.mem.eql(u8, arg, "--release-windows")) {
                config.release_windows = std.fmt.parseInt(usize, value, 10) catch return usageError(err_out, invalid, .{ arg, value });
                listen_options = true;
            } else return usageError(err_out, "unknown option {s}", .{arg});
        } else {
            if (count == positional.len) return usageError(err_out, "unexpected argument '{s}'", .{arg});
            positional[count] = arg;
            count += 1;
        }
    }
    if (count == 0) return usageError(err_out, "missing command", .{});
    const listening = std.mem.eql(u8, positional[0], "listen");
    if (listening) {
        if (count != 2) return usageError(err_out, "listen takes one keyword", .{});
        config.far = far;
        config.validate() catch return usageError(err_out, "invalid listen timing options", .{});
    } else if (!std.mem.eql(u8, positional[0], "detect")) {
        return usageError(err_out, "unknown command '{s}'", .{positional[0]});
    } else if (count != 3) {
        return usageError(err_out, "detect takes a WAV path and a keyword", .{});
    } else if (listen_options) {
        return usageError(err_out, "streaming options only apply to listen", .{});
    }

    if (models_explicit and force_embedded) return usageError(err_out, "--embedded cannot be combined with model paths", .{});
    var model = if (force_embedded or (bundled.available and !models_explicit)) try bundled.load(gpa) else try naptrap.Model.load(
        io,
        gpa,
        model_dir orelse try std.fmt.allocPrint(arena, "{s}/phonmatchnet", .{models}),
        google_dir orelse try std.fmt.allocPrint(arena, "{s}/google", .{models}),
        g2p_dir orelse try std.fmt.allocPrint(arena, "{s}/g2p", .{models}),
    );
    defer model.deinit();

    if (listening) {
        const keyword = try model.keyword(gpa, positional[1]);
        defer keyword.deinit(gpa);
        return listen(io, gpa, &model, keyword, config, all_windows, out);
    }

    const wav_bytes = try std.Io.Dir.cwd().readFileAlloc(io, positional[1], arena, .limited(16 << 20));
    const wave = try naptrap.wav.decode(arena, wav_bytes);
    const kw = try model.keyword(gpa, positional[2]);
    defer kw.deinit(gpa);
    const p = try model.detect(arena, wave, kw, far);

    try std.json.Stringify.value(.{
        .model = "phonmatchnet",
        .task = "C-KWS",
        .keyword = kw.text,
        .score = p.score,
        .threshold = p.threshold,
        .accepted = p.accepted,
        .target_far = p.target_far,
        .details = .{ .keyword_logit = p.logit },
    }, .{}, out);
    try out.writeByte('\n');
    return if (p.accepted) 0 else 1;
}

fn milliseconds(text: []const u8) !usize {
    const value = try std.fmt.parseInt(usize, text, 10);
    return std.math.mul(usize, value, naptrap.features.sample_rate / 1000);
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
            .start_seconds = @as(f64, @floatFromInt(event.start_sample)) / naptrap.features.sample_rate,
            .end_seconds = @as(f64, @floatFromInt(event.end_sample)) / naptrap.features.sample_rate,
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

fn listen(io: std.Io, gpa: std.mem.Allocator, model: *const naptrap.Model, keyword: naptrap.Keyword, config: naptrap.stream.Config, all_windows: bool, out: *std.Io.Writer) !u8 {
    var stream = try naptrap.Stream.init(gpa, model, keyword, config);
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

fn usageError(err_out: *std.Io.Writer, comptime format: []const u8, args: anytype) u8 {
    err_out.print("naptrap: " ++ format ++ "\n", args) catch {};
    err_out.writeAll(usage) catch {};
    return 2;
}
