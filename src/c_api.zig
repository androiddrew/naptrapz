//! Stable C ABI; see include/naptrap.h for ownership and threading contracts.
const std = @import("std");
const naptrap = @import("naptrap");
const bundled = @import("naptrap_bundled");
// Handles and buffers come from the host's malloc.
const allocator = std.heap.c_allocator;

// Exported functions return c_int, which matches the naptrap_status enum in
// naptrap.h on every supported C ABI (C enums with these values are int-sized).
const Status = enum(c_int) { ok = 0, invalid_argument, invalid_audio, invalid_keyword, unknown_far, model_error, out_of_memory, busy, internal_error };
threadlocal var diagnostic: [256:0]u8 = @splat(0);
// The first library log message since the last status was returned. It is
// appended to the error name, so the library never writes to the host's stderr.
threadlocal var detail: [192]u8 = undefined;
threadlocal var detail_len: usize = 0;

pub const std_options: std.Options = .{ .logFn = captureLog };

fn captureLog(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    if (@intFromEnum(level) > @intFromEnum(std.log.Level.warn) or detail_len != 0) return;
    const prefix = if (scope == .default or scope == .naptrap) "" else @tagName(scope) ++ ": ";
    var writer: std.Io.Writer = .fixed(&detail);
    writer.print(prefix ++ format, args) catch {};
    detail_len = writer.end;
}

fn success() c_int {
    diagnostic[0] = 0;
    detail_len = 0;
    return @intFromEnum(Status.ok);
}

/// Record `err` (and any captured log detail) as this thread's diagnostic.
fn fail(err: anyerror, status: Status) c_int {
    var writer: std.Io.Writer = .fixed(diagnostic[0 .. diagnostic.len - 1]);
    writer.writeAll(@errorName(err)) catch {};
    if (detail_len != 0) writer.print(": {s}", .{detail[0..detail_len]}) catch {};
    diagnostic[writer.end] = 0;
    detail_len = 0;
    return @intFromEnum(status);
}

// The switches below are exhaustive, so adding or renaming an error in the
// core library forces a decision about its C status here.

fn loadStatus(err: anyerror) Status {
    // File-system errors are open-ended; everything but OOM is a model error.
    return if (err == error.OutOfMemory) .out_of_memory else .model_error;
}

fn keywordStatus(err: KeywordCreateError) Status {
    return switch (err) {
        error.OutOfMemory => .out_of_memory,
        error.KeywordTooLong, error.InvalidUtf8, error.SpellOutNumbers, error.NoLetters, error.NoPhoneticFeatures, error.FeatureLengthOutOfRange, error.ShapeMismatch, error.InvalidFeatures => .invalid_keyword,
    };
}

fn detectStatus(err: naptrap.model.DetectError) Status {
    return switch (err) {
        error.OutOfMemory => .out_of_memory,
        error.UnknownOperatingPoint => .unknown_far,
        error.ClipLengthOutOfRange, error.InvalidSamples => .invalid_audio,
        // Keywords are validated when created, and audio before inference.
        error.InvalidFeatures, error.ClipTooShort, error.InputTooShort, error.ShapeMismatch, error.FeatureLengthOutOfRange, error.FeatureLengthMismatch, error.UnexpectedEmbeddingShape, error.InvalidLogit => .internal_error,
    };
}

fn streamInitStatus(err: naptrap.stream.InitError) Status {
    return switch (err) {
        error.OutOfMemory => .out_of_memory,
        error.InvalidStreamConfig => .invalid_argument,
        error.UnknownOperatingPoint => .unknown_far,
        error.FeatureLengthOutOfRange, error.ShapeMismatch, error.InvalidFeatures => .internal_error,
    };
}

fn pushStatus(err: naptrap.stream.PushError) Status {
    return switch (err) {
        error.SampleCounterOverflow => .invalid_argument,
        else => |e| detectStatus(e),
    };
}

// One busy flag per model serializes every call on the model and its
// dependents. The Zig Model itself is immutable and thread-safe; this guard
// protects the per-model scratch arena and the reference counts.
const Engine = struct {
    scratch: std.heap.ArenaAllocator,
    model: naptrap.Model,
    references: usize = 1,
    busy: std.atomic.Value(bool) = .init(false),

    fn enter(self: *Engine) error{Busy}!void {
        if (self.busy.cmpxchgStrong(false, true, .acquire, .monotonic) != null) return error.Busy;
    }
    fn leave(self: *Engine) void {
        self.busy.store(false, .release);
    }
};

const Keyword = struct {
    engine: *Engine,
    arena: std.heap.ArenaAllocator,
    value: naptrap.Keyword,
    text: [:0]const u8,
    references: usize = 1,
};
const Stream = struct { keyword: *Keyword, value: naptrap.Stream };

const Prediction = extern struct {
    score: f32,
    logit: f32,
    threshold: f64,
    target_far: f64,
    accepted: u32,
    reserved: u32,
    fn from(value: naptrap.Prediction) Prediction {
        return .{ .score = value.score, .logit = value.logit, .threshold = value.threshold, .target_far = value.target_far, .accepted = @intFromBool(value.accepted), .reserved = 0 };
    }
};
const StreamConfig = extern struct {
    window_samples: u32,
    hop_samples: u32,
    trigger_windows: u32,
    release_windows: u32,
    cooldown_samples: u64,
    far: f64,
    fn native(self: StreamConfig) naptrap.stream.Config {
        return .{ .window_samples = self.window_samples, .hop_samples = self.hop_samples, .trigger_windows = self.trigger_windows, .release_windows = self.release_windows, .cooldown_samples = self.cooldown_samples, .far = self.far };
    }
};
const Event = extern struct {
    start_sample: u64,
    end_sample: u64,
    prediction: Prediction,
    triggered: u32,
    reserved: u32,
};
const Callback = *const fn (?*anyopaque, *const Event) callconv(.c) void;

export fn naptrap_abi_version() u32 {
    return 1;
}
export fn naptrap_has_embedded_models() u32 {
    return @intFromBool(bundled.available);
}
export fn naptrap_last_error() [*:0]const u8 {
    return &diagnostic;
}

fn load(model_dir: [*:0]const u8, google_dir: [*:0]const u8, g2p_dir: [*:0]const u8) !*Engine {
    const engine = try allocator.create(Engine);
    errdefer allocator.destroy(engine);
    engine.* = .{ .scratch = .init(allocator), .model = undefined };
    // Loading only reads files, so it needs neither threads nor an allocator.
    var io: std.Io.Threaded = .init_single_threaded;
    engine.model = try naptrap.Model.load(io.io(), allocator, std.mem.span(model_dir), std.mem.span(google_dir), std.mem.span(g2p_dir));
    return engine;
}

fn loadEmbedded() !*Engine {
    const engine = try allocator.create(Engine);
    errdefer allocator.destroy(engine);
    engine.* = .{ .scratch = .init(allocator), .model = undefined };
    engine.model = try bundled.load(allocator);
    return engine;
}

export fn naptrap_model_load_embedded(out: ?*?*Engine) c_int {
    const result = out orelse return fail(error.InvalidArgument, .invalid_argument);
    result.* = null;
    result.* = loadEmbedded() catch |err| return fail(err, loadStatus(err));
    return success();
}

export fn naptrap_model_load(model_dir: ?[*:0]const u8, google_dir: ?[*:0]const u8, g2p_dir: ?[*:0]const u8, out: ?*?*Engine) c_int {
    const result = out orelse return fail(error.InvalidArgument, .invalid_argument);
    result.* = null;
    if (model_dir == null or google_dir == null or g2p_dir == null) return fail(error.InvalidArgument, .invalid_argument);
    result.* = load(model_dir.?, google_dir.?, g2p_dir.?) catch |err| return fail(err, loadStatus(err));
    return success();
}

// Both helpers consume the lock, either unlocking or destroying its owner.
fn releaseEngineLocked(engine: *Engine) void {
    engine.references -= 1;
    if (engine.references != 0) return engine.leave();
    engine.scratch.deinit();
    engine.model.deinit();
    allocator.destroy(engine);
}
fn releaseKeywordLocked(keyword: *Keyword) void {
    keyword.references -= 1;
    if (keyword.references != 0) return keyword.engine.leave();
    const engine = keyword.engine;
    keyword.arena.deinit();
    allocator.destroy(keyword);
    releaseEngineLocked(engine);
}

export fn naptrap_model_release(model: ?*Engine) c_int {
    if (model) |engine| {
        engine.enter() catch |err| return fail(err, .busy);
        releaseEngineLocked(engine);
    }
    return success();
}

export fn naptrap_model_threshold(model: ?*Engine, far: f64, out: ?*f64) c_int {
    const result = out orelse return fail(error.InvalidArgument, .invalid_argument);
    result.* = 0;
    const engine = model orelse return fail(error.InvalidArgument, .invalid_argument);
    engine.enter() catch |err| return fail(err, .busy);
    defer engine.leave();
    result.* = engine.model.threshold(far) catch |err| return fail(err, .unknown_far);
    return success();
}

const KeywordCreateError = naptrap.text.EncodeError || naptrap.model.KeywordError;

fn createKeyword(engine: *Engine, text: []const u8) KeywordCreateError!*Keyword {
    const keyword = try allocator.create(Keyword);
    errdefer allocator.destroy(keyword);
    keyword.* = .{ .engine = engine, .arena = .init(allocator), .value = undefined, .text = undefined };
    errdefer keyword.arena.deinit();
    keyword.value = try engine.model.keyword(keyword.arena.allocator(), text);
    try naptrap.model.validateKeyword(keyword.value);
    keyword.text = try keyword.arena.allocator().dupeSentinel(u8, keyword.value.text, 0);
    engine.references += 1;
    return keyword;
}

export fn naptrap_keyword_create(model: ?*Engine, utf8: ?[*]const u8, length: usize, out: ?*?*Keyword) c_int {
    const result = out orelse return fail(error.InvalidArgument, .invalid_argument);
    result.* = null;
    const engine = model orelse return fail(error.InvalidArgument, .invalid_argument);
    if (utf8 == null and length != 0) return fail(error.InvalidArgument, .invalid_argument);
    engine.enter() catch |err| return fail(err, .busy);
    defer engine.leave();
    result.* = createKeyword(engine, if (utf8) |ptr| ptr[0..length] else "") catch |err| return fail(err, keywordStatus(err));
    return success();
}

export fn naptrap_keyword_text(keyword: ?*const Keyword) ?[*:0]const u8 {
    return if (keyword) |value| value.text.ptr else null;
}

export fn naptrap_keyword_release(keyword: ?*Keyword) c_int {
    if (keyword) |value| {
        value.engine.enter() catch |err| return fail(err, .busy);
        releaseKeywordLocked(value);
    }
    return success();
}

export fn naptrap_detect(model: ?*Engine, keyword: ?*const Keyword, samples: ?[*]const f32, count: usize, sample_rate: u32, far: f64, out: ?*Prediction) c_int {
    const result = out orelse return fail(error.InvalidArgument, .invalid_argument);
    result.* = std.mem.zeroes(Prediction);
    const engine = model orelse return fail(error.InvalidArgument, .invalid_argument);
    const kw = keyword orelse return fail(error.InvalidArgument, .invalid_argument);
    if (kw.engine != engine) return fail(error.InvalidArgument, .invalid_argument);
    if (samples == null) return fail(error.InvalidArgument, .invalid_argument);
    if (sample_rate != naptrap.features.sample_rate) return fail(error.InvalidSampleRate, .invalid_audio);
    engine.enter() catch |err| return fail(err, .busy);
    defer engine.leave();
    _ = engine.scratch.reset(.retain_capacity);
    const p = engine.model.detect(engine.scratch.allocator(), samples.?[0..count], kw.value, far) catch |err| return fail(err, detectStatus(err));
    result.* = Prediction.from(p);
    return success();
}

export fn naptrap_stream_config_default(out: ?*StreamConfig) void {
    const defaults: naptrap.stream.Config = .{};
    if (out) |config| config.* = .{
        .window_samples = defaults.window_samples,
        .hop_samples = defaults.hop_samples,
        .trigger_windows = defaults.trigger_windows,
        .release_windows = defaults.release_windows,
        .cooldown_samples = defaults.cooldown_samples,
        .far = defaults.far,
    };
}

fn createStream(keyword: *Keyword, config: naptrap.stream.Config) naptrap.stream.InitError!*Stream {
    const stream = try allocator.create(Stream);
    errdefer allocator.destroy(stream);
    stream.* = .{ .keyword = keyword, .value = try naptrap.Stream.init(allocator, &keyword.engine.model, keyword.value, config) };
    keyword.references += 1;
    return stream;
}

export fn naptrap_stream_create(keyword: ?*Keyword, config: ?*const StreamConfig, out: ?*?*Stream) c_int {
    const result = out orelse return fail(error.InvalidArgument, .invalid_argument);
    result.* = null;
    const kw = keyword orelse return fail(error.InvalidArgument, .invalid_argument);
    kw.engine.enter() catch |err| return fail(err, .busy);
    defer kw.engine.leave();
    result.* = createStream(kw, if (config) |c| c.native() else .{}) catch |err| return fail(err, streamInitStatus(err));
    return success();
}

const EmitContext = struct {
    callback: Callback,
    context: ?*anyopaque,
    fn emit(self: *EmitContext, event: naptrap.stream.Event) void {
        const value: Event = .{ .start_sample = event.start_sample, .end_sample = event.end_sample, .prediction = Prediction.from(event.prediction), .triggered = @intFromBool(event.triggered), .reserved = 0 };
        self.callback(self.context, &value);
    }
};

export fn naptrap_stream_push(stream: ?*Stream, samples: ?[*]const f32, count: usize, sample_rate: u32, callback: ?Callback, context: ?*anyopaque) c_int {
    const value = stream orelse return fail(error.InvalidArgument, .invalid_argument);
    const emit = callback orelse return fail(error.InvalidArgument, .invalid_argument);
    if (sample_rate != naptrap.features.sample_rate) return fail(error.InvalidSampleRate, .invalid_audio);
    if (samples == null and count != 0) return fail(error.InvalidArgument, .invalid_argument);
    value.keyword.engine.enter() catch |err| return fail(err, .busy);
    defer value.keyword.engine.leave();
    var output = EmitContext{ .callback = emit, .context = context };
    value.value.push(if (samples) |ptr| ptr[0..count] else &.{}, &output, EmitContext.emit) catch |err| return fail(err, pushStatus(err));
    return success();
}

export fn naptrap_stream_reset(stream: ?*Stream) c_int {
    const value = stream orelse return fail(error.InvalidArgument, .invalid_argument);
    value.keyword.engine.enter() catch |err| return fail(err, .busy);
    defer value.keyword.engine.leave();
    value.value.reset();
    return success();
}

export fn naptrap_stream_release(stream: ?*Stream) c_int {
    if (stream) |value| {
        const keyword = value.keyword;
        keyword.engine.enter() catch |err| return fail(err, .busy);
        value.value.deinit();
        allocator.destroy(value);
        releaseKeywordLocked(keyword);
    }
    return success();
}

comptime {
    // Must match the layouts asserted in tests/abi_smoke.c.
    std.debug.assert(@sizeOf(Prediction) == 32);
    std.debug.assert(@sizeOf(StreamConfig) == 32);
    std.debug.assert(@sizeOf(Event) == 56);
}
