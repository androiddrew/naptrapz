//! Stable C ABI; see include/naptrap.h for ownership and threading contracts.
const std = @import("std");
const naptrap = @import("naptrap");
const bundled = @import("naptrap_bundled");
const allocator = std.heap.page_allocator;

const Status = enum(c_int) { ok = 0, invalid_argument, invalid_audio, invalid_keyword, unknown_far, model_error, out_of_memory, busy, internal_error };
threadlocal var diagnostic: [256:0]u8 = @splat(0);

fn success() c_int {
    diagnostic[0] = 0;
    return @intFromEnum(Status.ok);
}

fn fail(err: anyerror, fallback: Status) c_int {
    const name = @errorName(err);
    const count = @min(name.len, diagnostic.len - 1);
    @memcpy(diagnostic[0..count], name[0..count]);
    diagnostic[count] = 0;
    const status: Status = switch (err) {
        error.OutOfMemory => .out_of_memory,
        error.InvalidArgument, error.InvalidStreamConfig, error.SampleCounterOverflow => .invalid_argument,
        error.InvalidSamples, error.ClipLengthOutOfRange, error.InvalidSampleRate => .invalid_audio,
        error.InvalidUtf8, error.SpellOutNumbers, error.NoLetters, error.NoPhoneticFeatures, error.FeatureLengthOutOfRange, error.InvalidFeatures => .invalid_keyword,
        error.UnknownOperatingPoint => .unknown_far,
        error.Busy => .busy,
        else => fallback,
    };
    return @intFromEnum(status);
}

const Engine = struct {
    arena: std.heap.ArenaAllocator,
    scratch: std.heap.ArenaAllocator,
    model: naptrap.Model,
    references: usize = 1,
    busy: std.atomic.Value(bool) = .init(false),

    fn enter(self: *Engine) !void {
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
    engine.* = .{ .arena = std.heap.ArenaAllocator.init(allocator), .scratch = std.heap.ArenaAllocator.init(allocator), .model = undefined };
    errdefer engine.arena.deinit();
    var io = std.Io.Threaded.init(allocator, .{});
    defer io.deinit();
    engine.model = try naptrap.Model.load(io.io(), engine.arena.allocator(), std.mem.span(model_dir), std.mem.span(google_dir), std.mem.span(g2p_dir));
    return engine;
}

fn loadEmbedded() !*Engine {
    const engine = try allocator.create(Engine);
    errdefer allocator.destroy(engine);
    engine.* = .{ .arena = std.heap.ArenaAllocator.init(allocator), .scratch = std.heap.ArenaAllocator.init(allocator), .model = undefined };
    errdefer engine.arena.deinit();
    engine.model = try bundled.load(engine.arena.allocator());
    return engine;
}

export fn naptrap_model_load_embedded(out: ?*?*Engine) c_int {
    const result = out orelse return fail(error.InvalidArgument, .invalid_argument);
    result.* = null;
    result.* = loadEmbedded() catch |err| return fail(err, .model_error);
    return success();
}

export fn naptrap_model_load(model_dir: ?[*:0]const u8, google_dir: ?[*:0]const u8, g2p_dir: ?[*:0]const u8, out: ?*?*Engine) c_int {
    const result = out orelse return fail(error.InvalidArgument, .invalid_argument);
    result.* = null;
    if (model_dir == null or google_dir == null or g2p_dir == null) return fail(error.InvalidArgument, .invalid_argument);
    result.* = load(model_dir.?, google_dir.?, g2p_dir.?) catch |err| return fail(err, .model_error);
    return success();
}

// Both helpers consume the lock, either unlocking or destroying its owner.
fn releaseEngineLocked(engine: *Engine) void {
    engine.references -= 1;
    if (engine.references != 0) return engine.leave();
    engine.scratch.deinit();
    engine.arena.deinit();
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

fn createKeyword(engine: *Engine, text: []const u8) !*Keyword {
    const keyword = try allocator.create(Keyword);
    errdefer allocator.destroy(keyword);
    keyword.* = .{ .engine = engine, .arena = std.heap.ArenaAllocator.init(allocator), .value = undefined, .text = undefined };
    errdefer keyword.arena.deinit();
    keyword.value = try engine.model.keyword(keyword.arena.allocator(), text);
    try naptrap.model.validateKeyword(keyword.value);
    keyword.text = try keyword.arena.allocator().dupeZ(u8, keyword.value.text);
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
    result.* = createKeyword(engine, if (utf8) |ptr| ptr[0..length] else "") catch |err| return fail(err, .invalid_keyword);
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
    if (sample_rate != 16000) return fail(error.InvalidSampleRate, .invalid_audio);
    if (samples == null or count < 8000 or count > 32000) return fail(error.ClipLengthOutOfRange, .invalid_audio);
    engine.enter() catch |err| return fail(err, .busy);
    defer engine.leave();
    _ = engine.scratch.reset(.retain_capacity);
    const p = engine.model.detect(engine.scratch.allocator(), samples.?[0..count], kw.value, far) catch |err| return fail(err, .internal_error);
    result.* = Prediction.from(p);
    return success();
}

export fn naptrap_stream_config_default(out: ?*StreamConfig) void {
    if (out) |config| config.* = .{ .window_samples = 24000, .hop_samples = 4000, .trigger_windows = 2, .release_windows = 2, .cooldown_samples = 16000, .far = 0.01 };
}

fn createStream(keyword: *Keyword, config: naptrap.stream.Config) !*Stream {
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
    result.* = createStream(kw, if (config) |c| c.native() else .{}) catch |err| return fail(err, .invalid_argument);
    return success();
}

const EmitContext = struct {
    callback: Callback,
    context: ?*anyopaque,
    fn emit(self: *EmitContext, event: naptrap.stream.Event) !void {
        const value: Event = .{ .start_sample = event.start_sample, .end_sample = event.end_sample, .prediction = Prediction.from(event.prediction), .triggered = @intFromBool(event.triggered), .reserved = 0 };
        self.callback(self.context, &value);
    }
};

export fn naptrap_stream_push(stream: ?*Stream, samples: ?[*]const f32, count: usize, sample_rate: u32, callback: ?Callback, context: ?*anyopaque) c_int {
    const value = stream orelse return fail(error.InvalidArgument, .invalid_argument);
    const emit = callback orelse return fail(error.InvalidArgument, .invalid_argument);
    if (sample_rate != 16000) return fail(error.InvalidSampleRate, .invalid_audio);
    if (samples == null and count != 0) return fail(error.InvalidArgument, .invalid_argument);
    value.keyword.engine.enter() catch |err| return fail(err, .busy);
    defer value.keyword.engine.leave();
    var output = EmitContext{ .callback = emit, .context = context };
    value.value.push(if (samples) |ptr| ptr[0..count] else &.{}, &output, EmitContext.emit) catch |err| return fail(err, .internal_error);
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
