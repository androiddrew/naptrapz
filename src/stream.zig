//! Chunk-independent sliding-window detection on mono 16 kHz float32 PCM.
const std = @import("std");
const model_mod = @import("model.zig");

/// Sliding-window and debounce settings. Durations are in 16 kHz samples.
pub const Config = struct {
    /// Samples scored per window, 8000-32000 (0.5-2 s).
    window_samples: usize = 24000,
    /// Samples between window ends, 1 to `window_samples`.
    hop_samples: usize = 4000,
    /// Consecutive accepted windows needed to trigger a wake.
    trigger_windows: usize = 2,
    /// Consecutive rejected windows needed to rearm after a wake.
    release_windows: usize = 2,
    /// Minimum samples between the ends of two triggering windows.
    cooldown_samples: u64 = 16000,
    /// Calibrated operating point passed to `Model.detect`.
    far: f64 = 0.01,

    /// error.InvalidStreamConfig if a field is out of range.
    pub fn validate(self: Config) error{InvalidStreamConfig}!void {
        if (self.window_samples < 8000 or self.window_samples > 32000 or
            self.hop_samples == 0 or self.hop_samples > self.window_samples or
            self.trigger_windows == 0 or self.release_windows == 0)
            return error.InvalidStreamConfig;
    }
};

/// The result for one complete window.
pub const Event = struct {
    /// Audio sample positions since construction/reset; end is exclusive.
    start_sample: u64,
    end_sample: u64,
    prediction: model_mod.Prediction,
    /// Whether this window raised a debounced wake event.
    triggered: bool,
};

/// Errors from `Stream.init`.
pub const InitError = std.mem.Allocator.Error || model_mod.KeywordError || error{ InvalidStreamConfig, UnknownOperatingPoint };

/// Errors from `Stream.push`, not counting the callback's own errors.
pub const PushError = model_mod.DetectError || error{
    /// More than 2^64 samples would have been pushed since the last reset.
    SampleCounterOverflow,
};

fn CallbackError(comptime emit: anytype) type {
    const Return = @typeInfo(@TypeOf(emit)).@"fn".return_type.?;
    return switch (@typeInfo(Return)) {
        .error_union => |info| info.error_set,
        else => error{},
    };
}

const Gate = struct {
    positive: usize = 0,
    negative: usize = 0,
    armed: bool = true,
    last_trigger: ?u64 = null,

    fn update(self: *Gate, config: Config, accepted: bool, end: u64) bool {
        if (!accepted) {
            self.positive = 0;
            self.negative = @min(self.negative +| 1, config.release_windows);
            if (self.negative == config.release_windows) self.armed = true;
            return false;
        }
        self.negative = 0;
        self.positive = @min(self.positive +| 1, config.trigger_windows);
        if (!self.armed or self.positive < config.trigger_windows) return false;
        if (self.last_trigger) |last| if (end - last < config.cooldown_samples) return false;
        self.armed = false;
        self.last_trigger = end;
        return true;
    }
};

/// A stream borrows its model and keyword, which must outlive it. Use one stream
/// per audio source and serialize calls. Callbacks run synchronously once per
/// complete window and must not reenter this stream. On a callback/inference
/// error, reset the stream before feeding more audio. Invalid sample chunks are
/// rejected before changing state. EOF does not pad or evaluate partial windows.
pub const Stream = struct {
    model: *const model_mod.Model,
    keyword: model_mod.Keyword,
    config: Config,
    allocator: std.mem.Allocator,
    ring: []f32,
    window: []f32,
    scratch: std.heap.ArenaAllocator,
    write_at: usize = 0,
    total_samples: u64 = 0,
    next_end: u64,
    gate: Gate = .{},

    /// Validate `config` and allocate two window buffers from `allocator`.
    pub fn init(allocator: std.mem.Allocator, model: *const model_mod.Model, keyword: model_mod.Keyword, config: Config) InitError!Stream {
        try config.validate();
        _ = try model.threshold(config.far);
        try model_mod.validateKeyword(keyword);
        const ring = try allocator.alloc(f32, config.window_samples);
        errdefer allocator.free(ring);
        const window = try allocator.alloc(f32, config.window_samples);
        return .{
            .model = model,
            .keyword = keyword,
            .config = config,
            .allocator = allocator,
            .ring = ring,
            .window = window,
            .scratch = std.heap.ArenaAllocator.init(allocator),
            .next_end = config.window_samples,
        };
    }

    /// Free the stream's buffers.
    pub fn deinit(self: *Stream) void {
        self.scratch.deinit();
        self.allocator.free(self.window);
        self.allocator.free(self.ring);
        self.* = undefined;
    }

    /// Start a new audio timeline: clear buffered audio, sample positions and
    /// debounce state. Scratch capacity is kept.
    pub fn reset(self: *Stream) void {
        self.write_at = 0;
        self.total_samples = 0;
        self.next_end = self.config.window_samples;
        self.gate = .{};
        _ = self.scratch.reset(.retain_capacity);
    }

    /// Append `samples` (any length, finite, in [-1, 1]) and call
    /// `emit(context, event)` once for every window that completes. `emit` may
    /// return `void` or an error union; its errors are passed through.
    /// error.InvalidSamples leaves the stream unchanged.
    pub fn push(self: *Stream, samples: []const f32, context: anytype, comptime emit: anytype) (PushError || CallbackError(emit))!void {
        for (samples) |x| if (!std.math.isFinite(x) or @abs(x) > 1) return error.InvalidSamples;
        const final_end = std.math.add(u64, self.total_samples, samples.len) catch return error.SampleCounterOverflow;
        if (final_end > std.math.maxInt(u64) - @as(u64, self.config.hop_samples)) return error.SampleCounterOverflow;
        var offset: usize = 0;
        while (offset < samples.len) {
            const count: usize = @intCast(@min(samples.len - offset, self.next_end - self.total_samples));
            const first = @min(count, self.ring.len - self.write_at);
            @memcpy(self.ring[self.write_at..][0..first], samples[offset..][0..first]);
            @memcpy(self.ring[0 .. count - first], samples[offset + first ..][0 .. count - first]);
            self.write_at = (self.write_at + count) % self.ring.len;
            self.total_samples += count;
            offset += count;
            if (self.total_samples == self.next_end) {
                const split = self.ring.len - self.write_at;
                @memcpy(self.window[0..split], self.ring[self.write_at..]);
                @memcpy(self.window[split..], self.ring[0..self.write_at]);
                _ = self.scratch.reset(.retain_capacity);
                const prediction = try self.model.detect(self.scratch.allocator(), self.window, self.keyword, self.config.far);
                const triggered = self.gate.update(self.config, prediction.accepted, self.total_samples);
                self.next_end += self.config.hop_samples;
                const event: Event = .{
                    .start_sample = self.total_samples - self.ring.len,
                    .end_sample = self.total_samples,
                    .prediction = prediction,
                    .triggered = triggered,
                };
                const Return = @typeInfo(@TypeOf(emit)).@"fn".return_type.?;
                if (@typeInfo(Return) == .error_union) try emit(context, event) else emit(context, event);
            }
        }
    }
};

test "debounce requires consecutive positives, release, and audio-time cooldown" {
    var gate: Gate = .{};
    const config: Config = .{ .cooldown_samples = 16000 };
    try std.testing.expect(!gate.update(config, true, 24000));
    try std.testing.expect(!gate.update(config, false, 28000));
    try std.testing.expect(!gate.update(config, true, 32000));
    try std.testing.expect(gate.update(config, true, 36000));
    try std.testing.expect(!gate.update(config, true, 40000));
    try std.testing.expect(!gate.update(config, false, 44000));
    try std.testing.expect(!gate.update(config, false, 48000));
    try std.testing.expect(!gate.update(config, true, 52000));
    try std.testing.expect(gate.update(config, true, 56000));
    gate = .{};
    const quick: Config = .{ .trigger_windows = 1, .release_windows = 1, .cooldown_samples = 16000 };
    try std.testing.expect(gate.update(quick, true, 24000));
    try std.testing.expect(!gate.update(quick, false, 28000));
    try std.testing.expect(!gate.update(quick, true, 32000));
    try std.testing.expect(!gate.update(quick, true, 36000));
    try std.testing.expect(gate.update(quick, true, 40000));
}
