//! Chunk-independent sliding-window detection on mono 16 kHz float32 PCM.
const std = @import("std");
const model_mod = @import("model.zig");

pub const Config = struct {
    window_samples: usize = 24000,
    hop_samples: usize = 4000,
    trigger_windows: usize = 2,
    release_windows: usize = 2,
    cooldown_samples: u64 = 16000,
    far: f64 = 0.01,

    pub fn validate(self: Config) !void {
        if (self.window_samples < 8000 or self.window_samples > 32000 or
            self.hop_samples == 0 or self.hop_samples > self.window_samples or
            self.trigger_windows == 0 or self.release_windows == 0)
            return error.InvalidStreamConfig;
    }
};

pub const Event = struct {
    /// Audio sample positions since construction/reset; end is exclusive.
    start_sample: u64,
    end_sample: u64,
    prediction: model_mod.Prediction,
    triggered: bool,
};

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

    pub fn init(allocator: std.mem.Allocator, model: *const model_mod.Model, keyword: model_mod.Keyword, config: Config) !Stream {
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

    pub fn deinit(self: *Stream) void {
        self.scratch.deinit();
        self.allocator.free(self.window);
        self.allocator.free(self.ring);
        self.* = undefined;
    }

    pub fn reset(self: *Stream) void {
        self.write_at = 0;
        self.total_samples = 0;
        self.next_end = self.config.window_samples;
        self.gate = .{};
        _ = self.scratch.reset(.retain_capacity);
    }

    pub fn push(self: *Stream, samples: []const f32, context: anytype, comptime emit: fn (@TypeOf(context), Event) anyerror!void) !void {
        for (samples) |x| if (!std.math.isFinite(x) or @abs(x) > 1) return error.InvalidSamples;
        const final_end = std.math.add(u64, self.total_samples, samples.len) catch return error.SampleCounterOverflow;
        if (final_end > std.math.maxInt(u64) - self.config.hop_samples) return error.SampleCounterOverflow;
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
                try emit(context, .{
                    .start_sample = self.total_samples - self.ring.len,
                    .end_sample = self.total_samples,
                    .prediction = prediction,
                    .triggered = triggered,
                });
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
