//! End-to-end tests against the released model artifacts. Run with
//! `zig build test -Dtest-models=DIR`, where DIR holds phonmatchnet/, google/
//! and g2p/. Without that option every test here is skipped.

const std = @import("std");
const naptrap = @import("naptrap");
const options = @import("test_options");

const testing = std.testing;
const gpa = testing.allocator;

fn modelPath(comptime part: []const u8) []const u8 {
    return options.models_dir ++ "/" ++ part;
}

fn loadModel() !naptrap.Model {
    if (options.models_dir.len == 0) return error.SkipZigTest;
    return naptrap.Model.load(testing.io, gpa, modelPath("phonmatchnet"), modelPath("google"), modelPath("g2p"));
}

/// Deterministic noise in [-amplitude, amplitude].
fn noise(buffer: []f32, seed: u64, amplitude: f32) void {
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    for (buffer) |*x| x.* = (random.float(f32) * 2 - 1) * amplitude;
}

test "load and fromBytes give the same model" {
    var model = try loadModel();
    defer model.deinit();

    var parts: [3]naptrap.model.ArtifactBytes = undefined;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    for (&parts, [_][]const u8{ "phonmatchnet", "google", "g2p" }) |*part, name| {
        var dir = try std.Io.Dir.cwd().openDir(testing.io, try std.Io.Dir.path.join(arena, &.{ options.models_dir, name }), .{});
        defer dir.close(testing.io);
        part.* = .{
            .config = try dir.readFileAlloc(testing.io, "config.json", arena, .unlimited),
            .weights = try dir.readFileAlloc(testing.io, "model.safetensors", arena, .unlimited),
        };
    }
    var copy = try naptrap.Model.fromBytes(gpa, parts[0], parts[1], parts[2]);
    // The inputs may be released as soon as fromBytes returns.
    arena_state.deinit();
    arena_state = .init(gpa);
    defer copy.deinit();

    const kw = try model.keyword(gpa, "hey computer");
    defer kw.deinit(gpa);
    var wave: [24000]f32 = undefined;
    noise(&wave, 1, 0.3);
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const a = try model.detect(scratch.allocator(), &wave, kw, 0.01);
    const b = try copy.detect(scratch.allocator(), &wave, kw, 0.01);
    try testing.expectEqual(a.logit, b.logit);
}

test "thresholds and operating points" {
    var model = try loadModel();
    defer model.deinit();
    try testing.expect(model.thresholds.len >= 1);
    for (model.thresholds) |t| try testing.expectEqual(t.value, try model.threshold(t.far));
    try testing.expectError(error.UnknownOperatingPoint, model.threshold(0.5));
    try testing.expectError(error.UnknownOperatingPoint, model.threshold(std.math.nan(f64)));
}

test "keyword features" {
    var model = try loadModel();
    defer model.deinit();

    const kw = try model.keyword(gpa, "Hey, COMPUTER!");
    defer kw.deinit(gpa);
    try testing.expectEqualStrings("hey computer", kw.text);
    try testing.expectEqual(kw.rows * 256, kw.data.len);
    try naptrap.model.validateKeyword(kw);

    try testing.expectError(error.SpellOutNumbers, model.keyword(gpa, "route 66"));
    try testing.expectError(error.NoLetters, model.keyword(gpa, "?!"));
    try testing.expectError(error.KeywordTooLong, model.keyword(gpa, "a" ** 1025));
    try testing.expectError(error.FeatureLengthOutOfRange, model.keyword(gpa, "hello " ** 100));
}

test "detect validates its input" {
    var model = try loadModel();
    defer model.deinit();
    const kw = try model.keyword(gpa, "hey computer");
    defer kw.deinit(gpa);
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const s = scratch.allocator();

    var wave = [_]f32{0} ** 32001;
    try testing.expectError(error.ClipLengthOutOfRange, model.detect(s, wave[0..7999], kw, 0.01));
    try testing.expectError(error.ClipLengthOutOfRange, model.detect(s, &wave, kw, 0.01));
    try testing.expectError(error.UnknownOperatingPoint, model.detect(s, wave[0..16000], kw, 0.5));
    wave[10] = 1.5;
    try testing.expectError(error.InvalidSamples, model.detect(s, wave[0..16000], kw, 0.01));
    wave[10] = std.math.nan(f32);
    try testing.expectError(error.InvalidSamples, model.detect(s, wave[0..16000], kw, 0.01));
    wave[10] = 0;

    // Every supported length runs, and silence is not the keyword.
    for ([_]usize{ 8000, 8001, 16000, 24000, 31999, 32000 }) |len| {
        _ = scratch.reset(.retain_capacity);
        const p = try model.detect(s, wave[0..len], kw, 0.01);
        try testing.expect(!p.accepted);
        try testing.expect(p.score >= 0 and p.score <= 1);
    }
}

const Recorder = struct {
    events: std.ArrayList(naptrap.stream.Event) = .empty,

    fn record(self: *Recorder, event: naptrap.stream.Event) error{OutOfMemory}!void {
        try self.events.append(gpa, event);
    }
};

fn countWindows(count: *usize, _: naptrap.stream.Event) void {
    count.* += 1;
}

fn failOnSecondWindow(count: *usize, _: naptrap.stream.Event) error{Stop}!void {
    count.* += 1;
    if (count.* == 2) return error.Stop;
}

test "stream results do not depend on chunk size" {
    var model = try loadModel();
    defer model.deinit();
    const kw = try model.keyword(gpa, "hey computer");
    defer kw.deinit(gpa);

    const audio = try gpa.alloc(f32, 3 * 16000 + 123);
    defer gpa.free(audio);
    noise(audio, 2, 0.2);
    const config: naptrap.stream.Config = .{ .window_samples = 16000, .hop_samples = 8000 };

    var whole: Recorder = .{};
    defer whole.events.deinit(gpa);
    var stream = try naptrap.Stream.init(gpa, &model, kw, config);
    defer stream.deinit();
    try stream.push(audio, &whole, Recorder.record);
    try testing.expectEqual(@as(usize, 5), whole.events.items.len);

    var chunked: Recorder = .{};
    defer chunked.events.deinit(gpa);
    stream.reset();
    var prng: std.Random.DefaultPrng = .init(3);
    var offset: usize = 0;
    while (offset < audio.len) {
        const len = @min(audio.len - offset, prng.random().uintAtMost(usize, 5000));
        try stream.push(audio[offset..][0..len], &chunked, Recorder.record);
        offset += len;
    }
    try testing.expectEqual(whole.events.items.len, chunked.events.items.len);
    for (whole.events.items, chunked.events.items) |a, b| {
        try testing.expectEqual(a.start_sample, b.start_sample);
        try testing.expectEqual(a.end_sample, b.end_sample);
        try testing.expectEqual(a.prediction.logit, b.prediction.logit);
        try testing.expectEqual(a.triggered, b.triggered);
    }
}

test "stream callbacks and invalid chunks" {
    var model = try loadModel();
    defer model.deinit();
    const kw = try model.keyword(gpa, "hey computer");
    defer kw.deinit(gpa);
    var stream = try naptrap.Stream.init(gpa, &model, kw, .{ .window_samples = 8000, .hop_samples = 4000 });
    defer stream.deinit();

    var audio = [_]f32{0} ** 16000;
    var count: usize = 0;
    // A void callback needs no `try` inside push.
    try stream.push(&audio, &count, countWindows);
    try testing.expectEqual(@as(usize, 3), count);

    // Invalid samples are rejected before any state changes.
    audio[5] = 2;
    try testing.expectError(error.InvalidSamples, stream.push(&audio, &count, countWindows));
    audio[5] = 0;
    try stream.push(audio[0..4000], &count, countWindows);
    try testing.expectEqual(@as(usize, 4), count);

    // Callback errors are passed through unchanged.
    stream.reset();
    count = 0;
    try testing.expectError(error.Stop, stream.push(&audio, &count, failOnSecondWindow));

    try testing.expectError(error.InvalidStreamConfig, naptrap.Stream.init(gpa, &model, kw, .{ .hop_samples = 0 }));
    try testing.expectError(error.UnknownOperatingPoint, naptrap.Stream.init(gpa, &model, kw, .{ .far = 0.5 }));
}
