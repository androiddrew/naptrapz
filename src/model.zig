//! Loads the three naptrap artifact directories and runs end-to-end detection.
//!
//! Each directory holds config.json and model.safetensors, exactly as
//! downloaded from Hugging Face:
//!   androiddrew/naptrap-phonmatchnet-en       (format naptrap-phonmatchnet-v1)
//!   androiddrew/google-speech-embedding-pytorch (naptrap-google-speech-embedding-v1)
//!   androiddrew/g2p-en-decoder-pytorch        (naptrap-g2p-en-decoder-v1)

const std = @import("std");
const st = @import("safetensors.zig");
const ops = @import("ops.zig");
const features = @import("features.zig");
const google_mod = @import("google.zig");
const text_mod = @import("text.zig");
const network_mod = @import("phonmatchnet.zig");

const log = std.log.scoped(.naptrap);

/// Weights file name inside each artifact directory.
pub const weights_file = "model.safetensors";
const max_weights_bytes = 256 << 20;

/// One calibrated operating point: score threshold `value` for target per-clip false-accept rate `far`.
pub const Threshold = struct {
    far: f64,
    value: f64,
};

/// Precomputed keyword text features; see `Model.keyword`.
pub const Keyword = text_mod.G2p.Features;

/// Errors for keyword features that the matcher cannot consume.
pub const KeywordError = error{ FeatureLengthOutOfRange, ShapeMismatch, InvalidFeatures };

/// Inference failures that validated input should never produce.
pub const InferenceError = std.mem.Allocator.Error || error{
    ClipTooShort,
    InputTooShort,
    ShapeMismatch,
    FeatureLengthOutOfRange,
    FeatureLengthMismatch,
    UnexpectedEmbeddingShape,
};

/// Errors from `Model.detect`.
pub const DetectError = KeywordError || InferenceError || error{
    /// `far` is not one of the model's calibrated operating points.
    UnknownOperatingPoint,
    /// Fewer than 8000 or more than 32000 samples.
    ClipLengthOutOfRange,
    /// A sample is not finite or lies outside [-1, 1].
    InvalidSamples,
    /// The network produced a non-finite logit.
    InvalidLogit,
};

/// Check that keyword features have 1 to `text.max_rows` finite 256-wide rows.
pub fn validateKeyword(keyword: Keyword) KeywordError!void {
    if (keyword.rows < 1 or keyword.rows > text_mod.max_rows) return error.FeatureLengthOutOfRange;
    if (keyword.data.len != keyword.rows * 256) return error.ShapeMismatch;
    for (keyword.data) |value| if (!std.math.isFinite(value)) return error.InvalidFeatures;
}

/// The result of scoring one clip.
pub const Prediction = struct {
    /// sigmoid(logit), in [0, 1].
    score: f32,
    /// The matcher's raw keyword logit.
    logit: f32,
    /// Score threshold of the selected operating point.
    threshold: f64,
    /// Whether score >= threshold.
    accepted: bool,
    /// The operating point that was requested.
    target_far: f64,
};

/// The two files of one artifact directory, already in memory.
pub const ArtifactBytes = struct { config: []const u8, weights: []const u8 };

const Artifact = struct {
    config: std.json.Value,
    weights: st.File,
    sha256: [64]u8,
};

const max_config_bytes = 1 << 20;

/// Read config.json and model.safetensors from `dir` into `gpa`.
fn readArtifact(io: std.Io, gpa: std.mem.Allocator, dir: []const u8) !ArtifactBytes {
    var root = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer root.close(io);
    const config = try root.readFileAlloc(io, "config.json", gpa, .limited(max_config_bytes));
    errdefer gpa.free(config);
    const weights = try root.readFileAlloc(io, weights_file, gpa, .limited(max_weights_bytes));
    return .{ .config = config, .weights = weights };
}

/// Check the config's "format", verify the weights' SHA-256 and parse them.
/// The same checks apply to disk and embedded artifacts.
fn parseArtifact(arena: std.mem.Allocator, source: ArtifactBytes, label: []const u8, format: []const u8) !Artifact {
    if (source.config.len > max_config_bytes or source.weights.len > max_weights_bytes) return error.ArtifactTooLarge;
    const config = std.json.parseFromSliceLeaky(std.json.Value, arena, source.config, .{}) catch
        return error.InvalidConfig;

    const found = jsonPath(config, &.{"format"}) orelse return error.InvalidConfig;
    if (found != .string or !std.mem.eql(u8, found.string, format)) {
        log.err("{s} is not a {s} artifact", .{ label, format });
        return error.WrongArtifactFormat;
    }

    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(source.weights, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);

    const expected = jsonPath(config, &.{ "sha256", weights_file }) orelse return error.InvalidConfig;
    if (expected != .string or !std.mem.eql(u8, expected.string, &hex)) {
        log.err("{s} checksum mismatch in {s}", .{ weights_file, label });
        return error.ChecksumMismatch;
    }
    return .{ .config = config, .weights = try st.File.parse(arena, source.weights), .sha256 = hex };
}

fn jsonPath(root: std.json.Value, keys: []const []const u8) ?std.json.Value {
    var value = root;
    for (keys) |key| {
        if (value != .object) return null;
        value = value.object.get(key) orelse return null;
    }
    return value;
}

fn jsonNumber(value: std.json.Value) ?f64 {
    return switch (value) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch null,
        else => null,
    };
}

/// A loaded keyword model with its two frozen encoders.
///
/// A Model is immutable after loading: `keyword`, `threshold` and `detect`
/// may run concurrently from several threads, each `detect` call with its own
/// scratch allocator. Release it with `deinit`.
pub const Model = struct {
    /// Owns the decoded weights and everything else the model retains.
    arena: std.heap.ArenaAllocator,
    network: network_mod.Network,
    google: google_mod.Google,
    g2p: text_mod.G2p,
    /// Calibrated operating points from the model's config.json.
    thresholds: []const Threshold,

    const formats = [3][]const u8{ "naptrap-phonmatchnet-v1", "naptrap-google-speech-embedding-v1", "naptrap-g2p-en-decoder-v1" };

    /// Load the three artifact directories, each holding config.json and
    /// model.safetensors. The file contents are freed before returning; the
    /// model retains only its decoded weights, allocated from `gpa`.
    pub fn load(io: std.Io, gpa: std.mem.Allocator, model_dir: []const u8, google_dir: []const u8, g2p_dir: []const u8) !Model {
        const dirs = [3][]const u8{ model_dir, google_dir, g2p_dir };
        var files: [3]ArtifactBytes = undefined;
        var read: usize = 0;
        defer for (files[0..read]) |f| {
            gpa.free(f.config);
            gpa.free(f.weights);
        };
        for (&files, dirs) |*f, dir| {
            f.* = try readArtifact(io, gpa, dir);
            read += 1;
        }
        return fromSources(gpa, files, dirs);
    }

    /// Load from in-memory artifacts without file I/O. All retained data is
    /// copied, so the input buffers may be released after this returns.
    pub fn fromBytes(gpa: std.mem.Allocator, main_bytes: ArtifactBytes, google_bytes: ArtifactBytes, g2p_bytes: ArtifactBytes) !Model {
        return fromSources(gpa, .{ main_bytes, google_bytes, g2p_bytes }, .{ "phonmatchnet", "google", "g2p" });
    }

    /// Free everything the model owns. Keywords and streams created from it
    /// must not be used afterwards.
    pub fn deinit(self: *Model) void {
        self.arena.deinit();
        self.* = undefined;
    }

    fn fromSources(gpa: std.mem.Allocator, sources: [3]ArtifactBytes, labels: [3][]const u8) !Model {
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena_state.deinit();
        const arena = arena_state.allocator();
        var artifacts: [3]Artifact = undefined;
        for (&artifacts, sources, labels, formats) |*artifact, source, label, format| {
            artifact.* = try parseArtifact(arena, source, label, format);
        }
        var model = try fromArtifacts(arena, artifacts[0], artifacts[1], artifacts[2]);
        model.arena = arena_state;
        return model;
    }

    fn fromArtifacts(arena: std.mem.Allocator, main: Artifact, google: Artifact, g2p: Artifact) !Model {
        // The keyword model pins the exact encoders it was trained with.
        inline for (.{ .{ "google", &google }, .{ "g2p", &g2p } }) |pair| {
            if (jsonPath(main.config, &.{ "requires", pair[0], "sha256" })) |pinned| {
                if (pinned != .string or !std.mem.eql(u8, pinned.string, &pair[1].sha256)) {
                    log.err("the {s} encoder is not the one this model was trained with", .{pair[0]});
                    return error.EncoderMismatch;
                }
            }
        }

        const source = jsonPath(google.config, &.{ "source", "archive_sha256" }) orelse return error.InvalidConfig;
        if (source != .string or !std.mem.eql(u8, source.string, google_mod.source_archive_sha256))
            return error.UnexpectedGoogleSource;

        const g2p_encoder = try text_mod.G2p.init(g2p.weights);
        const hidden_size = jsonNumber(jsonPath(g2p.config, &.{"hidden_size"}) orelse return error.InvalidConfig) orelse return error.InvalidConfig;
        const phonemes = jsonNumber(jsonPath(g2p.config, &.{"phonemes"}) orelse return error.InvalidConfig) orelse return error.InvalidConfig;
        // The matcher consumes 256-wide decoder states. Config and weights must
        // describe the same encoder, as in Python's strict state_dict loading.
        if (g2p_encoder.size != 256 or hidden_size != @as(f64, @floatFromInt(g2p_encoder.size)) or
            phonemes != @as(f64, @floatFromInt(g2p_encoder.phonemes)))
            return error.G2pConfigMismatch;

        const table = jsonPath(main.config, &.{"thresholds"}) orelse return error.InvalidConfig;
        if (table != .object) return error.InvalidConfig;
        var thresholds: std.ArrayList(Threshold) = .empty;
        var it = table.object.iterator();
        while (it.next()) |kv| {
            const far = std.fmt.parseFloat(f64, kv.key_ptr.*) catch return error.InvalidConfig;
            const value = jsonNumber(kv.value_ptr.*) orelse return error.InvalidConfig;
            if (!std.math.isFinite(value)) return error.InvalidConfig;
            try thresholds.append(arena, .{ .far = far, .value = value });
        }
        if (thresholds.items.len == 0) return error.InvalidConfig;

        return .{
            .arena = undefined, // set by fromSources
            .network = try network_mod.Network.init(arena, main.weights),
            .google = try google_mod.Google.init(arena, google.weights),
            .g2p = g2p_encoder,
            .thresholds = thresholds.items,
        };
    }

    /// The score threshold for the calibrated false-accept rate `far`, which
    /// must exactly equal one of `thresholds` (0.01 or 0.1 for the released
    /// model); otherwise error.UnknownOperatingPoint.
    pub fn threshold(self: Model, far: f64) error{UnknownOperatingPoint}!f64 {
        for (self.thresholds) |t| {
            if (t.far == far) return t.value;
        }
        return error.UnknownOperatingPoint;
    }

    /// Normalize `keyword_text` and compute its text features once, for reuse
    /// across clips. The result is allocated from `gpa`; free it with
    /// `Keyword.deinit`.
    pub fn keyword(self: Model, gpa: std.mem.Allocator, keyword_text: []const u8) text_mod.EncodeError!Keyword {
        return self.g2p.encode(gpa, keyword_text);
    }

    /// Score one mono 16 kHz clip of 0.5-2 s (8000-32000 samples), samples
    /// finite and in [-1, 1], against `kw` at operating point `far`.
    ///
    /// `scratch` receives about 2-3 MiB of per-call buffers that are never
    /// freed individually: pass an arena (or a FixedBufferAllocator) and
    /// reset it between calls, e.g. `arena.reset(.retain_capacity)`.
    pub fn detect(self: Model, scratch: std.mem.Allocator, wave: []const f32, kw: Keyword, far: f64) DetectError!Prediction {
        const limit = try self.threshold(far);
        try validateKeyword(kw);
        if (wave.len < features.sample_rate / 2 or wave.len > features.sample_rate * 2) return error.ClipLengthOutOfRange;
        for (wave) |x| {
            if (!std.math.isFinite(x) or @abs(x) > 1) return error.InvalidSamples;
        }

        const mel = try self.network.keywordMel(scratch, wave);
        const t = mel.len / features.keyword_mels;
        const emb = try self.google.embed(scratch, wave);
        const logit = try self.network.forward(scratch, mel, t, emb.data, emb.rows, kw.data, kw.rows);
        if (!std.math.isFinite(logit)) return error.InvalidLogit;

        const score = ops.sigmoid(logit);
        return .{
            .score = score,
            .logit = logit,
            .threshold = limit,
            .accepted = @as(f64, score) >= limit,
            .target_far = far,
        };
    }
};
