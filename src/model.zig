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

pub const weights_file = "model.safetensors";
const max_weights_bytes = 256 << 20;

pub const Threshold = struct {
    far: f64,
    value: f64,
};

pub const Keyword = text_mod.G2p.Features;

pub fn validateKeyword(keyword: Keyword) !void {
    if (keyword.rows < 1 or keyword.rows > text_mod.max_rows) return error.FeatureLengthOutOfRange;
    if (keyword.data.len != keyword.rows * 256) return error.ShapeMismatch;
    for (keyword.data) |value| if (!std.math.isFinite(value)) return error.InvalidFeatures;
}

pub const Prediction = struct {
    score: f32,
    logit: f32,
    threshold: f64,
    accepted: bool,
    target_far: f64,
};

pub const Artifact = struct {
    config: std.json.Value,
    weights: st.File,
    sha256: [64]u8,
};

pub const ArtifactBytes = struct { config: []const u8, weights: []const u8 };

/// Read config.json, check its "format", verify the weights' SHA-256 and parse them.
pub fn readArtifact(io: std.Io, arena: std.mem.Allocator, dir: []const u8, format: []const u8) !Artifact {
    const cwd = std.Io.Dir.cwd();
    const config_path = try std.fmt.allocPrint(arena, "{s}/config.json", .{dir});
    const config_bytes = try cwd.readFileAlloc(io, config_path, arena, .limited(1 << 20));
    const weights_path = try std.fmt.allocPrint(arena, "{s}/" ++ weights_file, .{dir});
    const bytes = try cwd.readFileAlloc(io, weights_path, arena, .limited(max_weights_bytes));
    return parseArtifact(arena, .{ .config = config_bytes, .weights = bytes }, dir, format);
}

/// The same checks apply to disk and embedded artifacts.
pub fn parseArtifact(arena: std.mem.Allocator, source: ArtifactBytes, label: []const u8, format: []const u8) !Artifact {
    if (source.config.len > 1 << 20 or source.weights.len > max_weights_bytes) return error.ArtifactTooLarge;
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

pub fn jsonPath(root: std.json.Value, keys: []const []const u8) ?std.json.Value {
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

pub const Model = struct {
    network: network_mod.Network,
    google: google_mod.Google,
    g2p: text_mod.G2p,
    thresholds: []const Threshold,

    /// Everything, including the weights, is allocated from `arena`; keep it
    /// alive as long as the model.
    pub fn load(io: std.Io, arena: std.mem.Allocator, model_dir: []const u8, google_dir: []const u8, g2p_dir: []const u8) !Model {
        const main = try readArtifact(io, arena, model_dir, "naptrap-phonmatchnet-v1");
        const google = try readArtifact(io, arena, google_dir, "naptrap-google-speech-embedding-v1");
        const g2p = try readArtifact(io, arena, g2p_dir, "naptrap-g2p-en-decoder-v1");
        return fromArtifacts(arena, main, google, g2p);
    }

    /// Load without file I/O. All retained model data is copied into arena;
    /// the input byte buffers may be released after this function returns.
    pub fn fromBytes(arena: std.mem.Allocator, main_bytes: ArtifactBytes, google_bytes: ArtifactBytes, g2p_bytes: ArtifactBytes) !Model {
        const main = try parseArtifact(arena, main_bytes, "phonmatchnet", "naptrap-phonmatchnet-v1");
        const google = try parseArtifact(arena, google_bytes, "google", "naptrap-google-speech-embedding-v1");
        const g2p = try parseArtifact(arena, g2p_bytes, "g2p", "naptrap-g2p-en-decoder-v1");
        return fromArtifacts(arena, main, google, g2p);
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
            .network = try network_mod.Network.init(arena, main.weights),
            .google = try google_mod.Google.init(arena, google.weights),
            .g2p = g2p_encoder,
            .thresholds = thresholds.items,
        };
    }

    pub fn threshold(self: Model, far: f64) !f64 {
        for (self.thresholds) |t| {
            if (t.far == far) return t.value;
        }
        return error.UnknownOperatingPoint;
    }

    /// Precompute a keyword's text features; reuse across clips.
    pub fn keyword(self: Model, arena: std.mem.Allocator, keyword_text: []const u8) !Keyword {
        return self.g2p.encode(arena, keyword_text);
    }

    /// Score one mono 16 kHz clip of 0.5-2 s, samples in [-1, 1].
    /// `scratch` holds per-call buffers; an arena you reset per clip works well.
    pub fn detect(self: Model, scratch: std.mem.Allocator, wave: []const f32, kw: Keyword, far: f64) !Prediction {
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
