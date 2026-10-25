//! Port of Google's speech_embedding/1 (Apache-2.0): 16 kHz clip ->
//! [len / 1280][96] embeddings.
//!
//! log-mel [T][32] -> Conv2d(1->24, 3x3, pad (0,1)) -> ReLU -> BN -> act
//! -> 18 x [Conv2d (1x3 pad (0,1) on even index, 3x1 on odd) -> BN -> act],
//!    channels 24 -> 48 (i=2) -> 72 (i=6) -> 96 (i=10),
//!    max-pool 2x2 after i = 1, 9, 17 and 1x2 after i = 5, 13
//! -> Conv2d(96->96, 3x1) -> [96][G][1] -> transpose -> [G][96]
//!
//! act(x) = max(leaky_relu(x, 0.2), -0.4). BatchNorm eps is 0.001.
//! Tensor layout here is [C][time][freq].

const std = @import("std");
const ops = @import("ops.zig");
const st = @import("safetensors.zig");
const features = @import("features.zig");

pub const embedding_size = 96;
pub const samples_per_embedding = 1280;
const bn_eps = 0.001;

const Layer = struct {
    weight: []const f32,
    c_out: usize,
    kh: usize,
    kw: usize,
    ph: usize,
    pw: usize,
    pool_h: usize, // 0 = no pooling after this layer
    pool_w: usize,
};

pub const Google = struct {
    window: []const f32,
    mel: []const f32,
    initial: []const f32,
    layers: [18]Layer,
    norms: [19]ops.Affine,
    output: []const f32,

    pub fn init(arena: std.mem.Allocator, file: st.File) !Google {
        var self: Google = .{
            .window = try file.expect("window", &.{features.win_length}),
            .mel = try file.expect("mel_matrix", &.{ features.bins, features.google_mels }),
            .initial = try file.expect("initial.weight", &.{ 24, 1, 3, 3 }),
            .layers = undefined,
            .norms = undefined,
            .output = try file.expect("output.weight", &.{ 96, 96, 3, 1 }),
        };
        self.norms[0] = try loadNorm(arena, file, 0, 24);

        var channels: usize = 24;
        for (&self.layers, 0..) |*layer, i| {
            const target: usize = switch (i) {
                2 => 48,
                6 => 72,
                10 => 96,
                else => channels,
            };
            const frequency = i % 2 == 0;
            const kh: usize = if (frequency) 1 else 3;
            const kw: usize = if (frequency) 3 else 1;
            const name = try std.fmt.allocPrint(arena, "convs.{d}.weight", .{i});
            layer.* = .{
                .weight = try file.expect(name, &.{ target, channels, kh, kw }),
                .c_out = target,
                .kh = kh,
                .kw = kw,
                .ph = 0,
                .pw = if (frequency) 1 else 0,
                .pool_h = switch (i) {
                    1, 9, 17 => 2,
                    5, 13 => 1,
                    else => 0,
                },
                .pool_w = switch (i) {
                    1, 5, 9, 13, 17 => 2,
                    else => 0,
                },
            };
            self.norms[i + 1] = try loadNorm(arena, file, i + 1, target);
            channels = target;
        }
        return self;
    }

    fn loadNorm(arena: std.mem.Allocator, file: st.File, index: usize, channels: usize) !ops.Affine {
        var names: [4][]const u8 = undefined;
        for (&names, [_][]const u8{ "weight", "bias", "running_mean", "running_var" }) |*name, suffix| {
            name.* = try std.fmt.allocPrint(arena, "norms.{d}.{s}", .{ index, suffix });
        }
        return ops.Affine.fromBatchNorm(
            arena,
            try file.expect(names[0], &.{channels}),
            try file.expect(names[1], &.{channels}),
            try file.expect(names[2], &.{channels}),
            try file.expect(names[3], &.{channels}),
            bn_eps,
        );
    }

    fn activation(data: []f32) void {
        for (data) |*x| {
            const leaky = if (x.* < 0) 0.2 * x.* else x.*;
            x.* = @max(leaky, -0.4);
        }
    }

    pub fn logMel(self: Google, arena: std.mem.Allocator, dft: features.Dft, wave: []const f32) ![]f32 {
        return features.googleLogMel(arena, dft, wave, self.window, self.mel);
    }

    pub const Embeddings = struct {
        data: []const f32, // [rows][96]
        rows: usize,
    };

    /// log_mel is [frames][32].
    pub fn encodeLogMel(self: Google, arena: std.mem.Allocator, log_mel: []const f32, frames: usize) !Embeddings {
        std.debug.assert(log_mel.len == frames * features.google_mels);
        const input = try arena.dupe(f32, log_mel);
        var x = ops.Map{ .data = input, .c = 1, .h = frames, .w = features.google_mels };

        x = try ops.conv2d(arena, x, self.initial, 24, 3, 3, 0, 1);
        ops.relu(x.data);
        self.norms[0].apply(x.data, x.h * x.w);
        activation(x.data);

        for (self.layers, 1..) |layer, norm| {
            x = try ops.conv2d(arena, x, layer.weight, layer.c_out, layer.kh, layer.kw, layer.ph, layer.pw);
            self.norms[norm].apply(x.data, x.h * x.w);
            activation(x.data);
            if (layer.pool_h != 0) x = try ops.maxPool2d(arena, x, layer.pool_h, layer.pool_w);
        }

        const y = try ops.conv2d(arena, x, self.output, embedding_size, 3, 1, 0, 0);
        if (y.w != 1) return error.UnexpectedEmbeddingShape;
        // [96][G][1] -> [G][96]
        return .{ .data = try ops.transpose(arena, y.data, embedding_size, y.h), .rows = y.h };
    }

    pub fn embed(self: Google, arena: std.mem.Allocator, dft: features.Dft, wave: []const f32) !Embeddings {
        const log_mel = try self.logMel(arena, dft, wave);
        const result = try self.encodeLogMel(arena, log_mel, features.googleFrames(wave.len));
        if (result.rows != wave.len / samples_per_embedding) return error.UnexpectedEmbeddingShape;
        return result;
    }
};
