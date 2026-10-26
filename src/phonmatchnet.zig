//! PhonMatchNet matcher (Lee & Cho, Interspeech 2023), single-clip inference.
//!
//!   keyword_mel [T][40] -> Conv1d(40->128, k3 s2 p1) -> BN -> ReLU
//!                       -> Conv1d(128->128, k3 p1)   -> BN -> ReLU   = value [T1][128]
//!   google [G][96]      -> ConvTranspose1d(96->128, k5 s4) -> LeakyReLU,
//!                          zero-padded or cropped to T1              = pretrained
//!   text [P][256]       -> Linear(256->128) -> LeakyReLU             = textual
//!   joint = concat(value + pretrained, textual) along time   [T1 + P][128]
//!   -> single-head causal self-attention -> GRU(128) -> last hidden
//!   -> Linear(128->1) = keyword logit
//!
//! Weight names follow naptrap's PyTorch state dict.

const std = @import("std");
const ops = @import("ops.zig");
const st = @import("safetensors.zig");
const features = @import("features.zig");

const d = 128;
const bn_eps = 1e-5;
const leaky_slope = 0.01;
const max_text_rows = @import("text.zig").max_rows;

/// The PhonMatchNet matcher. Weight slices are borrowed from the safetensors
/// file it was initialized from.
pub const Network = struct {
    kw_window: []const f32,
    kw_mel: []const f32,
    conv1_w: []const f32,
    conv1_b: []const f32,
    bn1: ops.Affine,
    conv2_w: []const f32,
    conv2_b: []const f32,
    bn2: ops.Affine,
    up_w: []const f32,
    up_b: []const f32,
    text_w: []const f32,
    text_b: []const f32,
    in_w: []const f32,
    in_b: []const f32,
    out_w: []const f32,
    out_b: []const f32,
    gru: ops.Gru,
    head_w: []const f32,
    head_b: []const f32,

    /// Check every tensor shape in `file` and fold its BatchNorms into `arena`.
    pub fn init(arena: std.mem.Allocator, file: st.File) !Network {
        return .{
            .kw_window = try file.expect("audio.frontend.window", &.{features.win_length}),
            .kw_mel = try file.expect("audio.frontend.mel_matrix", &.{ features.bins, features.keyword_mels }),
            .conv1_w = try file.expect("audio.conv1.weight", &.{ d, 40, 3 }),
            .conv1_b = try file.expect("audio.conv1.bias", &.{d}),
            .bn1 = try loadNorm(arena, file, "audio.bn1.bn"),
            .conv2_w = try file.expect("audio.conv2.weight", &.{ d, d, 3 }),
            .conv2_b = try file.expect("audio.conv2.bias", &.{d}),
            .bn2 = try loadNorm(arena, file, "audio.bn2.bn"),
            .up_w = try file.expect("google_upsample.weight", &.{ 96, d, 5 }),
            .up_b = try file.expect("google_upsample.bias", &.{d}),
            .text_w = try file.expect("text_projection.weight", &.{ d, 256 }),
            .text_b = try file.expect("text_projection.bias", &.{d}),
            .in_w = try file.expect("attention.in_proj_weight", &.{ 3 * d, d }),
            .in_b = try file.expect("attention.in_proj_bias", &.{3 * d}),
            .out_w = try file.expect("attention.out_proj.weight", &.{ d, d }),
            .out_b = try file.expect("attention.out_proj.bias", &.{d}),
            .gru = .{
                .input = d,
                .hidden = d,
                .w_ih = try file.expect("utterance_gru.weight_ih_l0", &.{ 3 * d, d }),
                .w_hh = try file.expect("utterance_gru.weight_hh_l0", &.{ 3 * d, d }),
                .b_ih = try file.expect("utterance_gru.bias_ih_l0", &.{3 * d}),
                .b_hh = try file.expect("utterance_gru.bias_hh_l0", &.{3 * d}),
            },
            .head_w = try file.expect("utterance_head.weight", &.{ 1, d }),
            .head_b = try file.expect("utterance_head.bias", &.{1}),
        };
    }

    fn loadNorm(arena: std.mem.Allocator, file: st.File, comptime prefix: []const u8) !ops.Affine {
        return ops.Affine.fromBatchNorm(
            arena,
            try file.expect(prefix ++ ".weight", &.{d}),
            try file.expect(prefix ++ ".bias", &.{d}),
            try file.expect(prefix ++ ".running_mean", &.{d}),
            try file.expect(prefix ++ ".running_var", &.{d}),
            bn_eps,
        );
    }

    /// The matcher's 40-bin keyword log-mel: [ceil(len / 160)][40].
    pub fn keywordMel(self: Network, arena: std.mem.Allocator, wave: []const f32) ![]f32 {
        return features.keywordMel(arena, wave, self.kw_window, self.kw_mel);
    }

    /// mel [t][40], google [g][96], text [p][256] -> keyword logit.
    pub fn forward(self: Network, arena: std.mem.Allocator, mel: []const f32, t: usize, google: []const f32, g: usize, text: []const f32, p: usize) !f32 {
        // Same input checks as naptrap's predict().
        if (t < 50 or t > 200 or g < 6 or g > 25 or p < 1 or p > max_text_rows) return error.FeatureLengthOutOfRange;
        if (t < 8 * g or t > 8 * g + 8) return error.FeatureLengthMismatch;
        if (mel.len != t * 40 or google.len != g * 96 or text.len != p * 256) return error.ShapeMismatch;

        // Audio branch: [40][t] -> [128][t1]
        const mel_ct = try ops.transpose(arena, mel, t, 40);
        const t1 = ops.conv1dLen(t, 3, 2, 1);
        const c1 = try arena.alloc(f32, d * t1);
        ops.conv1d(c1, mel_ct, 40, t, self.conv1_w, self.conv1_b, d, 3, 2, 1);
        self.bn1.apply(c1, t1);
        ops.relu(c1);
        const c2 = try arena.alloc(f32, d * t1);
        ops.conv1d(c2, c1, d, t1, self.conv2_w, self.conv2_b, d, 3, 1, 1);
        self.bn2.apply(c2, t1);
        ops.relu(c2);

        // Google branch: [96][g] -> [128][up_len]
        const google_ct = try ops.transpose(arena, google, g, 96);
        const up_len = ops.convTranspose1dLen(g, 5, 4);
        const up = try arena.alloc(f32, d * up_len);
        ops.convTranspose1d(up, google_ct, 96, g, self.up_w, self.up_b, d, 5, 4);
        ops.leakyRelu(up, leaky_slope);

        // Joint sequence [s][128]: audio rows, then text rows.
        const s = t1 + p;
        const joint = try arena.alloc(f32, s * d);
        for (0..t1) |r| {
            for (0..d) |c| {
                const pretrained: f32 = if (r < up_len) up[c * up_len + r] else 0;
                joint[r * d + c] = c2[c * t1 + r] + pretrained;
            }
        }
        const textual = joint[t1 * d ..];
        ops.linear(textual, text, p, self.text_w, self.text_b, 256, d);
        ops.leakyRelu(textual, leaky_slope);

        // Causal single-head attention.
        const qkv = try arena.alloc(f32, s * 3 * d);
        ops.linear(qkv, joint, s, self.in_w, self.in_b, d, 3 * d);
        const scale: f32 = 1.0 / @sqrt(@as(f32, d));
        // Attention rows are independent. Project each row and advance the GRU
        // immediately instead of retaining two complete [s,128] intermediates.
        var weights_buffer: [ops.conv1dLen(200, 3, 2, 1) + max_text_rows]f32 = undefined;
        const weights = weights_buffer[0..s];
        var q: [d]f32 = undefined;
        var row: [d]f32 = undefined;
        var attended: [d]f32 = undefined;
        var h = [_]f32{0} ** d;
        var scratch: [6 * d]f32 = undefined;
        for (0..s) |i| {
            for (&q, qkv[i * 3 * d ..][0..d]) |*dst, src| dst.* = src * scale;
            for (0..i + 1) |j| weights[j] = ops.dot(&q, qkv[j * 3 * d + d ..][0..d]);
            ops.softmax(weights[0 .. i + 1]);
            @memset(&row, 0);
            for (0..i + 1) |j| {
                const v = qkv[j * 3 * d + 2 * d ..][0..d];
                for (&row, v) |*a, vv| a.* += weights[j] * vv;
            }
            ops.linear(&attended, &row, 1, self.out_w, self.out_b, d, d);
            self.gru.step(&attended, &h, &scratch);
        }
        return ops.dot(self.head_w, &h) + self.head_b[0];
    }
};
