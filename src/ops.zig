//! fp32 kernels for batch-1 inference.
//!
//! Layouts follow PyTorch so weights load without reshuffling:
//!   Linear           weight [out][in]
//!   Conv1d / Conv2d  weight [out][in][k] / [out][in][kh][kw]
//!   ConvTranspose1d  weight [in][out][k]
//! Activations are channel-major: [C][L] for 1-D and [C][H][W] for 2-D.
//!
//! Most kernels are plain loops over an 8-lane @Vector dot product;
//! conv2dInto packs eight output positions per SIMD vector.
//!
//! The void kernels (dot, linear, conv1d, convTranspose1d, Affine.apply,
//! Gru.step) assert their slice-length preconditions, which are
//! only checked in Debug and ReleaseSafe builds. Functions that return an
//! error union validate their inputs in every build mode.

const std = @import("std");

pub fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const V = @Vector(8, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= a.len) : (i += 8) {
        const va: V = a[i..][0..8].*;
        const vb: V = b[i..][0..8].*;
        acc += va * vb;
    }
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// out[r][o] = x[r] · w[o] + b[o]
pub fn linear(out: []f32, x: []const f32, rows: usize, w: []const f32, b: ?[]const f32, in_f: usize, out_f: usize) void {
    std.debug.assert(x.len == rows * in_f and w.len == out_f * in_f and out.len == rows * out_f);
    for (0..rows) |r| {
        const xr = x[r * in_f ..][0..in_f];
        for (0..out_f) |o| {
            const bias: f32 = if (b) |bb| bb[o] else 0;
            out[r * out_f + o] = dot(xr, w[o * in_f ..][0..in_f]) + bias;
        }
    }
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub fn relu(xs: []f32) void {
    for (xs) |*x| x.* = @max(x.*, 0);
}

pub fn leakyRelu(xs: []f32, slope: f32) void {
    for (xs) |*x| {
        if (x.* < 0) x.* *= slope;
    }
}

pub fn softmax(xs: []f32) void {
    var max: f32 = -std.math.inf(f32);
    for (xs) |x| max = @max(max, x);
    var sum: f32 = 0;
    for (xs) |*x| {
        x.* = @exp(x.* - max);
        sum += x.*;
    }
    for (xs) |*x| x.* /= sum;
}

/// First index of the maximum, like torch.argmax.
pub fn argmax(xs: []const f32) usize {
    var best: usize = 0;
    for (xs, 0..) |x, i| {
        if (x > xs[best]) best = i;
    }
    return best;
}

/// [rows][cols] -> [cols][rows]
pub fn transpose(alloc: std.mem.Allocator, x: []const f32, rows: usize, cols: usize) ![]f32 {
    if (x.len != rows * cols) return error.ShapeMismatch;
    const out = try alloc.alloc(f32, x.len);
    for (0..rows) |r| {
        for (0..cols) |c| out[c * rows + r] = x[r * cols + c];
    }
    return out;
}

/// Inference-mode BatchNorm folded into a per-channel scale and shift.
pub const Affine = struct {
    scale: []f32,
    shift: []f32,

    pub fn fromBatchNorm(alloc: std.mem.Allocator, weight: []const f32, bias: []const f32, mean: []const f32, variance: []const f32, eps: f32) !Affine {
        if (bias.len != weight.len or mean.len != weight.len or variance.len != weight.len) return error.ShapeMismatch;
        const scale = try alloc.alloc(f32, weight.len);
        const shift = try alloc.alloc(f32, weight.len);
        for (0..weight.len) |c| {
            scale[c] = weight[c] / @sqrt(variance[c] + eps);
            shift[c] = bias[c] - mean[c] * scale[c];
        }
        return .{ .scale = scale, .shift = shift };
    }

    /// data is [channels][plane]
    pub fn apply(self: Affine, data: []f32, plane: usize) void {
        std.debug.assert(data.len == self.scale.len * plane);
        for (self.scale, self.shift, 0..) |s, t, c| {
            for (data[c * plane ..][0..plane]) |*x| x.* = x.* * s + t;
        }
    }
};

/// One step of a PyTorch-style GRU (gate order r, z, n; the reset gate scales
/// the hidden-side candidate term including its bias). Updates h in place.
pub const Gru = struct {
    input: usize,
    hidden: usize,
    w_ih: []const f32, // [3H][input]
    w_hh: []const f32, // [3H][H]
    b_ih: []const f32, // [3H]
    b_hh: []const f32, // [3H]

    /// scratch.len must be at least 6 * hidden.
    pub fn step(self: Gru, x: []const f32, h: []f32, scratch: []f32) void {
        const H = self.hidden;
        const gi = scratch[0 .. 3 * H];
        const gh = scratch[3 * H .. 6 * H];
        linear(gi, x, 1, self.w_ih, self.b_ih, self.input, 3 * H);
        linear(gh, h, 1, self.w_hh, self.b_hh, H, 3 * H);
        for (0..H) |j| {
            const r = sigmoid(gi[j] + gh[j]);
            const z = sigmoid(gi[H + j] + gh[H + j]);
            const n = std.math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
            h[j] = (1 - z) * n + z * h[j];
        }
    }
};

pub fn conv1dLen(len: usize, k: usize, stride: usize, pad: usize) usize {
    return (len + 2 * pad - k) / stride + 1;
}

/// x [c_in][len] -> out [c_out][conv1dLen(len, k, stride, pad)], zero padding.
pub fn conv1d(out: []f32, x: []const f32, c_in: usize, len: usize, w: []const f32, b: ?[]const f32, c_out: usize, k: usize, stride: usize, pad: usize) void {
    const out_len = conv1dLen(len, k, stride, pad);
    std.debug.assert(out.len == c_out * out_len and w.len == c_out * c_in * k and x.len == c_in * len);
    for (0..c_out) |o| {
        for (0..out_len) |t| {
            var acc: f32 = if (b) |bb| bb[o] else 0;
            for (0..c_in) |c| {
                for (0..k) |j| {
                    const pos = t * stride + j; // position in the padded input
                    if (pos < pad or pos - pad >= len) continue;
                    acc += w[(o * c_in + c) * k + j] * x[c * len + pos - pad];
                }
            }
            out[o * out_len + t] = acc;
        }
    }
}

pub fn convTranspose1dLen(len: usize, k: usize, stride: usize) usize {
    return (len - 1) * stride + k;
}

/// x [c_in][len] -> out [c_out][(len - 1) * stride + k], no padding.
pub fn convTranspose1d(out: []f32, x: []const f32, c_in: usize, len: usize, w: []const f32, b: ?[]const f32, c_out: usize, k: usize, stride: usize) void {
    const out_len = convTranspose1dLen(len, k, stride);
    std.debug.assert(out.len == c_out * out_len and w.len == c_in * c_out * k and x.len == c_in * len);
    for (0..c_out) |o| @memset(out[o * out_len ..][0..out_len], if (b) |bb| bb[o] else 0);
    for (0..c_in) |c| {
        for (0..len) |i| {
            const xv = x[c * len + i];
            for (0..c_out) |o| {
                for (0..k) |j| out[o * out_len + i * stride + j] += xv * w[(c * c_out + o) * k + j];
            }
        }
    }
}

/// A [c][h][w] feature map.
pub const Map = struct {
    data: []f32,
    c: usize,
    h: usize,
    w: usize,

    pub fn readOnly(self: Map) ConstMap {
        return .{ .data = self.data, .c = self.c, .h = self.h, .w = self.w };
    }
};

pub const ConstMap = struct { data: []const f32, c: usize, h: usize, w: usize };
pub const ConvVector = @Vector(8, f32);

/// Stride-1 Conv2d without bias, zero padding (ph, pw).
pub fn conv2d(alloc: std.mem.Allocator, x: Map, w: []const f32, c_out: usize, kh: usize, kw: usize, ph: usize, pw: usize) !Map {
    std.debug.assert(w.len == c_out * x.c * kh * kw);
    if (x.h + 2 * ph < kh or x.w + 2 * pw < kw) return error.InputTooShort;
    const oh = x.h + 2 * ph - kh + 1;
    const ow = x.w + 2 * pw - kw + 1;
    const plane = oh * ow;
    const inner = x.c * kh * kw;
    const out = try alloc.alloc(f32, c_out * plane);
    errdefer alloc.free(out);
    const patches = try alloc.alloc(ConvVector, inner);
    defer alloc.free(patches);
    return conv2dInto(out, patches, x.readOnly(), w, c_out, kh, kw, ph, pw);
}

/// Caller-owned, non-overlapping output and patch buffers. Returned Map is a
/// view into out. Reuse buffers across layers to bound feature-map storage.
pub fn conv2dInto(out: []f32, workspace: []ConvVector, x: ConstMap, w: []const f32, c_out: usize, kh: usize, kw: usize, ph: usize, pw: usize) !Map {
    if (kh == 0 or kw == 0 or x.h + 2 * ph < kh or x.w + 2 * pw < kw) return error.InputTooShort;
    const oh = x.h + 2 * ph - kh + 1;
    const ow = x.w + 2 * pw - kw + 1;
    const plane = oh * ow;
    const inner = x.c * kh * kw;
    if (out.len < c_out * plane or workspace.len < inner or w.len != c_out * inner or x.data.len != x.c * x.h * x.w)
        return error.ShapeMismatch;
    const patches = workspace[0..inner];
    // Pack only eight spatial positions at a time, then reuse their patches
    // across every output channel. This removes padding branches from the
    // multiply-accumulate loop without allocating a full im2col matrix.
    // Each SIMD lane follows the original c/kh/kw accumulation order.
    const V = ConvVector;
    var at: usize = 0;
    while (at < plane) : (at += 8) {
        const lanes = @min(8, plane - at);
        for (0..x.c) |c| {
            for (0..kh) |dy| {
                for (0..kw) |dx| {
                    var values = [_]f32{0} ** 8;
                    for (0..lanes) |lane| {
                        const iy = (at + lane) / ow + dy;
                        const ix = (at + lane) % ow + dx;
                        if (iy >= ph and iy - ph < x.h and ix >= pw and ix - pw < x.w)
                            values[lane] = x.data[(c * x.h + iy - ph) * x.w + ix - pw];
                    }
                    patches[(c * kh + dy) * kw + dx] = values;
                }
            }
        }
        for (0..c_out) |o| {
            var acc: V = @splat(0);
            for (patches, w[o * inner ..][0..inner]) |patch, weight| acc += patch * @as(V, @splat(weight));
            const values: [8]f32 = acc;
            @memcpy(out[o * plane + at ..][0..lanes], values[0..lanes]);
        }
    }
    return .{ .data = out[0 .. c_out * plane], .c = c_out, .h = oh, .w = ow };
}

/// MaxPool2d with kernel == stride, floor mode (PyTorch default).
pub fn maxPool2d(alloc: std.mem.Allocator, x: Map, kh: usize, kw: usize) !Map {
    const oh = x.h / kh;
    const ow = x.w / kw;
    const out = try alloc.alloc(f32, x.c * oh * ow);
    errdefer alloc.free(out);
    return maxPool2dInto(out, x.readOnly(), kh, kw);
}

/// Caller-owned output, which must not overlap the input.
pub fn maxPool2dInto(out: []f32, x: ConstMap, kh: usize, kw: usize) !Map {
    if (kh == 0 or kw == 0) return error.InputTooShort;
    const oh = x.h / kh;
    const ow = x.w / kw;
    if (out.len < x.c * oh * ow or x.data.len != x.c * x.h * x.w) return error.ShapeMismatch;
    for (0..x.c) |c| {
        for (0..oh) |y| {
            for (0..ow) |xx| {
                var best: f32 = -std.math.inf(f32);
                for (0..kh) |dy| {
                    for (0..kw) |dx| best = @max(best, x.data[(c * x.h + y * kh + dy) * x.w + xx * kw + dx]);
                }
                out[(c * oh + y) * ow + xx] = best;
            }
        }
    }
    return .{ .data = out[0 .. x.c * oh * ow], .c = x.c, .h = oh, .w = ow };
}

const testing = std.testing;

test "dot handles vector and tail lanes" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try testing.expectApproxEqAbs(@as(f32, 385), dot(&a, &a), 1e-4);
}

test "conv1d with stride and padding" {
    // One channel, kernel [1, 1, 1], stride 2, pad 1 over [1, 2, 3, 4].
    var out: [2]f32 = undefined;
    conv1d(&out, &.{ 1, 2, 3, 4 }, 1, 4, &.{ 1, 1, 1 }, null, 1, 3, 2, 1);
    try testing.expectEqualSlices(f32, &.{ 3, 9 }, &out);
}

test "convTranspose1d overlaps and adds" {
    var out: [5]f32 = undefined;
    convTranspose1d(&out, &.{ 1, 2 }, 1, 2, &.{ 1, 1, 1 }, &.{0.5}, 1, 3, 2);
    try testing.expectEqualSlices(f32, &.{ 1.5, 1.5, 3.5, 2.5, 2.5 }, &out);
}

test "softmax is normalized and stable" {
    var xs = [_]f32{ 1000, 1000, -std.math.inf(f32) };
    softmax(&xs);
    try testing.expectApproxEqAbs(@as(f32, 0.5), xs[0], 1e-6);
    try testing.expectEqual(@as(f32, 0), xs[2]);
}

test "gru with zero weights halves the state" {
    // r = z = sigmoid(0) = 0.5, n = tanh(0) = 0, so h' = 0.5 * h.
    const zeros = [_]f32{0} ** 12;
    const gru = Gru{ .input = 2, .hidden = 2, .w_ih = &zeros, .w_hh = &zeros, .b_ih = zeros[0..6], .b_hh = zeros[0..6] };
    var h = [_]f32{ 2, -4 };
    var scratch: [12]f32 = undefined;
    gru.step(&.{ 1, 1 }, &h, &scratch);
    try testing.expectEqualSlices(f32, &.{ 1, -2 }, &h);
}

test "conv2d and maxPool2d shapes" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var data = [_]f32{1} ** (4 * 6);
    const x = Map{ .data = &data, .c = 1, .h = 4, .w = 6 };
    const y = try conv2d(arena, x, &([_]f32{1} ** 9), 1, 3, 3, 0, 1);
    try testing.expectEqual(@as(usize, 2), y.h);
    try testing.expectEqual(@as(usize, 6), y.w);
    try testing.expectEqual(@as(f32, 6), y.data[0]); // edge column sees 2 of 3 columns
    try testing.expectEqual(@as(f32, 9), y.data[1]);
    const p = try maxPool2d(arena, y, 2, 2);
    try testing.expectEqual(@as(usize, 3), p.w);
    try testing.expectEqual(@as(f32, 9), p.data[0]);
}
