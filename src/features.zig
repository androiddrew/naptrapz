//! Audio frontends. Both use a 400-sample window, 160-sample hop and a
//! 512-point real DFT (257 bins), but differ in framing and scaling:
//!
//!   keywordMel  torch.stft(center=True, pad_mode="reflect"), window centered
//!               in the 512-point frame, power spectrum, 40 mel bins, log10(x + 1e-7)
//!   googleLogMel  TensorFlow-style: left-pad 11,120 zeros, frames start at
//!               sample 0, window at the start of the frame, magnitude
//!               spectrum, 32 mel bins, ln(x + 0.001)
//!
//! The DFT is a direct matrix product with precomputed tables. That is
//! ~50 MFLOP per 2 s clip; swap in a radix-2 FFT later if it matters.

const std = @import("std");
const ops = @import("ops.zig");

pub const sample_rate = 16000;
pub const n_fft = 512;
pub const bins = n_fft / 2 + 1;
pub const hop_length = 160;
pub const win_length = 400;
pub const keyword_mels = 40;
pub const google_mels = 32;
pub const google_left_pad = 11120;

pub const Dft = struct {
    cos: []f32, // [bins][n_fft]
    sin: []f32, // [bins][n_fft]

    pub fn init(alloc: std.mem.Allocator) !Dft {
        const cos = try alloc.alloc(f32, bins * n_fft);
        const sin = try alloc.alloc(f32, bins * n_fft);
        for (0..bins) |k| {
            for (0..n_fft) |n| {
                // Reduce k*n mod N exactly before converting, for accuracy.
                const phase = @as(f64, @floatFromInt((k * n) % n_fft)) * (2.0 * std.math.pi / @as(f64, n_fft));
                cos[k * n_fft + n] = @floatCast(@cos(phase));
                sin[k * n_fft + n] = @floatCast(@sin(phase));
            }
        }
        return .{ .cos = cos, .sin = sin };
    }

    /// |X[k]|^2 for k in 0..bins.
    pub fn power(self: Dft, frame: *const [n_fft]f32, out: *[bins]f32) void {
        for (0..bins) |k| {
            const re = ops.dot(frame, self.cos[k * n_fft ..][0..n_fft]);
            const im = ops.dot(frame, self.sin[k * n_fft ..][0..n_fft]);
            out[k] = re * re + im * im;
        }
    }
};

fn reflect(pos: isize, len: usize) usize {
    const n: isize = @intCast(len);
    const p = if (pos < 0) -pos else if (pos >= n) 2 * (n - 1) - pos else pos;
    return @intCast(p);
}

/// PhonMatchNet's 40-bin keyword log-mel: [ceil(len / 160)][40].
/// `window` is [400], `mel` is [257][40], both from the PhonMatchNet weights.
pub fn keywordMel(alloc: std.mem.Allocator, dft: Dft, wave: []const f32, window: []const f32, mel: []const f32) ![]f32 {
    std.debug.assert(window.len == win_length and mel.len == bins * keyword_mels);
    if (wave.len <= n_fft / 2) return error.ClipTooShort; // reflect padding needs this
    const frames = (wave.len + hop_length - 1) / hop_length;
    const out = try alloc.alloc(f32, frames * keyword_mels);
    const offset = (n_fft - win_length) / 2; // torch centers a short window
    var frame: [n_fft]f32 = undefined;
    var spectrum: [bins]f32 = undefined;
    for (0..frames) |t| {
        for (&frame, 0..) |*x, j| {
            if (j < offset or j >= offset + win_length) {
                x.* = 0;
                continue;
            }
            const pos = @as(isize, @intCast(t * hop_length + j)) - n_fft / 2;
            x.* = wave[reflect(pos, wave.len)] * window[j - offset];
        }
        dft.power(&frame, &spectrum);
        for (0..keyword_mels) |m| {
            var sum: f32 = 0;
            for (0..bins) |k| sum += spectrum[k] * mel[k * keyword_mels + m];
            out[t * keyword_mels + m] = @log10(sum + 1e-7);
        }
    }
    return out;
}

pub fn googleFrames(wave_len: usize) usize {
    return (wave_len + google_left_pad - win_length) / hop_length + 1;
}

/// Google speech_embedding's 32-bin log-mel: [googleFrames(len)][32].
/// `window` is [400], `mel` is [257][32], both from the Google weights.
pub fn googleLogMel(alloc: std.mem.Allocator, dft: Dft, wave: []const f32, window: []const f32, mel: []const f32) ![]f32 {
    std.debug.assert(window.len == win_length and mel.len == bins * google_mels);
    const frames = googleFrames(wave.len);
    const out = try alloc.alloc(f32, frames * google_mels);
    var frame: [n_fft]f32 = undefined;
    var spectrum: [bins]f32 = undefined;
    for (0..frames) |f| {
        for (&frame, 0..) |*x, j| {
            const pos = f * hop_length + j;
            x.* = if (j >= win_length or pos < google_left_pad) 0 else wave[pos - google_left_pad] * window[j];
        }
        dft.power(&frame, &spectrum);
        for (0..google_mels) |m| {
            var sum: f32 = 0;
            for (0..bins) |k| sum += @sqrt(spectrum[k]) * mel[k * google_mels + m];
            out[f * google_mels + m] = @log(sum + 0.001);
        }
    }
    return out;
}

const testing = std.testing;

test "dft of a constant frame is all DC" {
    const dft = try Dft.init(testing.allocator);
    defer testing.allocator.free(dft.cos);
    defer testing.allocator.free(dft.sin);
    const frame = [_]f32{1} ** n_fft;
    var out: [bins]f32 = undefined;
    dft.power(&frame, &out);
    try testing.expectApproxEqRel(@as(f32, n_fft * n_fft), out[0], 1e-5);
    try testing.expect(out[1] < 1e-3);
}

test "reflect padding mirrors without repeating the edge" {
    try testing.expectEqual(@as(usize, 2), reflect(-2, 10));
    try testing.expectEqual(@as(usize, 7), reflect(11, 10));
    try testing.expectEqual(@as(usize, 4), reflect(4, 10));
}

test "frame counts match the Python shapes" {
    // 1.5 s clip: Python gives keyword_mel [150, 40]; Google log-mel has 218
    // frames, which the conv stack reduces to 24000 / 1280 = 18 embeddings.
    try testing.expectEqual(@as(usize, 150), (24000 + hop_length - 1) / hop_length);
    try testing.expectEqual(@as(usize, 218), googleFrames(24000));
}
