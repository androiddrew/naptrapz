//! Audio frontends. Both use a 400-sample window, 160-sample hop and a
//! 512-point FFT (257 real-input bins), but differ in framing and scaling:
//!
//!   keywordMel  torch.stft(center=True, pad_mode="reflect"), window centered
//!               in the 512-point frame, power spectrum, 40 mel bins, log10(x + 1e-7)
//!   googleLogMel  TensorFlow-style: left-pad 11,120 zeros, frames start at
//!               sample 0, window at the start of the frame, magnitude
//!               spectrum, 32 mel bins, ln(x + 0.001)
//!
//! A radix-2 FFT uses f64 butterflies to avoid cancellation noise in nearly
//! empty spectral bins. Windowed inputs, powers and model features remain f32.

const std = @import("std");

pub const sample_rate = 16000;
pub const n_fft = 512;
pub const bins = n_fft / 2 + 1;
pub const hop_length = 160;
pub const win_length = 400;
pub const keyword_mels = 40;
pub const google_mels = 32;
pub const google_left_pad = 11120;

/// Radix-2 twiddle factors e^(-2*pi*i*k/n_fft) for k < n_fft / 2.
const twiddles = blk: {
    @setEvalBranchQuota(10_000);
    var cos: [n_fft / 2]f64 = undefined;
    var sin: [n_fft / 2]f64 = undefined;
    for (0..n_fft / 2) |k| {
        const phase = @as(f64, @floatFromInt(k)) * (2.0 * std.math.pi / @as(f64, n_fft));
        cos[k] = @cos(phase);
        sin[k] = -@sin(phase);
    }
    cos[n_fft / 4] = 0;
    break :blk .{ .cos = cos, .sin = sin };
};

/// |X[k]|^2 for k in 0..bins: the power spectrum of one real 512-sample frame.
pub fn powerSpectrum(frame: *const [n_fft]f32, out: *[bins]f32) void {
    var real: [n_fft]f64 = undefined;
    var imag = [_]f64{0} ** n_fft;
    for (frame, 0..) |sample, i| real[@bitReverse(@as(u9, @intCast(i)))] = sample;
    var size: usize = 2;
    while (size <= n_fft) : (size *= 2) {
        const half = size / 2;
        const stride = n_fft / size;
        var start: usize = 0;
        while (start < n_fft) : (start += size) {
            for (0..half) |j| {
                const even = start + j;
                const odd = even + half;
                const c = twiddles.cos[j * stride];
                const s = twiddles.sin[j * stride];
                const re = c * real[odd] - s * imag[odd];
                const im = s * real[odd] + c * imag[odd];
                real[odd] = real[even] - re;
                imag[odd] = imag[even] - im;
                real[even] += re;
                imag[even] += im;
            }
        }
    }
    for (0..bins) |k| {
        out[k] = @floatCast(real[k] * real[k] + imag[k] * imag[k]);
    }
}

fn reflect(pos: isize, len: usize) usize {
    const n: isize = @intCast(len);
    const p = if (pos < 0) -pos else if (pos >= n) 2 * (n - 1) - pos else pos;
    return @intCast(p);
}

/// PhonMatchNet's 40-bin keyword log-mel: [ceil(len / 160)][40].
/// `window` is [400], `mel` is [257][40], both from the PhonMatchNet weights.
pub fn keywordMel(alloc: std.mem.Allocator, wave: []const f32, window: []const f32, mel: []const f32) ![]f32 {
    if (window.len != win_length or mel.len != bins * keyword_mels) return error.ShapeMismatch;
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
        powerSpectrum(&frame, &spectrum);
        const V = @Vector(keyword_mels, f32);
        var sums: V = @splat(0);
        for (spectrum, 0..) |power, k| {
            const weights: V = mel[k * keyword_mels ..][0..keyword_mels].*;
            sums += @as(V, @splat(power)) * weights;
        }
        const energies: [keyword_mels]f32 = sums;
        for (energies, 0..) |energy, m| out[t * keyword_mels + m] = @log10(energy + 1e-7);
    }
    return out;
}

pub fn googleFrames(wave_len: usize) usize {
    return (wave_len + google_left_pad - win_length) / hop_length + 1;
}

/// Google speech_embedding's 32-bin log-mel: [googleFrames(len)][32].
/// `window` is [400], `mel` is [257][32], both from the Google weights.
pub fn googleLogMel(alloc: std.mem.Allocator, wave: []const f32, window: []const f32, mel: []const f32) ![]f32 {
    if (window.len != win_length or mel.len != bins * google_mels) return error.ShapeMismatch;
    const frames = googleFrames(wave.len);
    const out = try alloc.alloc(f32, frames * google_mels);
    var frame: [n_fft]f32 = undefined;
    var spectrum: [bins]f32 = undefined;
    for (0..frames) |f| {
        for (&frame, 0..) |*x, j| {
            const pos = f * hop_length + j;
            x.* = if (j >= win_length or pos < google_left_pad) 0 else wave[pos - google_left_pad] * window[j];
        }
        powerSpectrum(&frame, &spectrum);
        const V = @Vector(google_mels, f32);
        var sums: V = @splat(0);
        for (spectrum, 0..) |power, k| {
            const weights: V = mel[k * google_mels ..][0..google_mels].*;
            sums += @as(V, @splat(@sqrt(power))) * weights;
        }
        const energies: [google_mels]f32 = sums;
        for (energies, 0..) |energy, m| out[f * google_mels + m] = @log(energy + 0.001);
    }
    return out;
}

const testing = std.testing;

test "power spectrum of a constant frame is all DC" {
    const frame = [_]f32{1} ** n_fft;
    var out: [bins]f32 = undefined;
    powerSpectrum(&frame, &out);
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
