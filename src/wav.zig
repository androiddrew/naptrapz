//! Minimal WAV reader: mono, 16-bit PCM, 16 kHz, like naptrap's load_wav().

const std = @import("std");
const features = @import("features.zig");

/// Samples as float32 in [-1, 1).
pub fn decode(alloc: std.mem.Allocator, bytes: []const u8) ![]f32 {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE"))
        return error.NotAWavFile;

    var format_ok = false;
    var pos: usize = 12;
    while (pos + 8 <= bytes.len) {
        const id = bytes[pos..][0..4];
        const size = std.mem.readInt(u32, bytes[pos + 4 ..][0..4], .little);
        const body_start = pos + 8;
        if (size > bytes.len - body_start) return error.TruncatedWav;
        const body = bytes[body_start..][0..size];

        if (std.mem.eql(u8, id, "fmt ")) {
            if (body.len < 16) return error.TruncatedWav;
            const audio_format = std.mem.readInt(u16, body[0..2], .little);
            const channels = std.mem.readInt(u16, body[2..4], .little);
            const rate = std.mem.readInt(u32, body[4..8], .little);
            const bits = std.mem.readInt(u16, body[14..16], .little);
            if (audio_format != 1 or channels != 1 or rate != features.sample_rate or bits != 16)
                return error.ExpectedMono16kHz16BitPcm;
            format_ok = true;
        } else if (std.mem.eql(u8, id, "data")) {
            if (!format_ok) return error.ExpectedMono16kHz16BitPcm;
            const out = try alloc.alloc(f32, body.len / 2);
            for (out, 0..) |*x, i| {
                const sample = std.mem.readInt(i16, body[i * 2 ..][0..2], .little);
                x.* = @as(f32, @floatFromInt(sample)) / 32768.0;
            }
            return out;
        }
        pos = body_start + size + (size & 1); // chunks are word aligned
    }
    return error.NoDataChunk;
}

test "decode a tiny wav" {
    const bytes = "RIFF" ++ "\x2c\x00\x00\x00" ++ "WAVE" ++
        "fmt " ++ "\x10\x00\x00\x00" ++ "\x01\x00" ++ "\x01\x00" ++ "\x80\x3e\x00\x00" ++
        "\x00\x7d\x00\x00" ++ "\x02\x00" ++ "\x10\x00" ++
        "data" ++ "\x04\x00\x00\x00" ++ "\x00\x40" ++ "\x00\xc0";
    const samples = try decode(std.testing.allocator, bytes);
    defer std.testing.allocator.free(samples);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, -0.5 }, samples);
}
