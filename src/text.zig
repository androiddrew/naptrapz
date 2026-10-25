//! Keyword text features: normalization plus the g2p_en neural decoder.
//!
//! PhonMatchNet consumes g2p_en decoder hidden states, not phoneme strings.
//! Each word is run through the grapheme GRU encoder, then decoded greedily;
//! every emitted decoder state becomes one 256-D row, and words are separated
//! by the decoder's padding embedding (row 0 of dec_emb).

const std = @import("std");
const ops = @import("ops.zig");
const st = @import("safetensors.zig");

/// Index 0 <pad>, 1 <unk>, 2 </s>, then a..z.
const grapheme_count = 29;
const grapheme_end = 2;
const phoneme_start = 2;
const phoneme_end = 3;
const max_decode_steps = 20;

// ---------------------------------------------------------------------------
// Normalization
//
// Python: replace U+2019 with ', NFKD, drop combining marks, lowercase, then
// take tokens matching [a-z]+(?:'[a-z]+)*|[0-9]+ and reject any digit token.
//
// Zig's standard library has no Unicode normalization, so fold() covers
// ASCII, Latin-1 and Latin Extended-A, which is where NFKD matters for
// English keywords. Other characters act as word separators. Extend the
// tables if you need fullwidth letters or ligatures.
// ---------------------------------------------------------------------------

const ascii_lower: [128]u8 = blk: {
    var table: [128]u8 = undefined;
    for (&table, 0..) |*c, i| c.* = std.ascii.toLower(@intCast(i));
    break :blk table;
};

/// U+00C0..U+00FF after NFKD and combining-mark removal. " " = not a letter.
const latin1 = [64][]const u8{
    "a", "a", "a", "a", "a", "a", " ", "c", "e", "e", "e", "e", "i", "i", "i", "i", // C0
    " ", "n", "o", "o", "o", "o", "o", " ", " ", "u", "u", "u", "u", "y", " ", " ", // D0
    "a", "a", "a", "a", "a", "a", " ", "c", "e", "e", "e", "e", "i", "i", "i", "i", // E0
    " ", "n", "o", "o", "o", "o", "o", " ", " ", "u", "u", "u", "u", "y", " ", "y", // F0
};

/// U+0100..U+017F (Latin Extended-A) after NFKD and combining-mark removal.
const latin_ext_a = [128][]const u8{
    "a",  "a", "a", "a", "a",  "a",  "c",  "c",  "c", "c", "c", "c", "c", "c", "d", "d", // 100
    " ",  " ", "e", "e", "e",  "e",  "e",  "e",  "e", "e", "e", "e", "g", "g", "g", "g", // 110
    "g",  "g", "g", "g", "h",  "h",  " ",  " ",  "i", "i", "i", "i", "i", "i", "i", "i", // 120
    "i",  " ", "ij", "ij", "j", "j", "k", "k", " ", "l", "l", "l", "l", "l", "l", "l ", // 130
    "l ", " ", " ", "n", "n",  "n",  "n",  "n",  "n", " n", " ", " ", "o", "o", "o", "o", // 140
    "o",  "o", " ", " ", "r",  "r",  "r",  "r",  "r", "r", "s", "s", "s", "s", "s", "s", // 150
    "s",  "s", "t", "t", "t",  "t",  " ",  " ",  "u", "u", "u", "u", "u", "u", "u", "u", // 160
    "u",  "u", "u", "u", "w",  "w",  "y",  "y",  "y", "z", "z", "z", "z", "z", "z", "s", // 170
};

fn fold(cp: u21) []const u8 {
    return switch (cp) {
        0...0x7f => ascii_lower[cp..][0..1],
        0xaa => "a", // ª
        0xba => "o", // º
        0xb2 => "2",
        0xb3 => "3",
        0xb9 => "1",
        0xc0...0xff => latin1[cp - 0xc0],
        0x100...0x17f => latin_ext_a[cp - 0x100],
        0x300...0x36f => "", // combining marks
        0x2019 => "'",
        else => " ",
    };
}

fn isLetter(c: u8) bool {
    return c >= 'a' and c <= 'z';
}

/// Lowercased words joined by single spaces, e.g. "Héllo, WORLD’s" -> "hello world's".
pub fn normalize(arena: std.mem.Allocator, keyword: []const u8) ![]const u8 {
    const view = std.unicode.Utf8View.init(keyword) catch return error.InvalidUtf8;
    var folded: std.ArrayList(u8) = .empty;
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| try folded.appendSlice(arena, fold(cp));

    const s = folded.items;
    var out: std.ArrayList(u8) = .empty;
    var has_digit = false;
    var i: usize = 0;
    while (i < s.len) {
        if (isLetter(s[i])) {
            const start = i;
            while (i < s.len and isLetter(s[i])) i += 1;
            while (i + 1 < s.len and s[i] == '\'' and isLetter(s[i + 1])) {
                i += 1;
                while (i < s.len and isLetter(s[i])) i += 1;
            }
            if (out.items.len > 0) try out.append(arena, ' ');
            try out.appendSlice(arena, s[start..i]);
        } else if (std.ascii.isDigit(s[i])) {
            has_digit = true;
            while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        } else {
            i += 1;
        }
    }
    if (has_digit) return error.SpellOutNumbers;
    if (out.items.len == 0) return error.NoLetters;
    return out.items;
}

// ---------------------------------------------------------------------------
// G2P decoder
// ---------------------------------------------------------------------------

pub const G2p = struct {
    size: usize,
    phonemes: usize,
    enc_emb: []const f32, // [29][size]
    dec_emb: []const f32, // [phonemes][size]
    fc_w: []const f32, // [phonemes][size]
    fc_b: []const f32, // [phonemes]
    encoder: ops.Gru,
    decoder: ops.Gru,

    pub fn init(file: st.File) !G2p {
        const enc = try file.get("enc_emb");
        const dec = try file.get("dec_emb");
        if (enc.shape.len != 2 or enc.shape[0] != grapheme_count or dec.shape.len != 2)
            return error.ShapeMismatch;
        const size = enc.shape[1];
        const phonemes = dec.shape[0];
        return .{
            .size = size,
            .phonemes = phonemes,
            .enc_emb = enc.data,
            .dec_emb = try file.expect("dec_emb", &.{ phonemes, size }),
            .fc_w = try file.expect("fc_w", &.{ phonemes, size }),
            .fc_b = try file.expect("fc_b", &.{phonemes}),
            .encoder = try loadCell(file, "encoder", size),
            .decoder = try loadCell(file, "decoder", size),
        };
    }

    fn loadCell(file: st.File, comptime prefix: []const u8, size: usize) !ops.Gru {
        return .{
            .input = size,
            .hidden = size,
            .w_ih = try file.expect(prefix ++ ".w_ih", &.{ 3 * size, size }),
            .w_hh = try file.expect(prefix ++ ".w_hh", &.{ 3 * size, size }),
            .b_ih = try file.expect(prefix ++ ".b_ih", &.{3 * size}),
            .b_hh = try file.expect(prefix ++ ".b_hh", &.{3 * size}),
        };
    }

    pub const Features = struct {
        /// Normalized keyword text.
        text: []const u8,
        /// [rows][size]
        data: []const f32,
        rows: usize,
    };

    pub fn encode(self: G2p, arena: std.mem.Allocator, keyword: []const u8) !Features {
        const normalized = try normalize(arena, keyword);
        var rows: std.ArrayList(f32) = .empty;
        var words = std.mem.tokenizeScalar(u8, normalized, ' ');
        var first = true;
        while (words.next()) |w| {
            if (!first) try rows.appendSlice(arena, self.dec_emb[0..self.size]);
            first = false;
            try self.encodeWord(arena, w, &rows);
        }
        if (rows.items.len == 0) return error.NoPhoneticFeatures;
        return .{ .text = normalized, .data = rows.items, .rows = rows.items.len / self.size };
    }

    fn encodeWord(self: G2p, arena: std.mem.Allocator, w: []const u8, rows: *std.ArrayList(f32)) !void {
        const size = self.size;
        const h = try arena.alloc(f32, size);
        @memset(h, 0);
        const scratch = try arena.alloc(f32, 6 * size);
        const logits = try arena.alloc(f32, self.phonemes);

        for (w) |c| {
            const index: usize = if (isLetter(c)) 3 + @as(usize, c - 'a') else 1; // apostrophe -> <unk>
            self.encoder.step(self.enc_emb[index * size ..][0..size], h, scratch);
        }
        self.encoder.step(self.enc_emb[grapheme_end * size ..][0..size], h, scratch);

        var token: usize = phoneme_start;
        for (0..max_decode_steps) |_| {
            self.decoder.step(self.dec_emb[token * size ..][0..size], h, scratch);
            ops.linear(logits, h, 1, self.fc_w, self.fc_b, size, self.phonemes);
            token = ops.argmax(logits);
            if (token == phoneme_end) break;
            try rows.appendSlice(arena, h);
        }
    }
};

const testing = std.testing;

test "normalize matches the Python policy" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    try testing.expectEqualStrings("hello world's", try normalize(arena, "Héllo,  WORLD\u{2019}s!"));
    try testing.expectEqualStrings("hey computer", try normalize(arena, "  hey-computer "));
    try testing.expectEqualStrings("cafe", try normalize(arena, "cafe\u{301}"));
    try testing.expectEqualStrings("a b", try normalize(arena, "a' b"));
    try testing.expectError(error.SpellOutNumbers, normalize(arena, "route 66"));
    try testing.expectError(error.NoLetters, normalize(arena, "!!! ß"));
}
