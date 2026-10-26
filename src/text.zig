//! Keyword text features: normalization plus the g2p_en neural decoder.
//!
//! PhonMatchNet consumes g2p_en decoder hidden states, not phoneme strings.
//! Each word is run through the grapheme GRU encoder, then decoded greedily;
//! every emitted decoder state becomes one 256-D row, and words are separated
//! by the decoder's padding embedding (row 0 of dec_emb).

const std = @import("std");
const ops = @import("ops.zig");
const st = @import("safetensors.zig");
const unicode_data = @import("unicode_data.zig");
pub const unicode_version = unicode_data.unicode_version;

/// Index 0 <pad>, 1 <unk>, 2 </s>, then a..z.
const grapheme_count = 29;
const grapheme_end = 2;
const phoneme_start = 2;
const phoneme_end = 3;
const max_decode_steps = 20;

/// The matcher accepts at most this many keyword feature rows.
pub const max_rows = 128;
/// Longer keyword strings are rejected before normalization. No accepted
/// keyword comes close: 128 rows is roughly twenty words.
pub const max_keyword_bytes = 1024;

// ---------------------------------------------------------------------------
// Normalization
//
// Python: replace U+2019 with ', NFKD, drop combining marks, lowercase, then
// take tokens matching [a-z]+(?:'[a-z]+)*|[0-9]+ and reject any digit token.
//
// Generated Unicode data projects this policy onto the ASCII token alphabet.
// NFKD reordering only moves combining marks, which are removed. Unicode's
// contextual lowercase rule (final sigma) cannot produce ASCII tokens, so
// per-codepoint projection has the same result as whole-string normalization.
// ---------------------------------------------------------------------------

const ascii_lower: [128]u8 = blk: {
    var table: [128]u8 = undefined;
    for (&table, 0..) |*c, i| c.* = std.ascii.toLower(@intCast(i));
    break :blk table;
};

fn fold(cp: u21) []const u8 {
    if (cp < 128) return ascii_lower[cp..][0..1];
    var low: usize = 0;
    var high: usize = unicode_data.rules.len;
    while (low < high) {
        const mid = low + (high - low) / 2;
        const rule = unicode_data.rules[mid];
        if (cp < rule.first) {
            high = mid;
        } else if (cp > rule.last) {
            low = mid + 1;
        } else {
            if (rule.increment) {
                const index = rule.value[0] + cp - rule.first;
                return ascii_lower[index..][0..1];
            }
            return rule.value;
        }
    }
    return " ";
}

fn isLetter(c: u8) bool {
    return c >= 'a' and c <= 'z';
}

/// Lowercased words joined by single spaces, e.g. "Héllo, WORLD’s" -> "hello world's".
/// Input longer than `max_keyword_bytes` returns error.KeywordTooLong.
pub fn normalize(arena: std.mem.Allocator, keyword: []const u8) ![]const u8 {
    if (keyword.len > max_keyword_bytes) return error.KeywordTooLong;
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

    /// Borrow the g2p weights from `file`. Embedding size and phoneme count
    /// come from the tensor shapes; every other tensor must match them.
    pub fn init(file: st.File) !G2p {
        const enc = try file.get("enc_emb");
        const dec = try file.get("dec_emb");
        if (enc.shape.len != 2 or enc.shape[0] != grapheme_count or dec.shape.len != 2)
            return error.ShapeMismatch;
        const size = enc.shape[1];
        const phonemes = dec.shape[0];
        // Decoding indexes the <s> and </s> phoneme rows (2 and 3).
        if (size == 0 or phonemes <= phoneme_end) return error.ShapeMismatch;
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

    /// A keyword's text features. Owns its memory; see `deinit`.
    pub const Features = struct {
        /// Normalized keyword text.
        text: []const u8,
        /// [rows][size]
        data: []const f32,
        rows: usize,

        /// Free memory returned by `G2p.encode` (or `Model.keyword`).
        pub fn deinit(self: Features, gpa: std.mem.Allocator) void {
            gpa.free(self.text);
            gpa.free(self.data);
        }
    };

    /// Normalize `keyword` and compute its decoder-state rows, allocated from
    /// `gpa`. Encoding stops with error.FeatureLengthOutOfRange as soon as it
    /// exceeds `max_rows`.
    pub fn encode(self: G2p, gpa: std.mem.Allocator, keyword: []const u8) !Features {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        const normalized = try normalize(arena, keyword);
        var rows: std.ArrayList(f32) = .empty;
        const work: Work = .{
            .h = try arena.alloc(f32, self.size),
            .scratch = try arena.alloc(f32, 6 * self.size),
            .logits = try arena.alloc(f32, self.phonemes),
        };
        var words = std.mem.tokenizeScalar(u8, normalized, ' ');
        var first = true;
        while (words.next()) |w| {
            if (!first) try self.appendRow(arena, &rows, self.dec_emb[0..self.size]);
            first = false;
            try self.encodeWord(arena, w, &rows, work);
        }
        if (rows.items.len == 0) return error.NoPhoneticFeatures;

        const text = try gpa.dupe(u8, normalized);
        errdefer gpa.free(text);
        return .{ .text = text, .data = try gpa.dupe(f32, rows.items), .rows = rows.items.len / self.size };
    }

    const Work = struct { h: []f32, scratch: []f32, logits: []f32 };

    fn appendRow(self: G2p, arena: std.mem.Allocator, rows: *std.ArrayList(f32), row: []const f32) !void {
        if (rows.items.len / self.size >= max_rows) return error.FeatureLengthOutOfRange;
        try rows.appendSlice(arena, row);
    }

    fn encodeWord(self: G2p, arena: std.mem.Allocator, w: []const u8, rows: *std.ArrayList(f32), work: Work) !void {
        const size = self.size;
        const h = work.h;
        @memset(h, 0);

        for (w) |c| {
            const index: usize = if (isLetter(c)) 3 + @as(usize, c - 'a') else 1; // apostrophe -> <unk>
            self.encoder.step(self.enc_emb[index * size ..][0..size], h, work.scratch);
        }
        self.encoder.step(self.enc_emb[grapheme_end * size ..][0..size], h, work.scratch);

        var token: usize = phoneme_start;
        for (0..max_decode_steps) |_| {
            self.decoder.step(self.dec_emb[token * size ..][0..size], h, work.scratch);
            ops.linear(work.logits, h, 1, self.fc_w, self.fc_b, size, self.phonemes);
            token = ops.argmax(work.logits);
            if (token == phoneme_end) break;
            try self.appendRow(arena, rows, h);
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
    try testing.expectError(error.KeywordTooLong, normalize(arena, "a" ** (max_keyword_bytes + 1)));
}
