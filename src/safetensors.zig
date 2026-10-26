//! Minimal safetensors reader.
//!
//! Format: u64 little-endian header length, a JSON header mapping tensor names
//! to {dtype, shape, data_offsets}, then the raw tensor bytes. Float tensors
//! (F32, F64, F16, BF16) are copied into aligned f32 slices; integer tensors
//! such as BatchNorm's num_batches_tracked are skipped.
//!
//! Everything is allocated from the arena passed to parse(); free the arena to
//! release the file.

const std = @import("std");

// warn, not err: the test runner fails any test that logs at error level.
const log = std.log.scoped(.safetensors);

/// A float tensor decoded to f32, row-major.
pub const Tensor = struct {
    shape: []const usize,
    data: []const f32,
};

const Entry = struct {
    name: []const u8,
    tensor: Tensor,
};

/// The float tensors of one safetensors file.
pub const File = struct {
    entries: []const Entry,
    /// The optional "__metadata__" object (string -> string), or .null.
    metadata: std.json.Value,

    /// Parse `bytes`, decoding every float tensor into `arena`. Returns
    /// error.InvalidSafetensors for malformed headers or inconsistent sizes.
    pub fn parse(arena: std.mem.Allocator, bytes: []const u8) !File {
        if (bytes.len < 8) return error.InvalidSafetensors;
        const header_len = std.mem.readInt(u64, bytes[0..8], .little);
        if (header_len > bytes.len - 8) return error.InvalidSafetensors;
        const header_end: usize = 8 + @as(usize, @intCast(header_len));
        const data = bytes[header_end..];

        const root = std.json.parseFromSliceLeaky(std.json.Value, arena, bytes[8..header_end], .{}) catch
            return error.InvalidSafetensors;
        if (root != .object) return error.InvalidSafetensors;

        var entries: std.ArrayList(Entry) = .empty;
        var metadata: std.json.Value = .null;
        var it = root.object.iterator();
        while (it.next()) |kv| {
            const name = kv.key_ptr.*;
            const info = kv.value_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) {
                metadata = info;
                continue;
            }
            if (info != .object) return error.InvalidSafetensors;

            const dtype = try field(info, "dtype", .string);
            const shape_json = try field(info, "shape", .array);
            const offsets = try field(info, "data_offsets", .array);
            if (offsets.array.items.len != 2) return error.InvalidSafetensors;

            const shape = try arena.alloc(usize, shape_json.array.items.len);
            var numel: usize = 1;
            for (shape_json.array.items, shape) |dim, *out| {
                out.* = try toUsize(dim);
                numel = std.math.mul(usize, numel, out.*) catch return error.InvalidSafetensors;
            }

            const begin = try toUsize(offsets.array.items[0]);
            const end = try toUsize(offsets.array.items[1]);
            if (end < begin or end > data.len) return error.InvalidSafetensors;

            const values = (try decode(arena, dtype.string, data[begin..end], numel)) orelse continue;
            try entries.append(arena, .{ .name = name, .tensor = .{ .shape = shape, .data = values } });
        }
        return .{ .entries = entries.items, .metadata = metadata };
    }

    /// The tensor called `name`, or error.MissingTensor.
    pub fn get(self: File, name: []const u8) !Tensor {
        for (self.entries) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry.tensor;
        }
        log.warn("missing tensor '{s}'", .{name});
        return error.MissingTensor;
    }

    /// The tensor's data after checking that its shape is exactly `shape`.
    pub fn expect(self: File, name: []const u8, shape: []const usize) ![]const f32 {
        const tensor = try self.get(name);
        if (!std.mem.eql(usize, tensor.shape, shape)) {
            log.warn("tensor '{s}' has shape {any}, expected {any}", .{ name, tensor.shape, shape });
            return error.ShapeMismatch;
        }
        return tensor.data;
    }

    /// A string value from the "__metadata__" object, if present.
    pub fn metadataString(self: File, key: []const u8) ?[]const u8 {
        if (self.metadata != .object) return null;
        const value = self.metadata.object.get(key) orelse return null;
        return if (value == .string) value.string else null;
    }
};

fn field(object: std.json.Value, key: []const u8, comptime tag: std.meta.Tag(std.json.Value)) !std.json.Value {
    const value = object.object.get(key) orelse return error.InvalidSafetensors;
    if (value != tag) return error.InvalidSafetensors;
    return value;
}

fn toUsize(value: std.json.Value) !usize {
    if (value != .integer) return error.InvalidSafetensors;
    return std.math.cast(usize, value.integer) orelse error.InvalidSafetensors;
}

/// Returns null for non-float dtypes, which callers skip.
fn decode(arena: std.mem.Allocator, dtype: []const u8, raw: []const u8, numel: usize) !?[]f32 {
    const Kind = enum { F32, F64, F16, BF16 };
    const kind = std.meta.stringToEnum(Kind, dtype) orelse return null;
    const width: usize = switch (kind) {
        .F32 => 4,
        .F64 => 8,
        .F16, .BF16 => 2,
    };

    const len = std.math.mul(usize, numel, width) catch return error.InvalidSafetensors;
    if (raw.len != len) return error.InvalidSafetensors;
    const out = try arena.alloc(f32, numel);
    for (out, 0..) |*value, i| {
        const at = i * width;
        value.* = switch (kind) {
            .F32 => @bitCast(std.mem.readInt(u32, raw[at..][0..4], .little)),
            .F64 => @floatCast(@as(f64, @bitCast(std.mem.readInt(u64, raw[at..][0..8], .little)))),
            .F16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[at..][0..2], .little)))),
            .BF16 => @bitCast(@as(u32, std.mem.readInt(u16, raw[at..][0..2], .little)) << 16),
        };
    }
    return out;
}

test "parse a small safetensors file" {
    std.testing.log_level = .err; // the error cases below log expected warnings
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const header =
        \\{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
        \\ "n":{"dtype":"I64","shape":[],"data_offsets":[8,16]},
        \\ "__metadata__":{"keyword":"hey computer"}}
    ;
    var bytes: [8 + header.len + 16]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    const data = bytes[8 + header.len ..];
    std.mem.writeInt(u32, data[0..4], @bitCast(@as(f32, 1.5)), .little);
    std.mem.writeInt(u32, data[4..8], @bitCast(@as(f32, -2.0)), .little);
    std.mem.writeInt(u64, data[8..16], 7, .little);

    const file = try File.parse(arena, &bytes);
    const w = try file.expect("w", &.{2});
    try std.testing.expectEqualSlices(f32, &.{ 1.5, -2.0 }, w);
    try std.testing.expectError(error.MissingTensor, file.get("n"));
    try std.testing.expectError(error.ShapeMismatch, file.expect("w", &.{ 1, 2 }));
    try std.testing.expectEqualStrings("hey computer", file.metadataString("keyword").?);
}

test "reject shapes whose element count overflows" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    inline for (.{
        \\{"w":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,0]}}
        ,
        \\{"w":{"dtype":"F32","shape":[1073741824],"data_offsets":[0,0]}}
        ,
        \\{"w":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}
        ,
    }) |header| {
        var bytes: [8 + header.len]u8 = undefined;
        std.mem.writeInt(u64, bytes[0..8], header.len, .little);
        @memcpy(bytes[8..], header);
        try std.testing.expectError(error.InvalidSafetensors, File.parse(arena, &bytes));
    }
}
