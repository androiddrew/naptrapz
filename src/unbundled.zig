//! Default build: no model bytes included. Build.zig replaces this module when
//! -Dembed-models=DIR is supplied; core naptrap imports need no build options.
const std = @import("std");
const naptrap = @import("naptrap");
pub const available = false;
pub fn load(_: std.mem.Allocator) !naptrap.Model {
    return error.ModelsNotEmbedded;
}
