//! Default build: no model bytes included. Build.zig replaces this module when
//! -Dembed-models=DIR is supplied; core naptrap imports need no build options.
const std = @import("std");
const naptrap = @import("naptrap");
/// Whether this build embeds model files.
pub const available = false;
/// Always error.ModelsNotEmbedded in this build.
pub fn load(_: std.mem.Allocator) !naptrap.Model {
    return error.ModelsNotEmbedded;
}
