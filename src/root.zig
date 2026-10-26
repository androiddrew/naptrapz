//! naptrap: text-defined wake word detection (PhonMatchNet) in Zig.
//!
//! Port of https://github.com/androiddrew/naptrap. Stages:
//!   features.zig     40-bin keyword log-mel and Google's 32-bin log-mel (STFT)
//!   google.zig       Google speech_embedding conv stack -> [G, 96]
//!   text.zig         keyword normalization and the g2p_en decoder states -> [P, 256]
//!   phonmatchnet.zig the PhonMatchNet matcher -> keyword logit
//!   model.zig        loads the three artifact directories and runs detect()

const std = @import("std");

pub const safetensors = @import("safetensors.zig");
pub const ops = @import("ops.zig");
pub const features = @import("features.zig");
pub const text = @import("text.zig");
pub const google = @import("google.zig");
pub const phonmatchnet = @import("phonmatchnet.zig");
pub const wav = @import("wav.zig");
pub const model = @import("model.zig");
pub const stream = @import("stream.zig");
pub const Stream = stream.Stream;

pub const Model = model.Model;
pub const Keyword = model.Keyword;
pub const Prediction = model.Prediction;

test {
    std.testing.refAllDecls(@This());
}
