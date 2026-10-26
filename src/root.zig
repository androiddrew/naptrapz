//! naptrap: text-defined wake word detection (PhonMatchNet) in Zig.
//!
//! Port of https://github.com/androiddrew/naptrap. Start with `Model`:
//!
//!     var model = try naptrap.Model.load(io, gpa, "models/phonmatchnet", "models/google", "models/g2p");
//!     defer model.deinit();
//!     const kw = try model.keyword(gpa, "hey computer");
//!     defer kw.deinit(gpa);
//!     const prediction = try model.detect(scratch_arena, samples, kw, 0.01);
//!
//! `Stream` runs `detect` over a continuous audio stream with debouncing.
//!
//! `Model`, `Keyword`, `Prediction`, `Stream` and the `model`, `stream`, `text`
//! and `wav` modules are the supported API. The remaining modules (`features`,
//! `google`, `phonmatchnet`, `ops`, `safetensors`) expose the individual
//! pipeline stages for parity testing and experiments; they may change in any
//! release.

const std = @import("std");

/// Loading, keyword encoding and single-clip detection.
pub const model = @import("model.zig");
/// Sliding-window detection over continuous audio.
pub const stream = @import("stream.zig");
/// Keyword normalization and g2p_en text features.
pub const text = @import("text.zig");
/// Mono 16 kHz 16-bit PCM WAV decoding.
pub const wav = @import("wav.zig");

/// Audio frontends: keyword log-mel and Google log-mel (unstable).
pub const features = @import("features.zig");
/// Google speech_embedding encoder stage (unstable).
pub const google = @import("google.zig");
/// PhonMatchNet matcher stage (unstable).
pub const phonmatchnet = @import("phonmatchnet.zig");
/// fp32 kernels (unstable).
pub const ops = @import("ops.zig");
/// Minimal safetensors reader (unstable).
pub const safetensors = @import("safetensors.zig");

pub const Model = model.Model;
pub const Keyword = model.Keyword;
pub const Prediction = model.Prediction;
pub const Stream = stream.Stream;

test {
    std.testing.refAllDecls(@This());
}
