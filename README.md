# naptrapz

A Zig 0.16 inference runtime for [naptrap](https://github.com/androiddrew/naptrap)'s
PhonMatchNet text-defined wake word model. It has no dependencies beyond the Zig
standard library, does fp32 batch-1 CPU inference, and reads the same three
Hugging Face artifact directories as the Python package. It ships as a Zig
package, a command-line tool and a C library.

Requires Zig **0.16.0** or newer. Validated on Linux/x86_64 against Python
**3.12.11** and PyTorch **2.14.1+cpu** with the released `v1` artifacts: on a
corpus of 25 speech clips, five keywords and both calibrated operating points
(maintained outside this package), all 250 decisions match Python, with maximum
score difference below `2e-6` in ReleaseFast.

## Get the models

```sh
hf download androiddrew/naptrap-phonmatchnet-en       --revision v1 --local-dir models/phonmatchnet
hf download androiddrew/google-speech-embedding-pytorch --revision v1 --local-dir models/google
hf download androiddrew/g2p-en-decoder-pytorch        --revision v1 --local-dir models/g2p
```

`models/` is the default model directory of the CLI and is ignored by git.

## Build and run

```sh
zig build -Doptimize=ReleaseFast
zig-out/bin/naptrap detect clip.wav "hey computer"
```

`detect` accepts mono, 16 kHz, 16-bit PCM WAVs of 0.5–2 seconds. It prints JSON
and exits with **0** for an accepted keyword, **1** for rejection, or **2** for an
input/model error (usage errors go to stderr). `--far` must exactly select a
calibrated operating point; the released model provides `0.01` (default) and
`0.1`. `--models DIR` points at a directory holding `phonmatchnet/`, `google/`
and `g2p/`; `--model-dir`, `--google-dir` and `--g2p-dir` set them separately.

Keywords are normalized like the Python package (see
[Implementation details](#implementation-details)), may be at most 1024 bytes,
and may produce at most 128 feature rows (roughly twenty words).

## Using naptrap from Zig

```sh
zig fetch --save git+https://github.com/androiddrew/naptrapz#v0.1.0
```

```zig
// build.zig
const naptrap = b.dependency("naptrapz", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("naptrap", naptrap.module("naptrap"));
```

The `naptrapz` package exports the `naptrap` module (and `naptrap_bundled` for
embedded models). The CLI, C library and header keep the `naptrap` name.

```zig
const naptrap = @import("naptrap");

var model = try naptrap.Model.load(io, gpa, "models/phonmatchnet", "models/google", "models/g2p");
defer model.deinit();
const keyword = try model.keyword(gpa, "hey computer");
defer keyword.deinit(gpa);

var scratch: std.heap.ArenaAllocator = .init(gpa);
defer scratch.deinit();
const prediction = try model.detect(scratch.allocator(), samples, keyword, 0.01);
if (prediction.accepted) std.debug.print("wake ({d:.3})\n", .{prediction.score});
```

`Model` and `Keyword` own their memory and are released with `deinit`.
`detect` takes a *scratch* allocator whose buffers it never frees individually:
pass an arena (or a `FixedBufferAllocator`) and reset it between calls with
`.retain_capacity`. A loaded `Model` is immutable, so `keyword`, `threshold`
and `detect` can run on several threads at once, each `detect` with its own
scratch allocator. Errors are explicit sets such as `naptrap.model.DetectError`.

`Model`, `Keyword`, `Prediction`, `Stream` and the `model`, `stream`, `text`
and `wav` modules are the supported API. `features`, `google`, `phonmatchnet`,
`ops` and `safetensors` expose individual pipeline stages for parity work and
may change in any release. `zig build docs` installs the API documentation to
`zig-out/docs`.

Load diagnostics are logged through `std.log` under the `.naptrap` and
`.safetensors` scopes; filter them with `std_options.log_scope_levels`.

To embed models in a Zig program, pass a `LazyPath` such as
`.@"embed-models" = b.path("models")` to `b.dependency` and import the
`naptrap_bundled` module (see
[Self-contained embedded-model builds](#self-contained-embedded-model-builds)).

## Streaming listener

`listen` reads **raw mono 16 kHz signed 16-bit little-endian PCM** from stdin
(no WAV header), keeps the model and keyword loaded, and emits one JSON line per
debounced wake event. For example, from a Linux microphone or an existing file:

```sh
arecord -q -t raw -f S16_LE -r 16000 -c 1 | zig-out/bin/naptrap listen "hey computer"
ffmpeg -v error -i recording.wav -f s16le -ac 1 -ar 16000 - | \
    zig-out/bin/naptrap listen "hey computer" --all-windows
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--window-ms` | 1500 | Window duration, 500–2000 ms |
| `--hop-ms` | 250 | Time between evaluations, positive and no larger than the window |
| `--trigger-windows` | 2 | Consecutive accepted windows required to trigger |
| `--release-windows` | 2 | Consecutive rejected windows required to rearm |
| `--cooldown-ms` | 1000 | Minimum audio time between wake events |
| `--far` | 0.01 | Exact calibrated per-clip operating point |
| `--all-windows` | off | Emit scores for every window, including rejected/suppressed windows |

Events contain `event` (`wake` or `window`), normalized `keyword`, `score`,
`keyword_logit`, `threshold`, `accepted`, `triggered`, `target_far`, and audio
`start_sample`/`end_sample` and seconds. End positions are exclusive. `accepted`
describes the model threshold; `triggered` additionally applies debouncing.
Timestamps count received samples, not wall-clock time. Continuous acceptance
produces only one wake until rejected windows rearm the detector. A successful
EOF exits 0, even without wakes; errors exit 2. Partial final windows are not
padded. An odd trailing PCM byte is an error. Output is flushed per event.

Zig applications can use `naptrap.Stream.init(gpa, &model, keyword, config)`
and `stream.push(samples, context, callback)` with arbitrary float32 chunks in
`[-1, 1]`. The synchronous callback receives every complete window and its
`triggered` flag; it may return `void` or an error union, and its errors are
passed through `push`. The model and keyword are borrowed and must outlive the
stream. Serialize calls; callbacks must not reenter it. `reset()` clears audio
history, timestamps and debounce state while retaining scratch capacity. Invalid
sample chunks leave state unchanged; reset after other push/callback failures.
Call `deinit()` when finished. Feed already resampled 16 kHz audio:
capture/resampling and VAD/STT/TTS remain the host application's
responsibilities. Repeated-window false-alarm rates need evaluation separately
from the per-clip FAR calibration.

## Embedding through the C ABI

The normal build also installs `lib/libnaptrap.so` and `lib/libnaptrap.a` (on macOS
`libnaptrap.dylib`; on Windows `bin/naptrap.dll` with import library
`lib/naptrap.lib` and static library `lib/naptrap_static.lib`; on wasm only the
static library), plus `include/naptrap.h`. ABI version **1** uses opaque handles,
fixed-width result fields, explicit status codes and thread-local error
messages. The shared library's soname major version tracks the ABI version
(`libnaptrap.so.1`). The library allocates with the host's `malloc` and never
writes to stdout or stderr; diagnostics are appended to `naptrap_last_error()`.
The static archive is position-independent and includes Zig's compiler runtime,
so a normal C compiler can link it into a PIE executable. Use a release build
(`-Doptimize=ReleaseFast` or `ReleaseSafe`) for libraries you distribute.

```sh
# Shared library (set an appropriate runtime search path for your installation):
cc app.c -I zig-out/include -L zig-out/lib -lnaptrap -o app
# Static library:
cc app.c -I zig-out/include zig-out/lib/libnaptrap.a -o app
```

Typical C use, with application-owned mono 16 kHz float samples:

```c
#include <naptrap.h>
#include <stdio.h>

int score(const float *samples, size_t count, naptrap_prediction *prediction) {
    naptrap_model *model = NULL;
    naptrap_keyword *keyword = NULL;
    naptrap_status status = naptrap_model_load("models/phonmatchnet", "models/google",
                                              "models/g2p", &model);
    if (status == NAPTRAP_OK)
        status = naptrap_keyword_create(model, "hey computer", 12, &keyword);
    if (status == NAPTRAP_OK)
        status = naptrap_detect(model, keyword, samples, count, NAPTRAP_SAMPLE_RATE,
                                0.01, prediction);
    if (status != NAPTRAP_OK) fprintf(stderr, "%s\n", naptrap_last_error());
    naptrap_keyword_release(keyword);
    naptrap_model_release(model);
    return status;
}
```

In a long-running application, load the model and keyword once and reuse them.
`naptrap_stream_create` / `naptrap_stream_push` expose the streaming detector with
a synchronous callback for every complete window. Inspect `event->triggered` for
wakes. `naptrap_stream_config_default` fills the standard settings; a NULL config
also selects defaults. `naptrap_stream_reset` starts a fresh audio timeline.

A keyword retains its model and a stream retains its keyword; you may release
parent references while dependent handles are alive. Release each returned
reference exactly once. NULL release calls succeed. Calls on handles sharing a
model are serialized: overlapping or reentrant calls, including releases
attempted from callbacks, fail with `NAPTRAP_BUSY`. Separate models run
concurrently, so load one model per thread for parallel detection. Do not retain
callback event pointers: copy the event instead. The full contracts and error
codes are in `include/naptrap.h`; `tests/abi_smoke.c` exercises them. Python
`ctypes`, Rust FFI and Swift's C interoperability can use this same interface.

## Self-contained embedded-model builds

Supply **`-Dembed-models=DIR`** to embed all three `config.json` and
`model.safetensors` pairs. The directory must contain `phonmatchnet/`, `google/`
and `g2p/`. On the command line, a relative path is resolved from the current
directory, like `-Dtest-models`.

```sh
zig build -Doptimize=ReleaseFast -Dembed-models=models --prefix embedded

# No model directory needed at runtime:
embedded/bin/naptrap detect clip.wav "hey computer"
arecord -q -t raw -f S16_LE -r 16000 -c 1 | embedded/bin/naptrap listen "hey computer"
```

The tested Linux CLI is statically linked and about 10.8 MB, so distribution can
consist of that executable alone for a compatible target. Embedded builds
contain the model weights, so distribute them with the model repositories'
license and NOTICE files (see `THIRD_PARTY_NOTICES.md`). Generated embedding
sources and copied model inputs live in the Zig build cache, never in `src/`.
Changing weights/configs requires rebuilding, and the normal build without this
option contains no weights.

Bundled CLI builds select their embedded models by default. Explicit `--models`,
`--model-dir`, `--google-dir` or `--g2p-dir` selects filesystem loading instead;
there is no silent fallback when an explicit path is wrong. `--embedded` forces
the bundled path and cannot be combined with model-path options. It reports
`ModelsNotEmbedded` in an ordinary build. Both `detect` and `listen` support this
selection; the parity tool supports `--models DIR` or `--embedded` as well.

The shared and static C libraries also include the bundle when built this way.
Call `naptrap_has_embedded_models()` to inspect availability and
`naptrap_model_load_embedded(&model)` to load it with no filesystem access.
`naptrap_model_load(...)` always honors its explicit paths. For Zig consumers,
the build exports the `naptrap_bundled` module with `available` and
`load(gpa)`. The core `naptrap` module remains independent of build options;
`Model.fromBytes(gpa, phonmatchnet_bytes, google_bytes, g2p_bytes)` also
accepts caller-provided `{config, weights}` byte slices and copies all retained
model data, so the inputs can be freed afterwards. Release either kind of model
with `model.deinit()`.

Embedded and filesystem loaders share format, SHA-256, encoder-pin, provenance,
dimension and threshold validation. Embedding does not bypass these checks.

## Testing

```sh
zig build test                        # unit tests and the C ABI smoke test
zig build test -Dtest-models=models   # plus end-to-end tests against the v1 models
zig build check -Dtarget=arm-linux-musleabihf   # compile everything for a target
```

The end-to-end tests cover loading, keyword limits, input validation, stream
chunk independence and callbacks, and the C ABI with real models. CI runs them
on Linux, macOS and Windows and compiles for 32-bit and other cross targets.

## Parity workflow

`naptrap-parity` compares every stage with tensors dumped from the PyTorch
reference. It is a developer tool, installed only with `-Dinstall-parity`:

```sh
python3.12 -m venv .venv
.venv/bin/pip install "naptrap[hub]"
.venv/bin/python tools/dump_fixtures.py --models models \
    --out fixtures/hey_computer.safetensors \
    --keyword "hey computer"            # optionally --wav clip.wav
zig build -Doptimize=ReleaseFast -Dinstall-parity
zig-out/bin/naptrap-parity fixtures/hey_computer.safetensors --models models
```

Output reports each stage (`text`, `keyword_mel`, `google_log_mel`, `google`,
`logit`, end-to-end `logit`), maximum absolute difference, and the worst index
with its reference and computed values. The Google convolution stack and matcher
also run on reference features to isolate their errors. Reference shapes,
finite values, waveform range and normalized keyword metadata are checked before
inference. Nonfinite computed values fail too. Exit codes are **0** for parity,
**1** for a numerical mismatch, and **2** for an invalid fixture/model or other error.

### Numerical tolerances

| Stage | Criterion |
| --- | --- |
| Text | absolute difference ≤ `1e-4` |
| Keyword log-mel | log10 difference ≤ `1e-3`, **or** linear mel-power difference ≤ `5e-9` |
| Google log-mel | natural-log difference ≤ `1e-3`, **or** linear mel-magnitude difference ≤ `1e-5` |
| Google embeddings / matcher on reference inputs | absolute difference ≤ `1e-3` |
| End-to-end logit | absolute difference ≤ `5e-3` |

The small linear-domain allowances account for float32 FFT roundoff in nearly
empty spectral bins. They do not relax the embedding or logit checks. On DC and
narrowband signals, the Zig frontends match a float64 FFT of the same
float32-windowed samples within `3.6e-7`, while PyTorch's float32 output can differ
by about `0.007` after the logarithm. The corresponding linear differences are
below `3.7e-9` in mel power and `7.1e-6` in mel magnitude in the validation cases.
The checker reports when bins use this allowance.

Add **`--strict-frontends`** to retain the original log-only `1e-3` criterion.
Some DC and pure-tone fixtures intentionally fail that diagnostic mode; it is
useful for inspecting numerical differences rather than hiding them.

### Implementation details

- **STFT details.** `torch.stft(center=True)` reflect-pads 256 samples per side
  and centers the 400-sample window inside the 512-point frame (offset 56).
  Google's frontend left-pads 11,120 zeros, does not center, and uses the
  magnitude rather than the power.
- **GRU math.** PyTorch's candidate gate is `tanh(W_in x + b_in + r * (W_hn h + b_hn))`;
  the reset gate multiplies the hidden-side bias too.
- **ConvTranspose1d length.** `(G - 1) * 4 + 5`, then zero-padded or cropped to
  the audio branch's length.
- **Keyword normalization.** Matches Python's Unicode **15.0.0** policy: replace
  curly apostrophes, NFKD, remove nonzero combining classes, lowercase, tokenize
  English letters/apostrophes, and reject numbers. Fullwidth letters, ligatures,
  compatibility digits and fractions are handled. The generated table projects
  normalization onto this ASCII token alphabet; it is not a general-purpose
  Unicode normalizer. All 1,112,064 scalar values and 5,000 mixed-context strings
  were checked against Python. A Python reference using a newer Unicode database
  can differ for newly assigned characters.

Check or regenerate the Unicode table with Python 3.12 (Unicode 15.0.0):

```sh
python3.12 tools/generate_unicode.py --out src/unicode_data.zig --check
# Omit --check to regenerate the checked-in source.
```

## Performance and memory

The convolution kernel packs eight spatial positions at a time and uses SIMD
across them, preserving the per-output accumulation order. Mel projections also
use SIMD. The audio frontends use f64 FFT butterflies to reduce cancellation
error, then return f32 features. The Google encoder alternates between two
feature buffers and reuses its convolution patch storage. The matcher feeds
attention rows directly into its GRU instead of retaining full attention/output
sequences. No BLAS, FFT or Unicode library is required at runtime.

On a Linux/x86_64 development host in ReleaseFast, a warm 1.5-second detection
takes about **22 ms** with a reused model and scratch arena, model loading about
10 ms and encoding `hey computer` about 1 ms. These are host-specific
measurements, not a latency guarantee. A loaded model holds about **9 MiB**
(its decoded weights; `Model.load` frees the raw file contents before
returning). `detect` needs about **2.2 MiB** of scratch for a 1.5-second clip
and **2.7 MiB** for a 2-second clip, and a reused arena stays at that capacity.
These figures exclude keyword storage and a stream's ring/window buffers.
Keywords must outlive the detections and streams that use them.

## Supported targets

The library has no OS dependencies beyond file reading in `Model.load` (use
`Model.fromBytes` or embedded models where there is no file system). CI tests
x86_64 Linux, macOS and Windows, runs the end-to-end tests on i386 Linux, and
compiles for armv7, aarch64 and riscv64 Linux and wasm32-wasi.

## Layout

| File | What it does |
| --- | --- |
| `src/root.zig` | Package root and API overview |
| `src/model.zig` | Verifies artifact formats, SHA-256, encoder pins, Google provenance and G2P dimensions; `keyword()` and `detect()` |
| `src/stream.zig` | Chunk-independent sliding windows, sample timestamps and debounced wake events |
| `src/text.zig` | Keyword normalization and the g2p_en decoder states → `[P, 256]` |
| `src/unicode_data.zig` | Generated Unicode 15.0.0 normalization data; no runtime Unicode dependency |
| `src/features.zig` | 512-point radix-2 FFT, the 40-bin keyword log-mel (`torch.stft` semantics) and Google's 32-bin log-mel (TF framing) |
| `src/google.zig` | Google `speech_embedding` conv stack → `[len/1280, 96]` |
| `src/phonmatchnet.zig` | The matcher: conv audio branch, Google upsampling, causal attention, GRU, logit |
| `src/ops.zig` | Kernels: `@Vector` dot, Linear, Conv1d/2d, ConvTranspose1d, MaxPool, BatchNorm-as-affine, GRU step, softmax |
| `src/safetensors.zig` | Parses safetensors (F32/F64/F16/BF16 → f32; integer tensors skipped) |
| `src/wav.zig` | Mono 16 kHz 16-bit PCM WAV reader |
| `src/main.zig` | CLI with the same JSON output and exit codes as `naptrap detect` |
| `src/c_api.zig`, `include/naptrap.h` | Versioned native ABI with reference-counted model, keyword and stream handles |
| `src/unbundled.zig`, build-generated `naptrap_bundled` | Optional model/config embedding, with the same validation as disk loading |
| `src/parity.zig` | Compares each stage against PyTorch fixtures |
| `tests/` | End-to-end model tests and the C ABI smoke test |
| `tools/dump_fixtures.py` | Writes parity fixtures from the Python model |
| `tools/generate_unicode.py` | Rebuilds/checks the versioned Unicode source table with Python 3.12 |

## License

Apache-2.0; see `LICENSE` and `NOTICE`. The network follows the PhonMatchNet
design (Lee & Cho, Interspeech 2023) and NCSOFT's BSD-3-Clause reference code,
and the keyword normalization table is generated from Unicode data. Third-party
licenses are in `licenses/` and summarized in `THIRD_PARTY_NOTICES.md`.
