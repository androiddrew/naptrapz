# naptrap-zig

A Zig 0.16 inference runtime for [naptrap](https://github.com/androiddrew/naptrap)'s
PhonMatchNet text-defined wake word model. It has no dependencies beyond the Zig
standard library, does fp32 batch-1 CPU inference, and reads the same three
Hugging Face artifact directories as the Python package. The audio frontends use
f64 FFT butterflies to reduce cancellation error, then return f32 features.

Tested on Linux/x86_64 with Zig **0.16.0**, Python **3.12.11** and PyTorch
**2.14.1+cpu**, using the released `v1` artifacts. The validation corpus includes
25 speech clips, five keywords and both calibrated operating points: all 250
decisions match Python, with maximum score difference below `2e-6` in ReleaseFast.

## Layout

| File | What it does |
| --- | --- |
| `src/safetensors.zig` | Parses safetensors (F32/F64/F16/BF16 → f32; integer tensors skipped) |
| `src/ops.zig` | Kernels: `@Vector` dot, Linear, Conv1d/2d, ConvTranspose1d, MaxPool, BatchNorm-as-affine, GRU step, softmax |
| `src/features.zig` | 512-point radix-2 FFT, the 40-bin keyword log-mel (`torch.stft` semantics) and Google's 32-bin log-mel (TF framing) |
| `src/google.zig` | Google `speech_embedding` conv stack → `[len/1280, 96]` |
| `src/text.zig` | Keyword normalization and the g2p_en decoder states → `[P, 256]` |
| `src/unicode_data.zig` | Generated Unicode 15.0.0 normalization data; no runtime Unicode dependency |
| `src/phonmatchnet.zig` | The matcher: conv audio branch, Google upsampling, causal attention, GRU, logit |
| `src/model.zig` | Verifies artifact formats, SHA-256, encoder pins, Google provenance and G2P dimensions; `keyword()` and `detect()` |
| `src/wav.zig` | Mono 16 kHz 16-bit PCM WAV reader |
| `src/main.zig` | CLI with the same JSON output and exit codes as `naptrap detect` |
| `src/parity.zig` | Compares each stage against PyTorch fixtures |
| `tools/dump_fixtures.py` | Writes those fixtures from the Python model |
| `tools/generate_unicode.py` | Rebuilds/checks the versioned Unicode source table with Python 3.12 |

## Get the models

Commands below run from the source directory and put model files, fixtures and
build outputs in the sibling `../work/` directory.

```sh
hf download androiddrew/naptrap-phonmatchnet-en       --revision v1 --local-dir ../work/models/phonmatchnet
hf download androiddrew/google-speech-embedding-pytorch --revision v1 --local-dir ../work/models/google
hf download androiddrew/g2p-en-decoder-pytorch        --revision v1 --local-dir ../work/models/g2p
```

## Build and run

```sh
zig build test --cache-dir ../work/zig-cache --global-cache-dir ../work/zig-global-cache
zig build -Doptimize=ReleaseFast --prefix ../work \
    --cache-dir ../work/zig-cache --global-cache-dir ../work/zig-global-cache
../work/bin/naptrap detect clip.wav "hey computer" --models ../work/models
```

The CLI accepts mono, 16 kHz, 16-bit PCM WAVs of 0.5–2 seconds. It prints JSON
and exits with **0** for an accepted keyword, **1** for rejection, or **2** for an
input/model error. `--far` must exactly select a calibrated operating point;
the released model provides `0.01` (default) and `0.1`.

## Parity workflow

```sh
python3.12 -m venv ../work/venv
../work/venv/bin/pip install "naptrap[hub]"
../work/venv/bin/python tools/dump_fixtures.py --models ../work/models \
    --out ../work/fixtures/hey_computer.safetensors \
    --keyword "hey computer"            # optionally --wav clip.wav
../work/bin/naptrap-parity ../work/fixtures/hey_computer.safetensors --models ../work/models
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

## Performance and reuse

The convolution kernel packs eight spatial positions at a time and uses SIMD
across them, preserving the per-output accumulation order. Mel projections also
use SIMD. No BLAS, FFT or Unicode library is required at runtime.

On the validation Linux/x86_64 host, a warm 1.5-second inference averaged about
**22 ms**, down from **251 ms**, over 30 iterations with a reused model and scratch
arena. Model loading took about 11 ms and encoding `hey computer` about 1 ms.
These are host-specific measurements, not a latency guarantee. The CLI's median
end-to-end time over the speech corpus was about 38 ms, including model loading.

For repeated detection, keep `Model` and its allocation arena alive, compute
`keyword()` once, and pass a separate scratch arena to `detect()`. Reset that
arena with `.retain_capacity` between clips. The measured scratch capacity for a
1.5-second clip was about 8.1 MiB. Keyword features must outlive their use and
must not be allocated from an arena reset between detections.

## Next steps

- Add a streaming listener (sliding 1.5 s windows) with debouncing.
- Reduce intermediate feature-map storage and profile the remaining matcher work.
- Export a C ABI (`keyword`, `detect`) so Python, Swift or Rust can call it.
- Optionally `@embedFile` the weights into a single static binary.

## License

Apache-2.0, following naptrap. The network follows the PhonMatchNet design
(Lee & Cho, Interspeech 2023) and NCSOFT's BSD-3-Clause reference code; see
naptrap's NOTICE and THIRD_PARTY_NOTICES.md.
