# naptrap-zig

A Zig 0.16 inference runtime for [naptrap](https://github.com/androiddrew/naptrap)'s
PhonMatchNet text-defined wake word model. It has no dependencies beyond the Zig
standard library, does fp32 batch-1 CPU inference, and reads the same three
Hugging Face artifact directories as the Python package.

> **Status: written but never compiled.** The environment this was scaffolded in
> had no Zig toolchain or model access. Expect a round of compile fixes
> (Zig 0.16 moved I/O to `std.Io`), then use the parity tool to find numerical
> mismatches stage by stage.

## Layout

| File | What it does |
| --- | --- |
| `src/safetensors.zig` | Parses safetensors (F32/F64/F16/BF16 → f32; integer tensors skipped) |
| `src/ops.zig` | Kernels: `@Vector` dot, Linear, Conv1d/2d, ConvTranspose1d, MaxPool, BatchNorm-as-affine, GRU step, softmax |
| `src/features.zig` | 512-point DFT, the 40-bin keyword log-mel (`torch.stft` semantics) and Google's 32-bin log-mel (TF framing) |
| `src/google.zig` | Google `speech_embedding` conv stack → `[len/1280, 96]` |
| `src/text.zig` | Keyword normalization and the g2p_en decoder states → `[P, 256]` |
| `src/phonmatchnet.zig` | The matcher: conv audio branch, Google upsampling, causal attention, GRU, logit |
| `src/model.zig` | Loads and verifies the three artifacts (format, SHA-256, pinned encoders, thresholds); `keyword()` and `detect()` |
| `src/wav.zig` | Mono 16 kHz 16-bit PCM WAV reader |
| `src/main.zig` | CLI with the same JSON output and exit codes as `naptrap detect` |
| `src/parity.zig` | Compares each stage against PyTorch fixtures |
| `tools/dump_fixtures.py` | Writes those fixtures from the Python model |

## Get the models

```sh
hf download androiddrew/naptrap-phonmatchnet-en       --revision v1 --local-dir models/phonmatchnet
hf download androiddrew/google-speech-embedding-pytorch --revision v1 --local-dir models/google
hf download androiddrew/g2p-en-decoder-pytorch        --revision v1 --local-dir models/g2p
```

## Build and run

```sh
zig build test                      # unit tests, no models needed
zig build -Doptimize=ReleaseFast
./zig-out/bin/naptrap detect clip.wav "hey computer" --models models
```

If Zig rejects the `fingerprint` in `build.zig.zon`, it prints a suggested
value; paste that in.

## Parity workflow

```sh
pip install "naptrap[hub]"
python tools/dump_fixtures.py --models models --out fixtures/hey_computer.safetensors \
    --keyword "hey computer"            # optionally --wav clip.wav
zig build parity -- fixtures/hey_computer.safetensors --models models
```

Output is one line per stage (`text`, `keyword_mel`, `google_log_mel`,
`google`, `logit`, end-to-end `logit`) with the max absolute difference. Each
stage after the first runs on the reference inputs, so a failure points at one
stage.

Things most likely to need attention:

- **STFT details.** `torch.stft(center=True)` reflect-pads 256 samples per side
  and centers the 400-sample window inside the 512-point frame (offset 56).
  Google's frontend left-pads 11,120 zeros, does not center, and uses the
  magnitude rather than the power.
- **GRU math.** PyTorch's candidate gate is `tanh(W_in x + b_in + r * (W_hn h + b_hn))`;
  the reset gate multiplies the hidden-side bias too.
- **ConvTranspose1d length.** `(G - 1) * 4 + 5`, then zero-padded or cropped to
  the audio branch's length.
- **Keyword normalization.** Python uses Unicode NFKD. `text.zig` folds
  ASCII, Latin-1 and Latin Extended-A by table and treats everything else as a
  word separator. Ligatures and fullwidth letters will differ.

## Next steps

- Add a streaming listener (sliding 1.5 s windows) with debouncing.
- Replace the direct DFT with a radix-2 FFT and add blocked matmul if profiling
  says so. Inference should already be well under real time.
- Export a C ABI (`keyword`, `detect`) so Python, Swift or Rust can call it.
- Optionally `@embedFile` the weights into a single static binary.

## License

Apache-2.0, following naptrap. The network follows the PhonMatchNet design
(Lee & Cho, Interspeech 2023) and NCSOFT's BSD-3-Clause reference code; see
naptrap's NOTICE and THIRD_PARTY_NOTICES.md.
