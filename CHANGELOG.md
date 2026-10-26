# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/); the C ABI is versioned separately
by `NAPTRAP_ABI_VERSION`.

## [0.1.0] - Unreleased

First public release.

### Added

- Zig 0.16 inference runtime for naptrap's PhonMatchNet text-defined wake word
  model, reading the released `v1` Hugging Face artifacts, with SHA-256,
  encoder-pin, provenance and dimension checks.
- `Model.load` / `Model.fromBytes`, `Model.keyword`, `Model.detect` and
  `Stream` for chunk-independent, debounced detection on continuous audio.
- `naptrap` CLI with `detect` (WAV) and `listen` (raw PCM on stdin)
  subcommands and JSON output.
- C ABI version 1 (`include/naptrap.h`) with shared and static libraries.
- Optional embedded-model builds (`-Dembed-models=DIR`).
- `naptrap-parity` tool and `tools/dump_fixtures.py` for stage-by-stage
  comparison with the PyTorch reference.
- Unit tests, end-to-end model tests (`-Dtest-models=DIR`), a C ABI smoke test,
  and `check` and `docs` build steps.
