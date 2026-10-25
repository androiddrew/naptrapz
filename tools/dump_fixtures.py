"""Dump reference tensors from the PyTorch naptrap model for Zig parity tests.

    pip install "naptrap[hub]"
    python tools/dump_fixtures.py --models models --out fixtures/hey_computer.safetensors \
        --keyword "hey computer" [--wav clip.wav]

Without --wav, a deterministic 1.5 s synthetic clip is used. That is enough
for numerical parity; use a real recording to sanity-check scores.

The output file holds float32 tensors:
    wave            [N]
    keyword_mel     [T, 40]
    google_log_mel  [F, 32]
    google          [G, 96]
    text            [P, 256]
    logit           [1]
and metadata {"keyword": ..., "normalized": ...}.
"""

import argparse
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from naptrap import load_model, load_wav
from naptrap.text import normalize_keyword


def synthetic_clip(seconds=1.5, rate=16000):
    generator = torch.Generator().manual_seed(1234)
    t = torch.arange(int(seconds * rate), dtype=torch.float64) / rate
    chirp = torch.sin(2 * math.pi * (200 * t + 400 * t * t))
    noise = torch.randn(t.shape, generator=generator, dtype=torch.float64)
    return (0.3 * chirp + 0.05 * noise).clamp(-1, 1).to(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models", help="directory with phonmatchnet/, google/, g2p/")
    parser.add_argument("--out", default="fixtures/fixture.safetensors")
    parser.add_argument("--keyword", default="hey computer")
    parser.add_argument("--wav", help="mono 16 kHz 16-bit PCM WAV, 0.5-2 s")
    args = parser.parse_args()

    root = Path(args.models)
    model = load_model("phonmatchnet", model_dir=root / "phonmatchnet",
                       google_dir=root / "google", g2p_dir=root / "g2p")
    wave = load_wav(args.wav) if args.wav else synthetic_clip()

    with torch.inference_mode():
        keyword_mel = model._frontend(wave)            # [1, T, 40]
        google_log_mel = model.google_encoder.log_mel(wave)
        google = model.google_encoder(wave)            # [G, 96]
        text = model.text_encoder(args.keyword)        # [P, 256]
        logit = model._model(keyword_mel, google.unsqueeze(0), text.unsqueeze(0))

    tensors = {
        "wave": wave,
        "keyword_mel": keyword_mel[0],
        "google_log_mel": google_log_mel,
        "google": google,
        "text": text,
        "logit": logit.reshape(1),
    }
    tensors = {name: value.detach().to(torch.float32).contiguous() for name, value in tensors.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata={"keyword": args.keyword,
                                          "normalized": normalize_keyword(args.keyword)})
    for name, value in tensors.items():
        print(f"{name:16s} {tuple(value.shape)}")
    print(f"score {torch.sigmoid(logit).item():.4f} -> {out}")


if __name__ == "__main__":
    main()
