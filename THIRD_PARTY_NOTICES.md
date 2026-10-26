# Third-party notices

naptrap-zig's own code is licensed under Apache-2.0 (see `LICENSE` and `NOTICE`).
The source package contains **no model weights**. Weights are downloaded
separately from Hugging Face, and each repository carries its own license and
notices. Builds made with `-Dembed-models=DIR` contain the weights you supply;
distribute them with those repositories' license and NOTICE files.

| Component | Used for | License | Where |
|---|---|---|---|
| PhonMatchNet architecture and training code, © NCSOFT Corporation | `src/phonmatchnet.zig` follows its network design | BSD-3-Clause | `licenses/PhonMatchNet-BSD-3-Clause.txt`, <https://github.com/ncsoft/PhonMatchNet> |
| PhonMatchNet weights, trained independently by Drew Bednar | default keyword model | Apache-2.0 | <https://huggingface.co/androiddrew/naptrap-phonmatchnet-en> |
| Google speech_embedding/1 | frozen audio features (`src/google.zig`) | Apache-2.0 | <https://huggingface.co/androiddrew/google-speech-embedding-pytorch> |
| g2p_en 2.1.0, Kyubyong Park & Jongseok Kim | frozen keyword text features (`src/text.zig`) | Apache-2.0 | <https://huggingface.co/androiddrew/g2p-en-decoder-pytorch> |
| Unicode Character Database 15.0.0, © Unicode, Inc. | generated normalization table (`src/unicode_data.zig`) | Unicode License V3 | `licenses/Unicode-License-v3.txt`, <https://www.unicode.org/license.txt> |
| LibriSpeech (Panayotov et al.) | training data for the PhonMatchNet weights | CC BY 4.0 | <https://www.openslr.org/12> |

naptrap-zig and these weights are not affiliated with or endorsed by NCSOFT,
the PhonMatchNet authors, Google, the g2p_en authors or Unicode, Inc.
