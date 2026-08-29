# Third-Party Notices

Diana Voice is licensed under Apache-2.0 (see [LICENSE](LICENSE)). It
incorporates or downloads the following third-party components.

## Bundled code

### qwen3-tts

- Path: `crates/qwen3-tts`
- License: MIT (see `crates/qwen3-tts/LICENSE`)
- Pure-Rust inference engine for Qwen3-TTS.
- Bundles one CUDA kernel (`kernels/fused_residual_rmsnorm.cu`) adapted from
  candle-kernels' rmsnorm (ggml heritage), used under MIT.
  candle-kernels is dual-licensed MIT / Apache-2.0.

### transcribe.cpp / transcribe-cpp-sys

- Path: `vendor/transcribe-cpp-sys-patched` (transcribe-cpp-sys 0.2.2)
- License: MIT (see `vendor/transcribe-cpp-sys-patched/LICENSE`)
- Upstream: https://github.com/handy-computer/transcribe.cpp
- Vendored with an encoder-window patch
  (`vendor/transcribe-cpp-sys-patched/PATCH-vs-crates-io-0.2.2.diff`);
  the patch is a candidate for upstreaming.

### ggml

- Path: `vendor/transcribe-cpp-sys-patched/ggml` (vendored inside
  transcribe.cpp)
- License: MIT (see `vendor/transcribe-cpp-sys-patched/ggml/LICENSE`)

### candle (candle-core, candle-nn, candle-transformers)

- Dependency of `qwen3-tts` and `voice-engine` (crates.io, v0.9)
- License: MIT / Apache-2.0 (dual)
- https://github.com/huggingface/candle

### ort / ONNX Runtime

- Dependency of `voice-runtime` (crates.io, ort 2.0.0-rc.10), used for
  Silero VAD inference.
- License: MIT (both the `ort` crate and Microsoft's ONNX Runtime)
- https://github.com/pykeio/ort · https://github.com/microsoft/onnxruntime

## Model weights (downloaded on first run, not bundled)

### Whisper Large v3 Turbo (GGUF)

- File: `whisper-large-v3-turbo-Q8_0.gguf` (~845 MB)
- License: Apache-2.0, inherited from
  [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)

### Qwen3-TTS 12Hz 0.6B Base

- License: Apache-2.0

### Silero VAD v5

- License: MIT, Copyright (c) 2020-present Silero Team
- https://github.com/snakers4/silero-vad
