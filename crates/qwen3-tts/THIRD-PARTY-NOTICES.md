# Third-party notices

This crate is MIT licensed (see `LICENSE`). It carries the following third-party
material, whose original notices are reproduced as their licenses require.

## candle-kernels — MIT OR Apache-2.0

`kernels/fused_residual_rmsnorm.cu` adapts the RMSNorm kernel from
[candle-kernels](https://github.com/huggingface/candle/tree/main/candle-kernels)
(Hugging Face), which itself carries ggml heritage. candle is dual-licensed
MIT OR Apache-2.0; this crate uses it under the MIT terms, whose full text is
identical to `LICENSE` here save for the copyright holder.

Upstream: https://github.com/huggingface/candle — see its `LICENSE-MIT` for the
canonical notice.

## librosa — ISC

`src/audio/mel.rs` implements mel-spectrogram computation following the approach
documented by [librosa](https://github.com/librosa/librosa) (ISC licensed),
tuned for the Qwen3-TTS speaker encoder. This is an independent Rust
implementation of a published signal-processing method, not a port of librosa's
source, and is credited here for provenance rather than out of a licensing
obligation.

## Model weights are NOT covered by this license

`LICENSE` covers **this crate's source code only**. The Qwen3-TTS model weights
are downloaded at runtime from Hugging Face and carry their own terms — see the
[Qwen3-TTS model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
Anyone redistributing this crate as part of a product must read those terms
separately.
