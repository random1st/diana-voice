//! # Qwen3-TTS
//!
//! Pure Rust inference for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
//! a high-quality text-to-speech model from Alibaba.
//!
//! ## Features
//!
//! - **CPU inference** with optional MKL/Accelerate for faster BLAS operations
//! - **CUDA** support for NVIDIA GPU acceleration
//! - **Metal** support for Apple Silicon
//! - **Streaming-friendly** architecture with incremental token generation
//! - **Voice cloning** via ECAPA-TDNN speaker encoder (Base models)
//! - **Auto-detection** of model variant from `config.json`
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use qwen3_tts::{Qwen3TTS, SynthesisOptions, auto_device};
//!
//! // Load model — variant auto-detected from config.json
//! let device = auto_device()?;
//! let model = Qwen3TTS::from_pretrained("path/to/model", device)?;
//!
//! // Synthesize speech with default settings
//! let audio = model.synthesize("Hello, world!", None)?;
//! audio.save("output.wav")?;
//!
//! // Or with custom options
//! let options = SynthesisOptions {
//!     temperature: 0.8,
//!     top_k: 30,
//!     ..Default::default()
//! };
//! let audio = model.synthesize("Custom settings!", Some(options))?;
//! ```
//!
//! ## Architecture
//!
//! The TTS pipeline consists of three stages:
//!
//! 1. **TalkerModel**: Transformer that generates semantic tokens from text
//!    autoregressively. Uses dual embeddings (text + codec) with MRoPE
//!    (multimodal rotary position encoding) across all variants.
//!
//! 2. **CodePredictor**: For each semantic token, generates 15 acoustic
//!    tokens using a 5-layer autoregressive decoder. The code predictor
//!    always has `hidden_size=1024` regardless of the talker size; 1.7B
//!    models use a `small_to_mtp_projection` layer to bridge the gap.
//!
//! 3. **Decoder12Hz**: Converts the 16-codebook codec tokens to audio
//!    waveform at 24kHz. Uses ConvNeXt blocks and transposed convolutions
//!    for upsampling. Shared across all model variants.
//!
//! ## Model Variants
//!
//! Five official variants exist in two size classes:
//!
//! | Variant | Size | Talker hidden | Speaker conditioning | HuggingFace ID |
//! |---------|------|---------------|---------------------|----------------|
//! | 0.6B Base | 1.8 GB | 1024 | Voice cloning (ECAPA-TDNN) | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
//! | 0.6B CustomVoice | 1.8 GB | 1024 | 9 preset speakers | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
//! | 1.7B Base | 3.9 GB | 2048 | Voice cloning (ECAPA-TDNN) | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
//! | 1.7B CustomVoice | 3.9 GB | 2048 | 9 preset speakers | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
//! | 1.7B VoiceDesign | 3.8 GB | 2048 | Text-described voices | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
//!
//! **Base**: Includes a speaker encoder for voice cloning from reference audio.
//! Supports x_vector_only (speaker embedding) and ICL (in-context learning
//! with reference audio + text) modes.
//!
//! **CustomVoice**: 9 preset speakers (Serena, Vivian, Ryan, Aiden, etc.) with
//! no speaker encoder. Uses discrete speaker token IDs for voice selection.
//!
//! **VoiceDesign**: Creates novel voices from text descriptions (e.g.,
//! "a deep male voice"). No speaker encoder or preset speakers.
//!
//! All variants share the same speech tokenizer and decoder weights. The
//! code predictor architecture is identical (1024 hidden, 5 layers, 16 heads)
//! across all variants.
//!
//! ## Sample Rate
//!
//! Output audio is always 24kHz mono. Use [`audio::resample()`] if you need
//! a different sample rate.

pub mod audio;
pub mod generation;
#[cfg(feature = "hub")]
pub mod hub;
pub mod models;
pub mod profiling;
pub mod tokenizer;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use models::codec::{Decoder12Hz, Encoder12Hz};
use models::speaker::SpeakerEncoder;
use models::AnyKVCache;

// SynthesisTiming is defined above, already public.

/// Re-exports for convenience
pub use audio::AudioBuffer;
#[cfg(feature = "hub")]
pub use hub::ModelPaths;
pub use models::config::Qwen3TTSConfig;
// StreamingSession is defined in this module, exported as top-level type
pub use generation::SamplingContext;
pub use models::talker::{codec_tokens, special_tokens, tts_tokens, Language, Speaker};
pub use models::{
    CodePredictor, CodePredictorConfig, ModelType, ParsedModelConfig, SpeakerEncoderConfig,
    TalkerConfig, TalkerModel,
};

/// A sequence of codec frames, where each frame contains 16 codebook values
/// (1 semantic + 15 acoustic, formatted as `[semantic, acoustic_0..14]`).
pub type FrameCodes = Vec<Vec<u32>>;

/// Reference audio prompt for voice cloning.
///
/// Holds the speaker embedding and optional ICL (in-context learning) data.
/// Created via [`Qwen3TTS::create_voice_clone_prompt`].
pub struct VoiceClonePrompt {
    /// Speaker embedding from the ECAPA-TDNN encoder, shape `[enc_dim]` (typically 1024).
    pub speaker_embedding: Tensor,
    /// Reference audio codec codes for ICL mode, shape `[T, 16]`. `None` = x_vector_only mode.
    pub ref_codes: Option<Tensor>,
    /// Tokenized reference text for ICL mode.
    pub ref_text_ids: Option<Vec<u32>>,
}

/// Everything the generation loop needs after a voice-clone prefill.
///
/// Produced by [`Qwen3TTS::prepare_voice_clone`] so the batch path and the
/// streaming session share one setup instead of two that can drift apart.
struct VoiceClonePrefill {
    config: generation::GenerationConfig,
    kv_caches: Vec<AnyKVCache>,
    /// Position after prefill **and** the ICL extension, when ICL is active.
    offset: usize,
    /// Hidden state the code predictor is conditioned on — the ICL tail when
    /// ICL is active, the prefill tail otherwise.
    last_hidden: Tensor,
    logits: Tensor,
    trailing_text_hidden: Tensor,
    trailing_text_len: usize,
    tts_pad_embed: Tensor,
}

/// Per-stage timing breakdown from a synthesis run.
#[derive(Debug, Clone, Serialize)]
pub struct SynthesisTiming {
    /// Time spent in the prefill phase (ms).
    pub prefill_ms: f64,
    /// Time spent in the autoregressive generation loop (ms).
    pub generation_ms: f64,
    /// Number of codec frames generated.
    pub generation_frames: usize,
    /// Time spent decoding codec frames to audio (ms).
    pub decode_ms: f64,
}

/// Main TTS interface using proper autoregressive pipeline.
///
/// Supports all 5 Qwen3-TTS model variants. Use [`model_type()`](Self::model_type)
/// to check which variant was loaded and [`supports_voice_cloning()`](Self::supports_voice_cloning)
/// / [`supports_preset_speakers()`](Self::supports_preset_speakers) to check capabilities.
pub struct Qwen3TTS {
    /// Talker model for semantic token generation
    talker: TalkerModel,
    /// Code predictor for acoustic token generation
    code_predictor: CodePredictor,
    /// 12Hz decoder for audio synthesis
    decoder: Decoder12Hz,
    /// CPU twin of the decoder, present unless `QWEN3_TTS_OVERLAP_DECODE=0`.
    /// Streaming decodes chunks 2..n on it, on a worker thread, while the Metal
    /// queue generates the next chunk — the two devices share nothing, which is
    /// what makes the overlap safe. The Metal `decoder` still serves the batch
    /// path and the latency-critical first chunk.
    cpu_decoder: Option<Decoder12Hz>,
    /// Text tokenizer
    text_tokenizer: tokenizer::TextTokenizer,
    /// Speaker encoder for voice cloning (loaded when weights are present)
    speaker_encoder: Option<SpeakerEncoder>,
    /// Speech tokenizer encoder for ICL voice cloning (encodes reference audio → codes)
    speech_encoder: Option<Encoder12Hz>,
    /// Detected model variant (None if loaded without config.json)
    model_type: Option<ModelType>,
    /// Device to run inference on
    device: Device,
    /// Compute dtype for talker + code predictor (BF16 on CUDA, F32 otherwise)
    compute_dtype: DType,
}

impl Qwen3TTS {
    /// Load a model from a HuggingFace model ID or local path.
    ///
    /// Auto-detects the model variant (0.6B/1.7B, Base/CustomVoice/VoiceDesign)
    /// from `config.json` if present, falling back to weight inspection.
    ///
    /// The text tokenizer is resolved from `model_id/tokenizer.json` if present,
    /// otherwise downloaded from HuggingFace Hub. Use `tokenizer_id` to override.
    pub fn from_pretrained(model_id: &str, device: Device) -> Result<Self> {
        Self::from_pretrained_with_tokenizer(model_id, None, device)
    }

    /// Load a model with an explicit tokenizer source.
    ///
    /// `tokenizer_id` can be a local directory, a file path, or a HuggingFace
    /// model ID (e.g. `"Qwen/Qwen2-0.5B"`). If `None`, resolves from the
    /// model directory or falls back to the default tokenizer repo.
    pub fn from_pretrained_with_tokenizer(
        model_id: &str,
        tokenizer_id: Option<&str>,
        device: Device,
    ) -> Result<Self> {
        tracing::info!("Loading Qwen3-TTS from: {}", model_id);
        tracing::info!("Compute dtype: {:?}", compute_dtype_for_device(&device));

        // Try to parse config.json for auto-detection
        let config_path = Path::new(model_id).join("config.json");
        let parsed_config = if config_path.exists() {
            match ParsedModelConfig::from_file(&config_path) {
                Ok(cfg) => {
                    tracing::info!("Detected model variant: {}", cfg.label());
                    Some(cfg)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse config.json, falling back to weight inspection: {}",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        // Load text tokenizer
        let tok_source = tokenizer_id.unwrap_or(model_id);
        let text_tokenizer = tokenizer::TextTokenizer::from_pretrained(tok_source)?;

        // Load model weights
        let model_path = Path::new(model_id).join("model.safetensors");
        if !model_path.exists() {
            anyhow::bail!(
                "Model weights not found at {}. Please download the model first.",
                model_path.display()
            );
        }
        let weights = Self::load_weights(&model_path, &device)?;

        // Load speech tokenizer for decoder
        let st_path = Path::new(model_id).join("speech_tokenizer/model.safetensors");
        let st_weights = if st_path.exists() {
            Self::load_weights(&st_path, &device)?
        } else {
            // Fall back to looking in parent dir
            let alt_path = Path::new(model_id)
                .parent()
                .map(|p| p.join("speech_tokenizer/model.safetensors"));
            if let Some(p) = alt_path {
                if p.exists() {
                    Self::load_weights(&p, &device)?
                } else {
                    anyhow::bail!("Speech tokenizer weights not found");
                }
            } else {
                anyhow::bail!("Speech tokenizer weights not found");
            }
        };

        Self::build_from_components(
            &weights,
            &st_weights,
            text_tokenizer,
            parsed_config.as_ref(),
            &device,
        )
    }

    /// Load from pre-loaded weight tensors.
    ///
    /// Uses weight inspection for auto-detection. For config.json-based
    /// detection, use [`from_pretrained`](Self::from_pretrained) instead.
    pub fn from_weights(
        model_weights: &HashMap<String, Tensor>,
        decoder_weights: &HashMap<String, Tensor>,
        text_tokenizer: tokenizer::TextTokenizer,
        device: &Device,
    ) -> Result<Self> {
        Self::build_from_components(model_weights, decoder_weights, text_tokenizer, None, device)
    }

    /// Load from downloaded model paths.
    ///
    /// Use with [`ModelPaths::download`] for automatic model downloading.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qwen3_tts::{Qwen3TTS, ModelPaths, auto_device};
    ///
    /// let paths = ModelPaths::download(None)?;
    /// let device = auto_device()?;
    /// let model = Qwen3TTS::from_paths(&paths, device)?;
    /// ```
    #[cfg(feature = "hub")]
    pub fn from_paths(paths: &hub::ModelPaths, device: Device) -> Result<Self> {
        tracing::info!("Loading Qwen3-TTS from downloaded paths...");

        let text_tokenizer = tokenizer::TextTokenizer::from_file(&paths.tokenizer)?;
        let weights = Self::load_weights(&paths.model_weights, &device)?;
        let st_weights = Self::load_weights(&paths.decoder_weights, &device)?;

        Self::build_from_components(&weights, &st_weights, text_tokenizer, None, &device)
    }

    /// Shared builder: assembles all model components from pre-loaded weights.
    ///
    /// When `parsed_config` is `Some`, uses config.json dimensions and model type.
    /// When `None`, auto-detects the model variant from weight shapes.
    fn build_from_components(
        model_weights: &HashMap<String, Tensor>,
        decoder_weights: &HashMap<String, Tensor>,
        text_tokenizer: tokenizer::TextTokenizer,
        parsed_config: Option<&ParsedModelConfig>,
        device: &Device,
    ) -> Result<Self> {
        let compute_dtype = compute_dtype_for_device(device);

        // Build TalkerModel
        let talker_config = if let Some(cfg) = parsed_config {
            TalkerConfig::from_parsed(cfg)
        } else {
            Self::detect_talker_config(model_weights)?
        };
        let talker = TalkerModel::from_weights_with_config_dtype(
            model_weights,
            talker_config,
            device,
            compute_dtype,
        )?;

        // Build CodePredictor
        let cp_config = if let Some(cfg) = parsed_config {
            CodePredictorConfig::from_parsed(cfg)
        } else {
            let talker_hidden = talker.config().hidden_size;
            if talker_hidden != 1024 {
                CodePredictorConfig {
                    codec_embed_dim: Some(talker_hidden),
                    ..CodePredictorConfig::default()
                }
            } else {
                CodePredictorConfig::default()
            }
        };
        let cp_weights = Self::filter_weights(model_weights, "talker.code_predictor.");
        let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, compute_dtype, device);
        let code_predictor = CodePredictor::new(cp_config, cp_vb)?;

        // Decoder (always F32 — convolutional, no attention)
        let decoder = Decoder12Hz::from_weights(decoder_weights, Default::default())?;

        // CPU twin for overlapped streaming decode. Not a fallback and not an
        // optimization of the decode itself (the CPU is no faster at it): its
        // whole value is living on a different device than the generation
        // loop, so a worker thread can decode chunk N while Metal generates
        // chunk N+1. The same overlap with both sides on one Metal queue
        // corrupted generation (audio nearly tripled — tts-lab, 2026-08-29).
        let cpu_decoder = if overlap_decode_enabled() && !device.is_cpu() {
            let cpu = Device::Cpu;
            let cpu_weights: HashMap<String, Tensor> = decoder_weights
                .iter()
                .map(|(k, v)| Ok((k.clone(), v.to_device(&cpu)?)))
                .collect::<Result<_>>()?;
            Some(Decoder12Hz::from_weights(&cpu_weights, Default::default())?)
        } else {
            None
        };

        // Speaker encoder (always F32, only present in Base models)
        let se_config = parsed_config.and_then(|c| c.speaker_encoder_config.clone());
        let speaker_encoder =
            Self::try_load_speaker_encoder(model_weights, se_config.as_ref(), device)?;

        // Speech encoder for ICL voice cloning
        let speech_encoder = Self::try_load_speech_encoder(decoder_weights, device)?;

        let model_type = parsed_config.map(|c| c.model_type);

        Ok(Self {
            talker,
            code_predictor,
            decoder,
            cpu_decoder,
            text_tokenizer,
            speaker_encoder,
            speech_encoder,
            model_type,
            device: device.clone(),
            compute_dtype,
        })
    }

    /// Detect talker config from weight shapes (fallback when no config.json).
    fn detect_talker_config(weights: &HashMap<String, Tensor>) -> Result<TalkerConfig> {
        let norm_weight = weights
            .get("talker.model.norm.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing talker.model.norm.weight"))?;
        let hidden_size = norm_weight.dim(0)?;
        Ok(if hidden_size == 2048 {
            TalkerConfig::custom_voice()
        } else {
            TalkerConfig::default()
        })
    }

    /// Returns the detected model type, or `None` if loaded without config.json.
    pub fn model_type(&self) -> Option<&ModelType> {
        self.model_type.as_ref()
    }

    /// Whether this model supports voice cloning (Base models with speaker encoder).
    pub fn supports_voice_cloning(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Whether this model supports preset speaker selection (CustomVoice models).
    ///
    /// Returns `true` for CustomVoice, `false` for Base and VoiceDesign.
    /// When `model_type` is unknown (loaded without config.json), returns `true`
    /// as a permissive default.
    pub fn supports_preset_speakers(&self) -> bool {
        match &self.model_type {
            Some(ModelType::CustomVoice) => true,
            Some(ModelType::Base) | Some(ModelType::VoiceDesign) => false,
            None => true, // permissive when unknown
        }
    }

    /// Whether this model supports voice design (text-described voice conditioning).
    ///
    /// Returns `true` for VoiceDesign, `false` for all other variants.
    pub fn supports_voice_design(&self) -> bool {
        matches!(&self.model_type, Some(ModelType::VoiceDesign))
    }

    /// Synthesize speech from text with default voice (Ryan, English).
    ///
    /// Convenience wrapper around [`synthesize_with_voice`](Self::synthesize_with_voice).
    pub fn synthesize(&self, text: &str, options: Option<SynthesisOptions>) -> Result<AudioBuffer> {
        self.synthesize_with_voice(text, Speaker::Ryan, Language::English, options)
    }

    /// Synthesize speech with per-stage timing breakdown.
    ///
    /// Same as [`synthesize_with_voice`](Self::synthesize_with_voice) but also
    /// returns a [`SynthesisTiming`] with prefill, generation, and decode durations.
    /// Uses [`sync_device`] at timing boundaries for accurate GPU measurements.
    pub fn synthesize_with_timing(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<(AudioBuffer, SynthesisTiming)> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;
        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // -- Prefill --
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        sync_device(&self.device)?;
        let t_prefill = Instant::now();

        let mut kv_caches = self.talker.new_kv_caches(gen_config.max_new_tokens + 256);
        let (hidden, logits) =
            self.talker
                .prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        sync_device(&self.device)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        // -- Generation --
        let t_gen = Instant::now();

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        sync_device(&self.device)?;
        let generation_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        let generation_frames = all_codes.len();

        // -- Decode --
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        let t_decode = Instant::now();
        let audio = self.decode_codes(&all_codes)?;

        sync_device(&self.device)?;
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

        let timing = SynthesisTiming {
            prefill_ms,
            generation_ms,
            generation_frames,
            decode_ms,
        };

        Ok((audio, timing))
    }

    /// Build trailing text embeddings from input token IDs.
    ///
    /// Returns (trailing_text_hidden, trailing_text_len, tts_pad_embed).
    /// The trailing text is: remaining text tokens (all except first) projected + tts_eos.
    /// After trailing text is exhausted, tts_pad is used for each subsequent step.
    fn build_trailing_text(&self, input_ids: &[u32]) -> Result<(Tensor, usize, Tensor)> {
        let trailing_text_hidden = if input_ids.len() > 1 {
            let remaining_proj = self.talker.get_projected_text_embeddings(&input_ids[1..])?;
            let tts_eos_embed = self.talker.get_tts_eos_embed()?;
            Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?
        } else {
            self.talker.get_tts_eos_embed()?
        };
        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;
        Ok((trailing_text_hidden, trailing_text_len, tts_pad_embed))
    }

    /// Core generation loop shared by all synthesis methods.
    ///
    /// Runs the autoregressive generation loop: for each frame, check EOS,
    /// generate acoustic codes via CodePredictor, build the residual VQ sum
    /// with trailing text fusion, and sample the next semantic token.
    ///
    /// Callers handle prefill (which varies by model variant) and post-processing
    /// (decode, ICL ref_codes prepending, etc.).
    #[allow(clippy::too_many_arguments)]
    fn generate_codes(
        &self,
        gen_config: &generation::GenerationConfig,
        sampling_ctx: &mut generation::SamplingContext,
        kv_caches: &mut [AnyKVCache],
        mut offset: usize,
        mut last_hidden: Tensor,
        initial_logits: &Tensor,
        trailing_text_hidden: &Tensor,
        trailing_text_len: usize,
        tts_pad_embed: &Tensor,
    ) -> Result<FrameCodes> {
        // Pre-build the token suppression mask once (reused every frame)
        let suppression_mask = generation::build_suppression_mask(
            codec_tokens::CODEC_VOCAB_SIZE,
            CODEC_EOS_TOKEN_ID,
            &self.device,
        )?;

        // GPU-side repetition penalty mask: [1, vocab] — updated incrementally
        // instead of transferring all generated tokens to CPU each frame.
        let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;
        let mut penalty_mask = Tensor::zeros((1, vocab_size), DType::F32, &self.device)?;

        // Pre-allocate code predictor KV caches (reused + reset each frame)
        let mut cp_kv_caches = self.code_predictor.new_kv_caches();

        // Sample first semantic token
        let logits_2d = initial_logits.squeeze(1)?;
        let logits_2d = self.apply_generation_penalties_gpu(
            &logits_2d,
            &penalty_mask,
            gen_config,
            0,
            Some(&suppression_mask),
        )?;
        let mut semantic_token_tensor = generation::sample(&logits_2d, gen_config, sampling_ctx)?;
        tracing::trace!(target: "gpu_sync", "to_vec1 in generate_codes first token");
        let mut semantic_token: u32 = semantic_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
        // Update penalty mask with this token (O(1) CPU work)
        Self::update_penalty_mask(&mut penalty_mask, semantic_token, vocab_size)?;

        // Accumulate frames as GPU tensors: Vec of [16] U32 tensors
        // Deferred to_vec1 at the end eliminates per-frame acoustic code sync.
        let mut gpu_frames: Vec<Tensor> = Vec::new();

        #[cfg(feature = "profiling")]
        let _gen_span = tracing::info_span!("generate_frames").entered();

        for frame_idx in 0..gen_config.max_new_tokens {
            if let Some(eos_id) = gen_config.eos_token_id {
                if semantic_token == eos_id {
                    break;
                }
            }

            // Embedding lookup using GPU-resident token tensor (no CPU→GPU roundtrip)
            let semantic_embed = self
                .talker
                .get_codec_embedding_from_tensor(&semantic_token_tensor)?;

            #[cfg(feature = "profiling")]
            let _cp_span = tracing::info_span!("code_predictor", frame = frame_idx).entered();

            let acoustic_codes_tensor = self.code_predictor.generate_acoustic_codes(
                &last_hidden,
                &semantic_embed,
                &mut cp_kv_caches,
            )?;

            // Metal/CUDA queue work asynchronously, so a span that ends without a
            // sync measures submission, not execution — and the next span that
            // happens to read from the device inherits the whole backlog. Without
            // these the trace blamed sampling for four times its real cost.
            #[cfg(feature = "profiling")]
            sync_device(&self.device)?;
            #[cfg(feature = "profiling")]
            drop(_cp_span);

            // Build [16] frame tensor on GPU: [semantic_token, acoustic_0..14]
            let frame_tensor = Tensor::cat(
                &[&semantic_token_tensor.reshape(1)?, &acoustic_codes_tensor],
                0,
            )?;
            gpu_frames.push(frame_tensor);

            // Use GPU tensor directly for embedding lookup (avoids 15 CPU→GPU transfers)
            let acoustic_embed_sum = self
                .code_predictor
                .get_acoustic_embeddings_sum_from_tensor(&acoustic_codes_tensor)?;
            let summed = semantic_embed.add(&acoustic_embed_sum)?;

            let text_addition = if frame_idx < trailing_text_len {
                trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            #[cfg(feature = "profiling")]
            let _talker_span = tracing::info_span!("talker_step", frame = frame_idx).entered();

            let (h, new_logits) =
                self.talker
                    .generate_step_with_embed(&step_input, kv_caches, offset)?;
            offset += 1;
            last_hidden = h;

            #[cfg(feature = "profiling")]
            sync_device(&self.device)?;
            #[cfg(feature = "profiling")]
            drop(_talker_span);

            #[cfg(feature = "profiling")]
            let _sample_span = tracing::info_span!("sampling", frame = frame_idx).entered();

            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.apply_generation_penalties_gpu(
                &logits_2d,
                &penalty_mask,
                gen_config,
                frame_idx + 1,
                Some(&suppression_mask),
            )?;
            semantic_token_tensor = generation::sample(&logits_2d, gen_config, sampling_ctx)?;
            tracing::trace!(target: "gpu_sync", "to_vec1 in generate_codes sampling");
            semantic_token = semantic_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
            Self::update_penalty_mask(&mut penalty_mask, semantic_token, vocab_size)?;
        }

        // Single GPU→CPU transfer: convert all accumulated GPU frames to FrameCodes
        self.gpu_frames_to_frame_codes(&gpu_frames)
    }

    /// Update the GPU-side penalty mask for a single token ID.
    ///
    /// Sets `penalty_mask[0, token_id] = 1.0` using slice_assign with a
    /// pre-built scalar. This is O(1) CPU work (no GPU→CPU transfer).
    fn update_penalty_mask(
        penalty_mask: &mut Tensor,
        token_id: u32,
        vocab_size: usize,
    ) -> Result<()> {
        let idx = token_id as usize;
        if idx < vocab_size {
            let one = Tensor::ones((1, 1), DType::F32, penalty_mask.device())?;
            *penalty_mask = penalty_mask.slice_assign(&[0..1, idx..idx + 1], &one)?;
        }
        Ok(())
    }

    /// Convert accumulated GPU frame tensors to FrameCodes via a single bulk transfer.
    fn gpu_frames_to_frame_codes(&self, gpu_frames: &[Tensor]) -> Result<FrameCodes> {
        if gpu_frames.is_empty() {
            return Ok(Vec::new());
        }
        // Stack all frames into [n_frames, 16], then single to_vec1
        let stacked = Tensor::stack(gpu_frames, 0)?; // [n_frames, 16]
        let n_frames = stacked.dim(0)?;
        let flat: Vec<u32> = stacked.flatten_all()?.to_vec1()?;
        let mut result = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let start = f * 16;
            result.push(flat[start..start + 16].to_vec());
        }
        Ok(result)
    }

    /// Synthesize speech with a specific voice and language.
    ///
    /// Uses the correct generation loop: CustomVoice prefill, autoregressive
    /// semantic tokens, per-frame acoustic code prediction via CodePredictor,
    /// residual VQ summation, and trailing text fusion.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `speaker` - Predefined speaker voice
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, etc.)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qwen3_tts::{Qwen3TTS, Speaker, Language, SynthesisOptions};
    ///
    /// let audio = model.synthesize_with_voice(
    ///     "Hello, world!",
    ///     Speaker::Ryan,
    ///     Language::English,
    ///     None,
    /// )?;
    /// audio.save("output.wav")?;
    /// ```
    pub fn synthesize_with_voice(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        if let Some(ModelType::Base) = &self.model_type {
            tracing::warn!(
                "Using preset speaker {:?} on a Base model. Base models are trained for \
                 voice cloning, not preset speakers — output will have an unpredictable voice. \
                 Use synthesize_voice_clone() with reference audio instead.",
                speaker
            );
        } else if let Some(ModelType::VoiceDesign) = &self.model_type {
            tracing::warn!(
                "Using preset speaker {:?} on a VoiceDesign model. VoiceDesign models \
                 are trained for text-described voice creation, not preset speakers.",
                speaker
            );
        }

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;

        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // Prefill with CustomVoice format
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        let mut kv_caches = self.talker.new_kv_caches(gen_config.max_new_tokens + 256);
        let (hidden, logits) =
            self.talker
                .prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        // Decode to audio
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        self.decode_codes(&all_codes)
    }

    /// Synthesize speech using a text-described voice (VoiceDesign model).
    ///
    /// Uses the same generation loop as [`Self::synthesize_with_voice`] but runs the
    /// VoiceDesign prefill instead of the predefined-speaker prefill. The voice
    /// is conditioned on a natural language description (e.g., "A cheerful young
    /// female voice with high pitch and energetic tone").
    ///
    /// The instruct text is tokenized with ChatML framing:
    /// `<|im_start|>user\n{instruct}<|im_end|>\n`
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `instruct` - Natural language voice description
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, etc.)
    pub fn synthesize_voice_design(
        &self,
        text: &str,
        instruct: &str,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        if let Some(ref mt) = self.model_type {
            if *mt != ModelType::VoiceDesign {
                tracing::warn!(
                    "Using VoiceDesign synthesis on a {:?} model. This model was not trained \
                     for text-described voice conditioning — output may be unpredictable.",
                    mt
                );
            }
        }

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;

        // Tokenize instruct with ChatML user framing: <|im_start|>user\n{instruct}<|im_end|>\n
        let instruct_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let instruct_ids = self.text_tokenizer.encode(&instruct_text)?;

        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // Prefill with VoiceDesign format
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        let mut kv_caches = self.talker.new_kv_caches(gen_config.max_new_tokens + 256);
        let (hidden, logits) = self.talker.prefill_voice_design(
            &input_ids,
            &instruct_ids,
            language,
            &mut kv_caches,
        )?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        // Decode to audio
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        self.decode_codes(&all_codes)
    }

    /// Convert list of frame codes to tensor [batch, 16, num_frames]
    pub fn codes_to_tensor(&self, codes: &[Vec<u32>]) -> Result<Tensor> {
        codes_to_tensor(codes, &self.device)
    }

    /// Decode raw frame codes to audio.
    ///
    /// Takes a slice of frames (each frame is a `Vec<u32>` of 16 codebook values)
    /// and runs the 12Hz decoder to produce an audio waveform at 24kHz.
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> Result<AudioBuffer> {
        let tensor = self.codes_to_tensor(codes)?;
        self.decode_tensor(&tensor)
    }

    /// Decode a codes tensor `[1, 16, T]` to audio.
    fn decode_tensor(&self, codes: &Tensor) -> Result<AudioBuffer> {
        let waveform = self.decoder.decode(codes)?;
        AudioBuffer::from_tensor(waveform, 24000)
    }

    /// Run the voice-clone prefill, including the ICL extension.
    ///
    /// Extracted so batch synthesis and the streaming session cannot drift: ICL
    /// mode does more than swap the prefill call. It raises the repetition
    /// penalty to the ICL floor, recomputes `max_new_tokens` from the text
    /// length, runs one extra pass of every layer over the ICL prompt, and moves
    /// both `offset` and `last_hidden` past that pass. A streaming constructor
    /// that copied only the prefill would keep the stale hidden state and
    /// produce audio that sounds plausible but is measurably worse.
    fn prepare_voice_clone(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: &SynthesisOptions,
    ) -> Result<VoiceClonePrefill> {
        let input_ids = self.text_tokenizer.encode(text)?;

        // Determine if ICL mode is active (ref_codes + ref_text present)
        let is_icl = prompt.ref_codes.is_some() && prompt.ref_text_ids.is_some();

        // ICL mode adjustments (matching mlx-audio):
        let repetition_penalty = if is_icl {
            options.repetition_penalty.max(ICL_MIN_REPETITION_PENALTY)
        } else {
            options.repetition_penalty
        };
        let max_new_tokens = if is_icl {
            options
                .max_length
                .min(ICL_MIN_FRAMES.max(input_ids.len() * ICL_FRAMES_PER_TOKEN))
        } else {
            options.max_length
        };
        let mut gen_config = options.to_gen_config();
        gen_config.max_new_tokens = max_new_tokens;
        gen_config.repetition_penalty = repetition_penalty;

        // Cast speaker embedding to compute dtype (speaker encoder produces F32)
        let speaker_embed = prompt.speaker_embedding.to_dtype(self.compute_dtype)?;

        // Voice clone prefill (9 positions for ICL, 10 for x_vector_only)
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        let mut kv_caches = self.talker.new_kv_caches(gen_config.max_new_tokens + 256);
        let (hidden, logits) = self.talker.prefill_voice_clone(
            &input_ids,
            &speaker_embed,
            language,
            is_icl,
            &mut kv_caches,
        )?;
        let prefill_len = hidden.dim(1)?;
        let mut offset = prefill_len;

        // Initialize last_hidden from prefill; updated by ICL block if active.
        let mut last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        // ICL extension (if reference codes + text are provided)
        let (trailing_text_hidden, logits) = if let (Some(ref_codes), Some(ref_text_ids)) =
            (&prompt.ref_codes, &prompt.ref_text_ids)
        {
            let ref_codec_embeds = self.sum_ref_codec_embeddings(ref_codes)?;

            // In ICL mode, all text tokens go into the ICL prompt (Python:
            // text_id=input_id[:, 3:-5] passes ALL target text tokens).
            // In the non-ICL path the first text token is consumed by the prefill,
            // so only the remaining tokens go to trailing_text.
            let (icl_embed, icl_trailing) =
                self.talker
                    .build_icl_prompt(&input_ids, ref_text_ids, &ref_codec_embeds, false)?;

            let icl_len = icl_embed.dim(1)?;
            if icl_len > 0 {
                let mask = models::transformer::create_causal_mask(icl_len, offset, &self.device)?;

                let mut icl_hidden = icl_embed;
                for (i, layer) in self.talker.layers_iter().enumerate() {
                    icl_hidden = layer.forward(
                        &icl_hidden,
                        self.talker.rope(),
                        Some(&mask),
                        Some(&mut kv_caches[i]),
                        offset,
                    )?;
                }
                icl_hidden = self.talker.apply_norm(&icl_hidden)?;
                offset += icl_len;

                let last_icl_hidden = icl_hidden.i((.., icl_len - 1..icl_len, ..))?;
                let new_logits = self.talker.apply_codec_head(&last_icl_hidden)?;

                // Update last_hidden so the code predictor is conditioned on
                // the ICL context, not the stale prefill hidden state.
                last_hidden = last_icl_hidden;

                (icl_trailing, new_logits)
            } else {
                let trailing = self.build_default_trailing_text(&input_ids)?;
                (trailing, logits)
            }
        } else {
            let trailing = self.build_default_trailing_text(&input_ids)?;
            (trailing, logits)
        };

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;

        Ok(VoiceClonePrefill {
            config: gen_config,
            kv_caches,
            offset,
            last_hidden,
            logits,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
        })
    }

    /// Synthesize speech using a cloned voice, returning raw codes alongside audio.
    ///
    /// Identical to [`synthesize_voice_clone`](Self::synthesize_voice_clone) but also
    /// returns the raw generated codes (`Vec<Vec<u32>>`) for debugging.
    /// Each inner `Vec<u32>` is one frame: `[semantic, acoustic_0..14]` (16 values).
    pub fn synthesize_voice_clone_debug(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<(AudioBuffer, FrameCodes)> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);

        let VoiceClonePrefill {
            config: gen_config,
            mut kv_caches,
            offset,
            last_hidden,
            logits,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
        } = self.prepare_voice_clone(text, prompt, language, &options)?;

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        // Prepend ref_codes for ICL decoder context (same fix as synthesize_voice_clone)
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        let audio = if let Some(ref_codes) = &prompt.ref_codes {
            let ref_frames = self.tensor_to_frame_codes(ref_codes)?;
            let ref_len = ref_frames.len();
            let mut combined = ref_frames;
            combined.extend(all_codes.iter().cloned());

            let mut audio = self.decode_codes(&combined)?;
            let total_frames = combined.len();
            // Proportional cut: matches official Qwen3-TTS Python implementation
            // cut = ref_len / total_len * wav.shape[0]
            let cut_samples = ref_len * audio.len() / total_frames.max(1);
            tracing::debug!(
                "ICL decode: ref_frames={}, gen_frames={}, total_samples={}, cut_samples={}",
                ref_len,
                all_codes.len(),
                audio.len(),
                cut_samples,
            );
            audio.samples = audio.samples[cut_samples.min(audio.len())..].to_vec();
            audio
        } else {
            self.decode_codes(&all_codes)?
        };
        Ok((audio, all_codes))
    }

    /// Get the device this model is running on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Create a streaming synthesis session with a specific voice and language.
    ///
    /// Returns an iterator that yields audio chunks as they are generated.
    /// Each chunk contains approximately `chunk_frames` frames worth of audio
    /// (default: 10 frames = ~800ms at 12.5 Hz frame rate).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qwen3_tts::{Qwen3TTS, Speaker, Language, SynthesisOptions};
    ///
    /// let options = SynthesisOptions::default();
    /// for chunk in model.synthesize_streaming("Hello!", Speaker::Ryan, Language::English, options)? {
    ///     let audio = chunk?;
    ///     // Play or process audio chunk (each ~800ms)
    /// }
    /// ```
    pub fn synthesize_streaming(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        let input_ids = self.text_tokenizer.encode(text)?;
        StreamingSession::new(self, &input_ids, speaker, language, options)
    }

    /// Synthesize speech using a text-described voice (VoiceDesign model), streaming.
    ///
    /// Same as [`Self::synthesize_voice_design`] but returns a streaming session
    /// that yields audio chunks as they are generated.
    ///
    /// The instruct text is tokenized with ChatML framing:
    /// `<|im_start|>user\n{instruct}<|im_end|>\n`
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `instruct` - Natural language voice description (e.g., "A cheerful young female voice")
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, chunk_frames, etc.)
    pub fn synthesize_voice_design_streaming(
        &self,
        text: &str,
        instruct: &str,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        if let Some(ref mt) = self.model_type {
            if *mt != ModelType::VoiceDesign {
                tracing::warn!(
                    "Using VoiceDesign synthesis on a {:?} model. This model was not trained \
                     for text-described voice conditioning — output may be unpredictable.",
                    mt
                );
            }
        }

        let input_ids = self.text_tokenizer.encode(text)?;

        // Tokenize instruct with ChatML user framing: <|im_start|>user\n{instruct}<|im_end|>\n
        let instruct_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let instruct_ids = self.text_tokenizer.encode(&instruct_text)?;

        StreamingSession::new_voice_design(self, &input_ids, &instruct_ids, language, options)
    }

    // ── Voice cloning API ─────────────────────────────────────────────────

    /// Create a voice clone prompt from reference audio.
    ///
    /// When `ref_text` is `None`, produces an **x_vector_only** prompt (speaker
    /// embedding only). When `Some`, produces an **ICL** prompt (speaker embedding
    /// + reference audio codes + reference text) — requires a speech encoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the speaker encoder is not loaded.
    pub fn create_voice_clone_prompt(
        &self,
        ref_audio: &AudioBuffer,
        ref_text: Option<&str>,
    ) -> Result<VoiceClonePrompt> {
        let encoder = self.speaker_encoder.as_ref().ok_or_else(|| {
            let hint = match &self.model_type {
                Some(ModelType::CustomVoice) => {
                    " CustomVoice models use preset speakers (synthesize_with_voice), \
                     not voice cloning. Use a Base model for voice cloning."
                }
                Some(ModelType::VoiceDesign) => {
                    " VoiceDesign models use text-described voices, not voice cloning. \
                     Use a Base model for voice cloning."
                }
                _ => {
                    " Ensure model weights contain `speaker_encoder.*` keys \
                     (only Base models include a speaker encoder)."
                }
            };
            anyhow::anyhow!("Speaker encoder not available.{}", hint)
        })?;

        // Resample to 24kHz if needed — both encoders assume 24kHz input
        let ref_audio_24k;
        let ref_audio = if ref_audio.sample_rate != 24000 {
            tracing::info!(
                "Resampling reference audio from {}Hz to 24000Hz",
                ref_audio.sample_rate
            );
            ref_audio_24k = audio::resample_to_24k(ref_audio)?;
            &ref_audio_24k
        } else {
            ref_audio
        };

        let speaker_embedding = encoder.encode(ref_audio)?; // [enc_dim]

        // ICL data: encode reference audio to codes and tokenize reference text
        let (ref_codes, ref_text_ids) = if let Some(text) = ref_text {
            let speech_enc = self.speech_encoder.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "ICL voice cloning requires a speech encoder, but it was not loaded. \
                     Ensure the speech tokenizer weights contain encoder keys, or use \
                     x_vector_only mode by passing ref_text=None."
                )
            })?;

            let codes = speech_enc.encode(ref_audio)?; // [T_frames, 16]
            let text_ids = self.text_tokenizer.encode(text)?;

            (Some(codes), Some(text_ids))
        } else {
            (None, None)
        };

        Ok(VoiceClonePrompt {
            speaker_embedding,
            ref_codes,
            ref_text_ids,
        })
    }

    /// Synthesize speech using a cloned voice.
    ///
    /// Uses the same generation loop as [`Self::synthesize_with_voice`] but runs the
    /// voice-clone prefill instead of the predefined-speaker prefill.
    ///
    /// When the prompt contains ICL data (ref_codes + ref_text_ids), the model
    /// is conditioned on reference audio/text to better reproduce the speaker's voice.
    pub fn synthesize_voice_clone(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        self.synthesize_voice_clone_debug(text, prompt, language, options)
            .map(|(audio, _codes)| audio)
    }

    /// Synthesize speech using a cloned voice, streaming.
    ///
    /// Same conditioning as [`Self::synthesize_voice_clone`] — including ICL when
    /// the prompt carries reference codes and text — but yields ~`chunk_frames`
    /// of audio at a time instead of returning only when the utterance is done.
    /// At the default 10 frames that is roughly 800 ms per chunk, which is what
    /// the caller waits for before the first sound.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `prompt` - Voice clone prompt from [`Self::create_voice_clone_prompt`]
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, chunk_frames, etc.)
    pub fn synthesize_voice_clone_streaming(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        StreamingSession::new_voice_clone(self, text, prompt, language, options)
    }

    /// Convert a ref_codes tensor `[T_frames, 16]` to `Vec<Vec<u32>>` frame format.
    fn tensor_to_frame_codes(&self, codes: &Tensor) -> Result<FrameCodes> {
        let (n_frames, n_codebooks) = codes.dims2()?;
        let codes_u32 = codes.to_dtype(DType::U32)?;
        let mut frames = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let frame_tensor = codes_u32.i(f)?; // [16]
            let frame_vec: Vec<u32> = frame_tensor.to_vec1()?;
            debug_assert_eq!(frame_vec.len(), n_codebooks);
            frames.push(frame_vec);
        }
        Ok(frames)
    }

    /// Sum reference codec embeddings across all 16 codebook groups.
    ///
    /// For each frame:
    /// - Group 0 (semantic): `talker.codec_embedding(ref_codes[:, 0])`
    /// - Groups 1–15 (acoustic): `code_predictor.codec_embeddings[i-1](ref_codes[:, i])`
    /// - Sum all 16 → single embedding per frame
    ///
    /// # Arguments
    /// * `ref_codes` — shape `[T_frames, 16]` of i64 codes
    ///
    /// # Returns
    /// Tensor of shape `[1, T_frames, hidden_size]`
    fn sum_ref_codec_embeddings(&self, ref_codes: &Tensor) -> Result<Tensor> {
        // Group 0: semantic codes → talker.codec_embedding
        let semantic_codes = ref_codes.i((.., 0))?; // [T_frames]
        let semantic_codes = semantic_codes.to_dtype(candle_core::DType::U32)?;
        let summed = self.talker.get_codec_embedding_batch(&semantic_codes)?; // [1, T, hidden]

        // Groups 1-15: acoustic codes → code_predictor.embed_codes_for_group
        let mut summed = summed;
        for group in 1..16 {
            let group_codes = ref_codes.i((.., group))?; // [T_frames]
            let group_codes = group_codes.to_dtype(candle_core::DType::U32)?;
            let group_embed = self
                .code_predictor
                .embed_codes_for_group(group - 1, &group_codes)?; // [1, T, embed_dim]
            summed = summed.add(&group_embed)?;
        }

        Ok(summed)
    }

    /// Build default trailing text embeddings (for non-ICL mode).
    fn build_default_trailing_text(&self, input_ids: &[u32]) -> Result<Tensor> {
        let (hidden, _len, _pad) = self.build_trailing_text(input_ids)?;
        Ok(hidden)
    }

    /// Apply repetition penalty, token suppression, and min_new_tokens EOS suppression
    /// using a pre-built `[1, vocab]` penalty mask on GPU.
    ///
    /// The mask is updated incrementally via [`update_penalty_mask`] after each
    /// sampled token, eliminating the O(n) GPU→CPU transfer that grows with
    /// each frame.
    fn apply_generation_penalties_gpu(
        &self,
        logits: &Tensor,
        penalty_mask: &Tensor,
        config: &generation::GenerationConfig,
        token_count: usize,
        suppression_mask: Option<&generation::SuppressionMask>,
    ) -> Result<Tensor> {
        let logits = logits.to_dtype(DType::F32)?;

        // 1. Repetition penalty via pre-built GPU mask
        let logits = if config.repetition_penalty != 1.0 {
            generation::apply_repetition_penalty_with_mask(
                &logits,
                penalty_mask,
                config.repetition_penalty,
            )?
        } else {
            logits
        };

        // 2. Token suppression
        let logits = if let Some(mask) = suppression_mask {
            generation::apply_token_suppression_with_mask(&logits, mask)?
        } else {
            generation::apply_token_suppression(
                &logits,
                codec_tokens::CODEC_VOCAB_SIZE,
                CODEC_EOS_TOKEN_ID,
            )?
        };

        // 3. Min new tokens EOS suppression
        if token_count < config.min_new_tokens {
            if let Some(eos_id) = config.eos_token_id {
                let vocab = logits.dim(1)?;
                let batch = logits.dim(0)?;
                let mut mask_data = vec![0.0f32; vocab];
                mask_data[eos_id as usize] = 1.0;
                let eos_mask = Tensor::new(mask_data.as_slice(), logits.device())?
                    .unsqueeze(0)?
                    .broadcast_as((batch, vocab))?;
                let neg_inf = Tensor::new(&[f32::NEG_INFINITY], logits.device())?
                    .broadcast_as((batch, vocab))?;
                let zeros = Tensor::zeros((batch, vocab), DType::F32, logits.device())?;
                let is_eos = eos_mask.gt(&zeros)?;
                return Ok(is_eos.where_cond(&neg_inf, &logits)?);
            }
        }

        Ok(logits)
    }

    /// Returns `true` if a speech encoder is loaded (ICL voice cloning is available).
    pub fn has_speech_encoder(&self) -> bool {
        self.speech_encoder.is_some()
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Attempt to load the speaker encoder from model weights.
    ///
    /// Returns `Ok(Some(encoder))` if `speaker_encoder.*` keys are found,
    /// `Ok(None)` if they are absent. When `config` is provided, uses the
    /// parsed enc_dim; otherwise falls back to defaults (enc_dim=1024).
    fn try_load_speaker_encoder(
        weights: &HashMap<String, Tensor>,
        config: Option<&SpeakerEncoderConfig>,
        device: &Device,
    ) -> Result<Option<SpeakerEncoder>> {
        let has_se_weights = weights.keys().any(|k| k.starts_with("speaker_encoder."));
        if !has_se_weights {
            return Ok(None);
        }

        let config = config.cloned().unwrap_or_default();
        tracing::info!(
            "Loading speaker encoder (ECAPA-TDNN, enc_dim={}) for voice cloning...",
            config.enc_dim
        );
        let se_weights = Self::filter_weights(weights, "speaker_encoder.");
        let se_vb = candle_nn::VarBuilder::from_tensors(se_weights, DType::F32, device);
        let encoder = SpeakerEncoder::new(config, se_vb)?;
        Ok(Some(encoder))
    }

    /// Attempt to load the speech encoder (Mimi) from speech tokenizer weights.
    ///
    /// The speech encoder encodes raw audio to 12Hz codec codes, needed for
    /// ICL voice cloning. Returns `Ok(None)` if encoder keys are absent or
    /// loading fails (non-fatal — ICL mode just won't be available).
    fn try_load_speech_encoder(
        weights: &HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Option<Encoder12Hz>> {
        // Check for encoder-related keys (either HF or candle format)
        let has_encoder_keys = weights
            .keys()
            .any(|k| k.starts_with("encoder.") || k.starts_with("encoder_transformer."));
        if !has_encoder_keys {
            return Ok(None);
        }

        tracing::debug!("Attempting to load speech encoder (Mimi) for ICL voice cloning...");
        match Encoder12Hz::from_weights(weights, device) {
            Ok(enc) => {
                tracing::info!("Loaded speech encoder — ICL voice cloning available");
                Ok(Some(enc))
            }
            Err(e) => {
                tracing::debug!(
                    "Speech encoder not available ({}). ICL voice cloning disabled.",
                    e
                );
                Ok(None)
            }
        }
    }

    /// Load weights from safetensors file.
    ///
    /// Tensors are loaded in their native dtype (typically BF16 for Qwen3-TTS).
    /// Each component's VarBuilder handles casting to its target dtype.
    fn load_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
        Ok(candle_core::safetensors::load(path, device)?)
    }

    /// Filter weights by prefix, removing the prefix from keys.
    pub(crate) fn filter_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> HashMap<String, Tensor> {
        weights
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix(prefix)
                    .map(|stripped| (stripped.to_string(), v.clone()))
            })
            .collect()
    }
}

/// Convert a slice of codec frames into a tensor of shape `[1, 16, T]`.
///
/// Each frame must contain exactly 16 codebook values. The output layout is
/// `[q0_f0, q0_f1, ...], [q1_f0, q1_f1, ...]` matching the decoder's expectation.
pub fn codes_to_tensor(codes: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let num_frames = codes.len();
    if num_frames == 0 {
        return Ok(Tensor::zeros((1, 16, 0), DType::I64, device)?);
    }

    let mut data = vec![0i64; 16 * num_frames];
    for (frame, frame_codes) in codes.iter().enumerate() {
        for (q, &code) in frame_codes.iter().enumerate() {
            data[q * num_frames + frame] = code as i64;
        }
    }

    Ok(Tensor::from_vec(data, (1, 16, num_frames), device)?)
}

/// Return the recommended compute dtype for the given device.
///
/// Returns `BF16` for CUDA/Metal (lower memory, faster attention) and `F32` for CPU.
///
/// `QWEN3_TTS_DTYPE` (`f16` | `bf16` | `f32`) overrides the choice. It exists
/// because the right answer is hardware-dependent and cheap to measure: the
/// generation loop is dominated by many tiny matmuls, where which dtype hits the
/// backend's fast path matters more than how much memory it saves.
pub fn compute_dtype_for_device(device: &Device) -> DType {
    if let Ok(name) = std::env::var("QWEN3_TTS_DTYPE") {
        match name.to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "half" => return DType::F16,
            "bf16" | "bfloat16" => return DType::BF16,
            "f32" | "fp32" | "float" => return DType::F32,
            other => tracing::warn!("QWEN3_TTS_DTYPE={other} not recognised — using default"),
        }
    }
    if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Force the GPU to complete all pending work before returning.
///
/// On CUDA/Metal, GPU operations are asynchronous — `Instant::now()` would
/// measure submission time, not completion time. This helper forces a sync
/// by creating a tiny tensor and reading it back to the CPU.
///
/// On CPU this is a no-op.
pub fn sync_device(device: &Device) -> Result<()> {
    match device {
        Device::Cpu => Ok(()),
        _ => {
            // Force a GPU→CPU sync by reading a scalar back
            let _: Vec<f32> = Tensor::zeros(1, DType::F32, device)?.to_vec1()?;
            Ok(())
        }
    }
}

/// The codec end-of-sequence token ID (2150).
///
/// Generation stops when this token is sampled. This is in the codec vocabulary
/// `[0, 3072)`, not the text vocabulary.
pub const CODEC_EOS_TOKEN_ID: u32 = codec_tokens::CODEC_EOS;

/// Number of audio samples per codec frame at 24kHz (1920 = 80ms per frame at 12Hz).
pub const SAMPLES_PER_FRAME: usize = 1920;

/// ICL mode: minimum frames to generate regardless of text length (matching mlx-audio)
const ICL_MIN_FRAMES: usize = 75;

/// ICL mode: estimated frames per input text token for max-length cap (matching mlx-audio)
const ICL_FRAMES_PER_TOKEN: usize = 6;

/// ICL mode: minimum repetition penalty to prevent degenerate loops (matching mlx-audio)
const ICL_MIN_REPETITION_PENALTY: f64 = 1.5;

/// Whether streaming may decode chunks on the CPU twin while Metal generates.
/// On by default; `QWEN3_TTS_OVERLAP_DECODE=0` disables (serial Metal decode).
fn overlap_decode_enabled() -> bool {
    std::env::var("QWEN3_TTS_OVERLAP_DECODE").as_deref() != Ok("0")
}

/// Frames of left context prepended to every decoded chunk after the first.
///
/// Measured against the batch waveform (streaming must converge to it —
/// identical codes, only the decode windows differ): rms divergence 0.37 with
/// no context, 0.19 with 4 frames, 0.17 with 8. Eight is past the knee; more
/// costs decode time linearly for little gain. `QWEN3_TTS_DECODE_CONTEXT`
/// overrides.
const DECODE_CONTEXT_FRAMES: usize = 8;

fn decode_context_frames() -> usize {
    std::env::var("QWEN3_TTS_DECODE_CONTEXT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DECODE_CONTEXT_FRAMES)
}

/// Trailing ICL reference frames given to the decoder on the first chunk.
///
/// The batch path hands it the whole reference — tens of frames — and pays for
/// decoding all of them to produce the first ten frames of speech. Measured on
/// the tts-lab bench (2026-08-28, 26 phrases, interleaved A/B, twice): dropping
/// to four frames moves time-to-first-audio from 1640/1787 ms to 1015/946 and
/// RTF from 1.262/1.371 to 1.173/1.101, with intelligibility unchanged at
/// 3.90%. Four rather than zero because the decoder is convolutional and a few
/// frames of lead-in cost nothing while a hard start might click.
const ICL_DECODE_FRAMES: usize = 4;

/// Trailing ICL reference frames for the decoder; `None` = all of them.
///
/// `QWEN3_TTS_ICL_DECODE_FRAMES` overrides [`ICL_DECODE_FRAMES`]; `full` restores
/// batch-path parity.
fn icl_decode_frames() -> Option<usize> {
    match std::env::var("QWEN3_TTS_ICL_DECODE_FRAMES") {
        Ok(v) if v.eq_ignore_ascii_case("full") => None,
        Ok(v) => v.parse::<usize>().ok().or(Some(ICL_DECODE_FRAMES)),
        Err(_) => Some(ICL_DECODE_FRAMES),
    }
}

/// Streaming synthesis session.
///
/// Yields audio chunks as they are generated. Use with
/// [`Qwen3TTS::synthesize_streaming`].
pub struct StreamingSession<'a> {
    model: &'a Qwen3TTS,
    config: generation::GenerationConfig,
    sampling_ctx: generation::SamplingContext,
    kv_caches: Vec<AnyKVCache>,
    offset: usize,
    last_hidden: Tensor,
    current_token: Option<u32>,
    current_token_tensor: Option<Tensor>,
    frames_generated: usize,
    /// Generated frames as device tensors (`[16]` each), never read back per frame.
    frame_buffer: Vec<Tensor>,
    chunk_frames: usize,
    done: bool,
    // Trailing text state for residual VQ + text fusion
    trailing_text_hidden: Tensor,
    trailing_text_len: usize,
    tts_pad_embed: Tensor,
    // GPU-side repetition penalty mask [1, vocab] — updated incrementally
    penalty_mask: Tensor,
    token_count: usize,
    // Pre-built suppression mask (reused every frame)
    suppression_mask: generation::SuppressionMask,
    // Pre-allocated code predictor KV caches (reused + reset each frame)
    cp_kv_caches: Vec<AnyKVCache>,
    /// Left context for the decoder: codes `[K, 16]` prepended to the chunk
    /// being decoded and cut back off the audio. Starts as the tail of the ICL
    /// reference, then becomes the tail of each decoded chunk. Without it every
    /// chunk decodes in isolation and the seams are audible — the decoder is
    /// convolutional, and a chunk edge with no left context is a discontinuity
    /// every 800 ms (reported by ear, 2026-08-29; confirmed against the batch
    /// waveform, which streaming must match sample-for-sample).
    decoder_context: Option<Tensor>,
}

impl<'a> StreamingSession<'a> {
    fn new(
        model: &'a Qwen3TTS,
        input_ids: &[u32],
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        let sampling_ctx = generation::SamplingContext::new(options.seed);
        let config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            model.build_trailing_text(input_ids)?;

        let mut kv_caches = model.talker.new_kv_caches(config.max_new_tokens + 256);
        let prefill_result =
            model
                .talker
                .prefill_custom_voice(input_ids, speaker, language, &mut kv_caches)?;

        Self::from_prefill(
            model,
            config,
            sampling_ctx,
            kv_caches,
            prefill_result,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            options.chunk_frames,
        )
    }

    /// Create a streaming session using voice design (text-described voice).
    ///
    /// Uses `prefill_voice_design` instead of `prefill_custom_voice` to condition
    /// on a natural language voice description rather than a predefined speaker.
    fn new_voice_design(
        model: &'a Qwen3TTS,
        input_ids: &[u32],
        instruct_ids: &[u32],
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        let sampling_ctx = generation::SamplingContext::new(options.seed);
        let config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            model.build_trailing_text(input_ids)?;

        let mut kv_caches = model.talker.new_kv_caches(config.max_new_tokens + 256);
        let prefill_result =
            model
                .talker
                .prefill_voice_design(input_ids, instruct_ids, language, &mut kv_caches)?;

        Self::from_prefill(
            model,
            config,
            sampling_ctx,
            kv_caches,
            prefill_result,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            options.chunk_frames,
        )
    }

    /// Create a streaming session for a cloned voice.
    ///
    /// Delegates the whole prefill to [`Qwen3TTS::prepare_voice_clone`], the same
    /// helper the batch path uses, so ICL mode gets its raised repetition penalty,
    /// its text-derived length cap and — the easy one to lose — the `offset` and
    /// `last_hidden` that sit *after* the ICL pass, not after the plain prefill.
    fn new_voice_clone(
        model: &'a Qwen3TTS,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        let sampling_ctx = generation::SamplingContext::new(options.seed);
        let prep = model.prepare_voice_clone(text, prompt, language, &options)?;

        // Reference codes give the codec decoder the same context the batch path
        // gives it, and are cut back off after decoding. Only the first chunk
        // needs them: from the second chunk on the decoder is already inside the
        // generated speech.
        //
        // They are not free: the reference runs tens of frames, and decoding them
        // alongside the first ten is 495 ms of the 1291 ms before the listener
        // hears anything (measured 2026-08-28). `QWEN3_TTS_ICL_DECODE_FRAMES`
        // keeps only the last N — the decoder is convolutional, so its context
        // need is local, and the far end of the reference cannot matter.
        let icl_decoder_context = match prompt.ref_codes.as_ref() {
            Some(codes) => match icl_decode_frames() {
                Some(0) => None,
                Some(n) => {
                    let total = codes.dim(0)?;
                    Some(codes.narrow(0, total.saturating_sub(n), n.min(total))?)
                }
                None => Some(codes.clone()),
            },
            None => None,
        };

        let mut session = Self::from_state(
            model,
            prep.config,
            sampling_ctx,
            prep.kv_caches,
            prep.offset,
            prep.last_hidden,
            &prep.logits,
            prep.trailing_text_hidden,
            prep.trailing_text_len,
            prep.tts_pad_embed,
            options.chunk_frames,
        )?;
        session.decoder_context = icl_decoder_context;
        Ok(session)
    }

    /// Shared post-prefill constructor.
    ///
    /// Extracts `last_hidden` from the prefill result, builds the suppression and
    /// penalty masks, samples the first semantic token, and assembles the session.
    #[allow(clippy::too_many_arguments)]
    fn from_prefill(
        model: &'a Qwen3TTS,
        config: generation::GenerationConfig,
        sampling_ctx: generation::SamplingContext,
        kv_caches: Vec<AnyKVCache>,
        prefill_result: (Tensor, Tensor),
        trailing_text_hidden: Tensor,
        trailing_text_len: usize,
        tts_pad_embed: Tensor,
        chunk_frames: usize,
    ) -> Result<Self> {
        let (hidden, logits) = prefill_result;
        let prefill_len = hidden.dim(1)?;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        Self::from_state(
            model,
            config,
            sampling_ctx,
            kv_caches,
            prefill_len,
            last_hidden,
            &logits,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            chunk_frames,
        )
    }

    /// Assemble a session from an explicit decode position and hidden state.
    ///
    /// Split out from [`Self::from_prefill`] because voice cloning does not start
    /// generating where its prefill ended: the ICL pass moves both `offset` and
    /// `last_hidden` forward, and deriving them from the prefill tensor — as
    /// `from_prefill` does — would silently rewind that.
    #[allow(clippy::too_many_arguments)]
    fn from_state(
        model: &'a Qwen3TTS,
        config: generation::GenerationConfig,
        mut sampling_ctx: generation::SamplingContext,
        kv_caches: Vec<AnyKVCache>,
        offset: usize,
        last_hidden: Tensor,
        logits: &Tensor,
        trailing_text_hidden: Tensor,
        trailing_text_len: usize,
        tts_pad_embed: Tensor,
        chunk_frames: usize,
    ) -> Result<Self> {
        // Build suppression mask once for reuse across all frames
        let suppression_mask = generation::build_suppression_mask(
            codec_tokens::CODEC_VOCAB_SIZE,
            CODEC_EOS_TOKEN_ID,
            &model.device,
        )?;

        // Sample first token with full penalty pipeline
        let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;
        let mut penalty_mask = Tensor::zeros((1, vocab_size), DType::F32, &model.device)?;
        let logits_2d = logits.squeeze(1)?;
        let logits_2d = model.apply_generation_penalties_gpu(
            &logits_2d,
            &penalty_mask,
            &config,
            0,
            Some(&suppression_mask),
        )?;
        let first_token = generation::sample(&logits_2d, &config, &mut sampling_ctx)?;
        let first_token_id: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
        Qwen3TTS::update_penalty_mask(&mut penalty_mask, first_token_id, vocab_size)?;

        let done = config.eos_token_id == Some(first_token_id);
        let cp_kv_caches = model.code_predictor.new_kv_caches();

        Ok(Self {
            model,
            config,
            sampling_ctx,
            kv_caches,
            offset,
            last_hidden,
            current_token: if done { None } else { Some(first_token_id) },
            current_token_tensor: if done { None } else { Some(first_token) },
            frames_generated: 0,
            frame_buffer: Vec::new(),
            chunk_frames,
            done,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            penalty_mask,
            token_count: 1,
            suppression_mask,
            cp_kv_caches,
            decoder_context: None,
        })
    }

    /// Decode the buffered frames, giving the decoder ICL reference context on
    /// the first chunk and trimming that context back out of the audio.
    ///
    /// Frames never leave the device on their way here: they are accumulated as
    /// GPU tensors and stacked into the decoder's `[1, 16, T]` layout directly.
    /// The obvious version — read each frame back, rebuild a tensor from the
    /// CPU vectors — costs a device sync per frame, which on Metal is a pipeline
    /// stall, not a memcpy.
    fn take_codes(&mut self, for_cpu: bool) -> Result<DecodeJob> {
        let stacked = Tensor::stack(&self.frame_buffer, 0)?.to_dtype(DType::I64)?; // [T, 16]
        self.frame_buffer.clear();
        let chunk_len = stacked.dim(0)?;

        let context = self.decoder_context.take();

        // This chunk's tail is the next chunk's left context. Taken from the
        // raw chunk, not the padded one, so the context is always real speech
        // adjacent to the seam.
        let keep = decode_context_frames().min(chunk_len);
        if keep > 0 {
            self.decoder_context = Some(stacked.narrow(0, chunk_len - keep, keep)?);
        }

        let (codes, ref_len, total_frames) = match context {
            Some(ref_codes) => {
                let ref_codes = ref_codes.to_dtype(DType::I64)?;
                let ref_len = ref_codes.dim(0)?;
                let combined = Tensor::cat(&[&ref_codes, &stacked], 0)?;
                let total = combined.dim(0)?;
                (combined, ref_len, total)
            }
            None => (stacked, 0, chunk_len),
        };

        // The device hop happens HERE, on the generation thread. The worker
        // thread that runs the job then touches no Metal state at all — that
        // is what makes the overlap safe; the same overlap with the decode
        // thread reading from the Metal device corrupted generation (tts-lab,
        // 2026-08-29).
        let codes = if for_cpu {
            codes.to_device(&Device::Cpu)?
        } else {
            codes
        };

        Ok(DecodeJob {
            codes,
            ref_len,
            total_frames,
        })
    }
}

/// One chunk's worth of codec frames, ready for the decoder.
///
/// Borrows nothing from the session, so it can run on another thread while the
/// session generates the next chunk.
struct DecodeJob {
    /// `[T, 16]` on the decoder's device, reference frames first when present.
    codes: Tensor,
    ref_len: usize,
    total_frames: usize,
}

impl DecodeJob {
    /// Decode to audio and trim the ICL reference back off the front.
    ///
    /// Takes the decoder alone: `Qwen3TTS` is not `Sync` (the speech *encoder*
    /// holds a `RefCell`), while the decoder is plain weights.
    fn run(self, decoder: &Decoder12Hz) -> Result<AudioBuffer> {
        let DecodeJob {
            codes,
            ref_len,
            total_frames,
        } = self;

        // Measurement escape hatch: how much of a chunk is the decoder? Returns
        // silence of the right length so the loop is otherwise untouched.
        if std::env::var("QWEN3_TTS_SKIP_DECODE").as_deref() == Ok("1") {
            let frames = total_frames - ref_len;
            return Ok(AudioBuffer {
                samples: vec![0.0; frames * SAMPLES_PER_FRAME],
                sample_rate: 24000,
            });
        }

        // [T, 16] → [1, 16, T], the layout the 12 Hz decoder expects.
        let codes = codes.t()?.unsqueeze(0)?.contiguous()?;
        let audio = decoder.decode(&codes)?;
        let mut audio = AudioBuffer::from_tensor(audio, 24000)?;

        if ref_len > 0 {
            // Proportional cut, same rule as the batch path: the reference
            // occupies ref_len/total_frames of the decoded waveform.
            let cut_samples = ref_len * audio.len() / total_frames.max(1);
            audio.samples = audio.samples[cut_samples.min(audio.len())..].to_vec();
        }
        Ok(audio)
    }
}

impl<'a> StreamingSession<'a> {
    /// Generate the next chunk of audio.
    ///
    /// Returns `Some(AudioBuffer)` for each chunk, or `None` when generation is complete.
    ///
    /// Chunks 2..n are decoded on the CPU twin, on a worker thread, while this
    /// call already generates the *following* chunk; the audio returned is
    /// still this chunk's. The first chunk takes the fast path instead — Metal,
    /// serial, returned at once — because overlapping it would delay the first
    /// audio by a whole extra chunk of generation, and the first chunk is the
    /// one the listener is waiting on.
    pub fn next_chunk(&mut self) -> Result<Option<AudioBuffer>> {
        if self.frame_buffer.is_empty() {
            if self.done {
                return Ok(None);
            }
            self.generate_chunk()?;
            if self.frame_buffer.is_empty() {
                return Ok(None);
            }
        }

        let first_chunk = self.frames_generated <= self.chunk_frames;
        let cpu_decoder = self.model.cpu_decoder.as_ref();
        let overlap = !self.done && !first_chunk && cpu_decoder.is_some();

        let job = self.take_codes(overlap)?;
        if !overlap {
            return Ok(Some(job.run(&self.model.decoder)?));
        }
        let cpu_decoder: &'a Decoder12Hz = self.model.cpu_decoder.as_ref().unwrap();

        std::thread::scope(|scope| {
            let decoding = scope.spawn(move || job.run(cpu_decoder));
            let generated = self.generate_chunk();
            let audio = decoding
                .join()
                .map_err(|_| anyhow::anyhow!("decoder thread panicked"))??;
            generated?;
            Ok(Some(audio))
        })
    }

    /// Fill `frame_buffer` up to `chunk_frames`, or until EOS / the length cap.
    fn generate_chunk(&mut self) -> Result<()> {
        // Generate frames until we have enough for a chunk
        while self.frame_buffer.len() < self.chunk_frames
            && self.frames_generated < self.config.max_new_tokens
        {
            let Some(token_tensor) = self.current_token_tensor.take() else {
                self.done = true;
                break;
            };

            // Embedding lookup using GPU-resident token tensor (no CPU→GPU roundtrip)
            let semantic_embed = self
                .model
                .talker
                .get_codec_embedding_from_tensor(&token_tensor)?;

            // Generate 15 acoustic codes (stays on GPU)
            let acoustic_codes_tensor = self.model.code_predictor.generate_acoustic_codes(
                &self.last_hidden,
                &semantic_embed,
                &mut self.cp_kv_caches,
            )?;

            // Build the frame on GPU and keep it there — the sampled token is
            // already a device tensor, so re-uploading `token_id` and reading the
            // frame back would be two syncs per frame for data nobody looks at
            // until the chunk is decoded.
            let frame_tensor =
                Tensor::cat(&[&token_tensor.reshape(1)?, &acoustic_codes_tensor], 0)?;
            self.frame_buffer.push(frame_tensor);

            let frame_idx = self.frames_generated;
            self.frames_generated += 1;

            // Build residual VQ sum + trailing text for next step
            let acoustic_embed_sum = self
                .model
                .code_predictor
                .get_acoustic_embeddings_sum_from_tensor(&acoustic_codes_tensor)?;
            let summed = semantic_embed.add(&acoustic_embed_sum)?;

            let text_addition = if frame_idx < self.trailing_text_len {
                self.trailing_text_hidden
                    .i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                self.tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            // Run talker step with fused embedding
            let (h, new_logits) = self.model.talker.generate_step_with_embed(
                &step_input,
                &mut self.kv_caches,
                self.offset,
            )?;
            self.offset += 1;
            self.last_hidden = h;

            // Sample next semantic token with repetition penalty + token suppression + min_new_tokens
            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.model.apply_generation_penalties_gpu(
                &logits_2d,
                &self.penalty_mask,
                &self.config,
                self.token_count,
                Some(&self.suppression_mask),
            )?;
            let next_token_tensor =
                generation::sample(&logits_2d, &self.config, &mut self.sampling_ctx)?;
            let next_token_id: u32 = next_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
            Qwen3TTS::update_penalty_mask(
                &mut self.penalty_mask,
                next_token_id,
                codec_tokens::CODEC_VOCAB_SIZE,
            )?;
            self.token_count += 1;

            if self.config.eos_token_id == Some(next_token_id) {
                self.current_token = None;
                self.current_token_tensor = None;
                self.done = true;
            } else {
                self.current_token = Some(next_token_id);
                self.current_token_tensor = Some(next_token_tensor);
            }
        }

        Ok(())
    }

    /// Returns the total number of frames generated so far.
    pub fn frames_generated(&self) -> usize {
        self.frames_generated
    }

    /// Returns true if generation is complete.
    pub fn is_done(&self) -> bool {
        self.done && self.frame_buffer.is_empty()
    }
}

impl<'a> Iterator for StreamingSession<'a> {
    type Item = Result<AudioBuffer>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(audio)) => Some(Ok(audio)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Options for speech synthesis
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Maximum number of frames to generate
    pub max_length: usize,
    /// Sampling temperature (higher = more random)
    pub temperature: f64,
    /// Top-k sampling
    pub top_k: usize,
    /// Top-p (nucleus) sampling
    pub top_p: f64,
    /// Repetition penalty (1.0 = disabled, 1.05 = Python default)
    pub repetition_penalty: f64,
    /// End-of-sequence token ID (defaults to codec EOS token 2150)
    pub eos_token_id: Option<u32>,
    /// Frames per streaming chunk (default: 10 = ~800ms)
    pub chunk_frames: usize,
    /// Minimum tokens before EOS is allowed (default: 2, matching Python)
    pub min_new_tokens: usize,
    /// Random seed for deterministic generation. `None` = non-deterministic.
    pub seed: Option<u64>,
}

impl SynthesisOptions {
    /// Convert to a [`GenerationConfig`](generation::GenerationConfig) for the generation loop.
    pub fn to_gen_config(&self) -> generation::GenerationConfig {
        generation::GenerationConfig {
            max_new_tokens: self.max_length,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            eos_token_id: self.eos_token_id,
            min_new_tokens: self.min_new_tokens,
        }
    }
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_length: 2048,
            temperature: 0.9,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.05,
            eos_token_id: Some(CODEC_EOS_TOKEN_ID),
            chunk_frames: 10, // ~800ms per chunk at 12.5 Hz
            min_new_tokens: 2,
            seed: None,
        }
    }
}

/// Select the best available compute device for inference.
///
/// Checks for available hardware in order: CUDA → Metal → CPU.
/// Falls back to CPU if no GPU acceleration is available.
///
/// # Feature Flags
///
/// - `cuda`: Enables NVIDIA GPU support
/// - `metal`: Enables Apple Silicon GPU support
///
/// # Example
///
/// ```rust,ignore
/// let device = qwen3_tts::auto_device()?;
/// let model = Qwen3TTS::from_pretrained("path/to/model", device)?;
/// ```
pub fn auto_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::cuda_if_available(0) {
            if device.is_cuda() {
                tracing::info!("Using CUDA device");
                return Ok(device);
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            tracing::info!("Using Metal device");
            return Ok(device);
        }
    }

    tracing::info!("Using CPU device");
    Ok(Device::Cpu)
}

/// Parse a device string into a [`Device`].
///
/// Supported formats:
/// - `"auto"` — select best available via [`auto_device`]
/// - `"cpu"` — force CPU
/// - `"cuda"` or `"cuda:0"` — CUDA device 0
/// - `"cuda:N"` — CUDA device N
/// - `"metal"` — Apple Silicon GPU
///
/// # Errors
///
/// Returns an error if the device string is unrecognized, the requested
/// backend wasn't compiled in, or hardware initialization fails.
pub fn parse_device(device_str: &str) -> Result<Device> {
    match device_str.to_lowercase().as_str() {
        "auto" => auto_device(),
        "cpu" => Ok(Device::Cpu),
        s if s.starts_with("cuda") => {
            #[cfg(feature = "cuda")]
            {
                let ordinal: usize = if s == "cuda" {
                    0
                } else if let Some(idx) = s.strip_prefix("cuda:") {
                    idx.parse()
                        .map_err(|e| anyhow::anyhow!("invalid CUDA device index: {e}"))?
                } else {
                    0
                };
                Device::cuda_if_available(ordinal)
                    .map_err(|e| anyhow::anyhow!("failed to init CUDA device {ordinal}: {e}"))
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("CUDA support not compiled in. Rebuild with: cargo build --features cuda")
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)
                    .map_err(|e| anyhow::anyhow!("failed to init Metal device: {e}"))
            }
            #[cfg(not(feature = "metal"))]
            anyhow::bail!(
                "Metal support not compiled in. Rebuild with: cargo build --features metal"
            )
        }
        other => {
            anyhow::bail!("unknown device '{other}'. Supported: auto, cpu, cuda, cuda:N, metal")
        }
    }
}

/// Human-readable label for a [`Device`].
pub fn device_info(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => "CUDA".to_string(),
        Device::Metal(_) => "Metal".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_options_default() {
        let options = SynthesisOptions::default();
        assert_eq!(options.max_length, 2048);
        assert!((options.temperature - 0.9).abs() < 1e-6);
        assert_eq!(options.top_k, 50);
        assert!((options.top_p - 0.9).abs() < 1e-6);
        assert!((options.repetition_penalty - 1.05).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(CODEC_EOS_TOKEN_ID));
        assert_eq!(options.chunk_frames, 10);
        assert_eq!(options.min_new_tokens, 2);
    }

    #[test]
    fn test_synthesis_options_custom() {
        let options = SynthesisOptions {
            max_length: 512,
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
            repetition_penalty: 1.2,
            eos_token_id: Some(CODEC_EOS_TOKEN_ID),
            chunk_frames: 5,
            min_new_tokens: 0,
            seed: Some(42),
        };
        assert_eq!(options.max_length, 512);
        assert!((options.temperature - 0.5).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(CODEC_EOS_TOKEN_ID));
        assert_eq!(options.chunk_frames, 5);
    }

    #[test]
    fn test_synthesis_options_clone() {
        let options = SynthesisOptions::default();
        let cloned = options.clone();
        assert_eq!(cloned.max_length, options.max_length);
        assert_eq!(cloned.top_k, options.top_k);
    }

    #[test]
    fn test_synthesis_options_debug() {
        let options = SynthesisOptions::default();
        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("max_length"));
        assert!(debug_str.contains("2048"));
    }

    #[test]
    fn test_auto_device() {
        // Should always succeed on CPU
        let device = auto_device().unwrap();
        // Just verify it returns a valid device
        assert!(
            matches!(device, Device::Cpu)
                || matches!(device, Device::Cuda(_))
                || matches!(device, Device::Metal(_))
        );
    }

    #[test]
    fn test_audio_buffer_reexport() {
        // Verify re-exports work
        let buffer = AudioBuffer::new(vec![0.0f32; 100], 24000);
        assert_eq!(buffer.sample_rate, 24000);
    }

    #[test]
    fn test_config_reexport() {
        // Verify config re-export works
        let config = Qwen3TTSConfig::default();
        assert_eq!(config.model_type, "qwen3_tts");
    }

    #[test]
    fn test_codes_to_tensor_empty() {
        let device = Device::Cpu;
        let codes: Vec<Vec<u32>> = vec![];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 0]);
    }

    #[test]
    fn test_codes_to_tensor_single_frame() {
        let device = Device::Cpu;
        let codes = vec![vec![0u32; 16]];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 1]);
    }

    #[test]
    fn test_codes_to_tensor_layout() {
        let device = Device::Cpu;
        // 2 frames, each with 16 codebooks
        let codes = vec![
            (0..16).map(|i| i as u32).collect::<Vec<_>>(), // frame 0
            (100..116).map(|i| i as u32).collect::<Vec<_>>(), // frame 1
        ];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 2]);

        // Verify layout: tensor[0, q, frame] = codes[frame][q]
        let vals: Vec<i64> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        // q=0: [frame0_q0, frame1_q0] = [0, 100]
        assert_eq!(vals[0], 0);
        assert_eq!(vals[1], 100);
        // q=1: [frame0_q1, frame1_q1] = [1, 101]
        assert_eq!(vals[2], 1);
        assert_eq!(vals[3], 101);
    }

    #[test]
    fn test_parse_device_cpu() {
        let device = parse_device("cpu").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_parse_device_auto() {
        let device = parse_device("auto").unwrap();
        // Should succeed regardless of hardware
        assert!(
            matches!(device, Device::Cpu)
                || matches!(device, Device::Cuda(_))
                || matches!(device, Device::Metal(_))
        );
    }

    #[test]
    fn test_parse_device_unknown() {
        let result = parse_device("tpu");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_device_case_insensitive() {
        let device = parse_device("CPU").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_device_info() {
        assert_eq!(device_info(&Device::Cpu), "CPU");
    }

    #[test]
    fn test_compute_dtype_for_device() {
        let dtype = compute_dtype_for_device(&Device::Cpu);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_update_penalty_mask() {
        let device = Device::Cpu;
        let vocab_size = 3072;
        let mut mask = Tensor::zeros((1, vocab_size), DType::F32, &device).unwrap();

        Qwen3TTS::update_penalty_mask(&mut mask, 42, vocab_size).unwrap();

        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals[42], 1.0);
        // Neighboring positions should be untouched
        assert_eq!(vals[41], 0.0);
        assert_eq!(vals[43], 0.0);
    }

    #[test]
    fn test_update_penalty_mask_out_of_range() {
        let device = Device::Cpu;
        let vocab_size = 3072;
        let mut mask = Tensor::zeros((1, vocab_size), DType::F32, &device).unwrap();

        // Token beyond vocab_size should be a no-op (no panic)
        Qwen3TTS::update_penalty_mask(&mut mask, 9999, vocab_size).unwrap();

        let sum: f32 = mask.sum_all().unwrap().to_scalar().unwrap();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_suppression_mask_deterministic() {
        let device = Device::Cpu;
        let vocab = codec_tokens::CODEC_VOCAB_SIZE;
        let mask1 = generation::build_suppression_mask(vocab, CODEC_EOS_TOKEN_ID, &device).unwrap();
        let mask2 = generation::build_suppression_mask(vocab, CODEC_EOS_TOKEN_ID, &device).unwrap();

        // Apply both masks to uniform logits and verify identical output
        let logits = Tensor::ones((1, vocab), DType::F32, &device).unwrap();
        let out1 = generation::apply_token_suppression_with_mask(&logits, &mask1).unwrap();
        let out2 = generation::apply_token_suppression_with_mask(&logits, &mask2).unwrap();
        let v1: Vec<f32> = out1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = out2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1, v2);
    }
}
