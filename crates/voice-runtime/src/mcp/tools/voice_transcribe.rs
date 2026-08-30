use async_trait::async_trait;
use serde_json::{json, Value};

use super::{Tool, ToolContext};
use crate::config::Config;

/// Transcribe an audio FILE with the resident Whisper engine — meetings,
/// voice messages, recordings. No mic, no VAD flow; the whole file goes
/// through the same transcribe → mojibake-repair → hallucination-strip
/// pipeline as live listening.
pub struct VoiceTranscribeTool;

/// Whisper's expected input rate.
const TARGET_RATE: u32 = 16_000;

#[async_trait]
impl Tool for VoiceTranscribeTool {
    fn name(&self) -> &'static str {
        "voice_transcribe"
    }

    fn description(&self) -> &'static str {
        "Transcribe an audio file (WAV) with the local Whisper engine. Returns the transcript text."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to a WAV file (any sample rate; downmixed and resampled internally)."
                },
                "language": {
                    "type": "string",
                    "description": "BCP-47 language hint; omit or \"auto\" for autodetect."
                }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Value {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        if path.is_empty() {
            return error("No path provided");
        }
        let language = args
            .get("language")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| Config::load().language);

        let stt = match ctx.state.stt.read().await.as_ref().cloned() {
            Some(s) => s,
            None => return error("STT not initialized"),
        };

        let samples = match load_wav_mono_16k(path) {
            Ok(s) => s,
            Err(e) => return error(&format!("{e:#}")),
        };

        ctx.state.emit_to_ui("avatar-mood", "processing").await;
        let result = stt.transcribe_samples(&samples, &language).await;
        ctx.state.emit_to_ui("avatar-mood", "neutral").await;

        match result {
            Ok(text) => json!({ "content": [{ "type": "text", "text": text }] }),
            Err(e) => error(&format!("Transcription failed: {e:#}")),
        }
    }
}

fn error(msg: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": msg }], "isError": true })
}

/// Decode a WAV to mono f32 @ 16 kHz: downmix by channel averaging, then
/// linear-interpolation resample. Linear is audibly imperfect but Whisper is
/// robust to it, and it keeps this dependency-free; compressed formats
/// (mp3/m4a) are out of scope — the error says so instead of half-working.
fn load_wav_mono_16k(path: &str) -> anyhow::Result<Vec<f32>> {
    if !path.to_ascii_lowercase().ends_with(".wav") {
        anyhow::bail!("Only WAV files are supported (got {path}); convert first, e.g. `afconvert -f WAVE -d LEI16 in.m4a out.wav`");
    }
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| anyhow::anyhow!("failed to open {path}: {e}"))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };

    let mono: Vec<f32> = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();

    if spec.sample_rate == TARGET_RATE {
        return Ok(mono);
    }
    let ratio = spec.sample_rate as f64 / TARGET_RATE as f64;
    let out_len = (mono.len() as f64 / ratio) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let idx = pos as usize;
        let frac = (pos - idx as f64) as f32;
        let a = mono[idx.min(mono.len() - 1)];
        let b = mono[(idx + 1).min(mono.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    Ok(out)
}
