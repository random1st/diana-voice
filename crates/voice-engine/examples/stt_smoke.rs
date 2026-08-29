//! Transcribe a 16 kHz mono WAV: `cargo run --release -p voice-engine --example stt_smoke -- clip.wav`
//!
//! Set RUST_LOG=info to see the backend and adaptive-window lines.

use anyhow::{bail, Context, Result};

fn main() -> Result<()> {
    env_logger::init();
    let path = std::env::args()
        .nth(1)
        .context("usage: stt_smoke <clip.wav>")?;

    let mut reader = hound::WavReader::open(&path).with_context(|| format!("open {path}"))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 {
        bail!(
            "expected 16 kHz mono, got {} Hz / {} ch",
            spec.sample_rate,
            spec.channels
        );
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };

    let engine = voice_engine::NativeSttEngine::new();
    let started = std::time::Instant::now();
    let text = engine.transcribe(samples.clone(), "auto")?;
    println!("[cold {} ms] {text}", started.elapsed().as_millis());

    // Warm run: model resident, no load cost — comparable to the bench p50.
    let started = std::time::Instant::now();
    let text = engine.transcribe(samples, "auto")?;
    println!("[warm {} ms] {text}", started.elapsed().as_millis());
    Ok(())
}
