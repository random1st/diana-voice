//! Synthesize text to a WAV file:
//! `cargo run --release -p voice-engine --example tts_smoke -- "текст" out.wav`
//!
//! Needs a voice reference: DIANA_VOICE_REF_AUDIO + DIANA_VOICE_REF_TEXT, or
//! ref.wav/ref.txt in ~/Library/Application Support/DianaVoice/.

use anyhow::{Context, Result};

fn main() -> Result<()> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let text = args.next().context("usage: tts_smoke <text> <out.wav>")?;
    let out = args.next().context("usage: tts_smoke <text> <out.wav>")?;

    let engine = voice_engine::tts::QwenTtsEngine::load()?;
    let started = std::time::Instant::now();
    let wav = engine.synthesize_wav(&text)?;
    println!(
        "[{} ms] {} bytes -> {out}",
        started.elapsed().as_millis(),
        wav.len()
    );
    std::fs::write(&out, wav)?;
    Ok(())
}
