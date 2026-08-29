//! stt-lab kernel-compatible bench harness for the extracted engine.
//!
//! Same contract as ~/.diana/stt-lab/kernel: argv[1..] = 16 kHz mono WAVs,
//! stdout = one JSON line per clip {"id", "text", "ms"}. Model load + warm-up
//! happen before the timed section (the app keeps the engine resident, so
//! per-utterance latency is what matters). Pipe stdout into the lab's LOCKED
//! scorers (score.py / score_hard.py / score_mixed.py / score_long.py).

use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::time::Instant;

fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(spec.channels == 1, "expected mono, got {}", spec.channels);
    ensure!(
        spec.sample_rate == 16_000,
        "expected 16 kHz, got {}",
        spec.sample_rate
    );
    let samples = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };
    Ok(samples)
}

fn main() -> Result<()> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    ensure!(!paths.is_empty(), "usage: stt_bench_kernel <wav>...");

    let engine = voice_engine::NativeSttEngine::new();
    engine.warm_up().context("engine warm-up failed")?;

    for path in &paths {
        let path = Path::new(path);
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .context("bad wav path")?;
        let samples = read_wav_16k_mono(path)?;

        let started = Instant::now();
        let text = engine.transcribe(samples, "auto")?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{}",
            serde_json::json!({ "id": id, "text": text, "ms": ms })
        );
    }
    Ok(())
}
