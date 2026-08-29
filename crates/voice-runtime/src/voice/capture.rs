//! Capture session registry — routes externally-pushed audio frames into the
//! `voice_listen` flow.
//!
//! There is deliberately no cpal here (donor had one, grabbing ghost input
//! devices on this hardware — see stt.rs). The real capture path for this
//! product is native on the Swift side (AVAudioEngine), which POSTs already
//! 16 kHz mono f32 frames to this process. This module is just the handoff
//! point: a session id ties a stream of pushed frames to the one
//! `voice_listen` call waiting on them.
//!
//! Single-user, single-listen-at-a-time daemon, so a process-global registry
//! (rather than threading a registry handle through every caller) is the
//! right shape — mirrors `SharedState` being reached via `Arc` everywhere
//! else in this crate.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use tokio::sync::mpsc;

/// What the frame pusher should do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureControl {
    Continue,
    Stop,
}

struct CaptureSession {
    tx: mpsc::UnboundedSender<Vec<f32>>,
}

#[derive(Default)]
struct Registry {
    sessions: Mutex<HashMap<u64, CaptureSession>>,
    next_id: AtomicU64,
}

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(Registry::default)
}

/// Start a new capture session. Returns the id the pusher must tag frames
/// with, and the receiving end `voice_listen` drains frames from.
pub fn create_session() -> (u64, mpsc::UnboundedReceiver<Vec<f32>>) {
    let id = registry().next_id.fetch_add(1, Ordering::SeqCst) + 1;
    let (tx, rx) = mpsc::unbounded_channel();
    registry()
        .sessions
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(id, CaptureSession { tx });
    (id, rx)
}

/// Push a frame of 16 kHz mono f32 samples into a session. An unknown session
/// id (already finished, or never existed) tells the pusher to stop sending
/// rather than erroring — the capture side has no other way to learn that
/// `voice_listen` has already moved on.
pub fn push_frames(session_id: u64, frames: Vec<f32>) -> CaptureControl {
    let sessions = registry()
        .sessions
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    match sessions.get(&session_id) {
        Some(session) => {
            if session.tx.send(frames).is_ok() {
                CaptureControl::Continue
            } else {
                CaptureControl::Stop
            }
        }
        None => CaptureControl::Stop,
    }
}

/// End a capture session: drop its sender, which closes the receiver
/// `voice_listen` is draining and lets that loop exit cleanly.
pub fn finish_capture(session_id: u64) {
    registry()
        .sessions
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&session_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_to_unknown_session_stops() {
        assert_eq!(push_frames(u64::MAX, vec![0.0; 4]), CaptureControl::Stop);
    }

    #[tokio::test]
    async fn create_push_finish_round_trip() {
        let (id, mut rx) = create_session();
        assert_eq!(push_frames(id, vec![1.0, 2.0]), CaptureControl::Continue);
        assert_eq!(rx.recv().await, Some(vec![1.0, 2.0]));

        finish_capture(id);
        assert_eq!(push_frames(id, vec![3.0]), CaptureControl::Stop);
        assert_eq!(rx.recv().await, None);
    }
}
