//! Bridges externally-pushed audio frames (Swift's AVAudioEngine tap) into
//! `voice-runtime`'s capture registry (`voice_runtime::voice::capture`). The
//! real capture path for this product is native on the Swift side; this
//! module is just the uniffi handoff point.

use voice_runtime::voice::capture as runtime_capture;

/// What the frame pusher (Swift) should do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum CaptureControl {
    Continue,
    Stop,
}

impl From<runtime_capture::CaptureControl> for CaptureControl {
    fn from(value: runtime_capture::CaptureControl) -> Self {
        match value {
            runtime_capture::CaptureControl::Continue => CaptureControl::Continue,
            runtime_capture::CaptureControl::Stop => CaptureControl::Stop,
        }
    }
}

/// Push a frame of 16 kHz mono f32 samples into `session_id`'s capture
/// stream. An unknown session id (already finished, or never existed) tells
/// the pusher to stop rather than erroring — the capture side has no other
/// way to learn that `voice_listen` has already moved on.
#[uniffi::export]
pub fn push_audio_frame(session_id: u64, samples: Vec<f32>) -> CaptureControl {
    runtime_capture::push_frames(session_id, samples).into()
}

/// Signal end-of-stream for `session_id`: closes the receiver `voice_listen`
/// is draining so that flow exits cleanly.
#[uniffi::export]
pub fn finish_capture(session_id: u64) {
    runtime_capture::finish_capture(session_id);
}
