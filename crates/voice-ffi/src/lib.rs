uniffi::setup_scaffolding!();

mod capture;
mod runtime;

pub use capture::{finish_capture, push_audio_frame, CaptureControl};
pub use runtime::{start_runtime, FfiRuntimeError};

/// Resolve the app-support directory used for logs and the discovery file:
/// `~/Library/Application Support/DianaVoice`.
pub(crate) fn app_support_dir() -> std::path::PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("DianaVoice")
}

/// Returns the resolved app-support directory as a string. Also proves the
/// Swift↔Rust bridge is linked — the native app calls this at startup.
#[uniffi::export]
pub fn data_dir_path() -> String {
    app_support_dir().to_string_lossy().into_owned()
}
