pub mod server;
pub mod tools;

pub const PROTOCOL_VERSION: &str = "2025-03-26";
pub const SERVER_NAME: &str = "diana-voice";
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SSE_CHANNEL_SIZE: usize = 64;
