//! ffmpeg I/O, in-process FLAC encoder, number formatting, JSON emitter.
pub mod ffmpeg;
pub mod flac;
pub mod json;
pub mod machine;
pub mod numfmt;
pub mod prof;
pub mod spill;
pub mod writer;

pub use ffmpeg::{Audio, Info, info, load, load_chunks, lossy_default, output_note, probe, save};
pub use numfmt::{fmt_fixed, fmt_general, np_round, repr_shortest, round_dec};
pub use spill::Samples;
pub use writer::Writer;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot run {0}: {1}")]
    Spawn(&'static str, std::io::Error),
    #[error("{0} failed (exit {1:?}): {2}")]
    Failed(&'static str, Option<i32>, String),
    #[error("{0}")]
    Parse(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
