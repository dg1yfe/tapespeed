//! Output dispatch: `.flac` (codec unset) -> in-process parallel encoder ([`crate::flac`]);
//! everything else, or `TAPESPEED_FLAC=ffmpeg`, -> the ffmpeg pipe ([`crate::ffmpeg::Writer`]).
use crate::{Error, ffmpeg, flac};
use std::path::Path;

type Result<T> = std::result::Result<T, Error>;

enum Inner {
    Ffmpeg(ffmpeg::Writer),
    Flac(flac::FlacWriter),
}

/// Streaming audio writer: `create`, `write` interleaved f32 (whole frames), `finish`.
pub struct Writer(Inner);

/// `.flac` (case-insens.) with no explicit codec and `TAPESPEED_FLAC` not `ffmpeg`.
fn native_flac(path: &Path, codec: Option<&str>) -> bool {
    codec.is_none()
        && path.to_string_lossy().to_lowercase().ends_with(".flac")
        && std::env::var("TAPESPEED_FLAC").map_or(true, |v| v != "ffmpeg")
}

impl Writer {
    /// `codec = None` -> `pcm_f32le` for `.wav`, 24-bit FLAC for `.flac` (in-process; same samples as
    /// ffmpeg's s32 -> 24 bit), otherwise ffmpeg's default for the container.
    pub fn create(
        path: impl AsRef<Path>,
        sr: u32,
        channels: usize,
        codec: Option<&str>,
    ) -> Result<Writer> {
        let path = path.as_ref();
        Ok(Writer(if native_flac(path, codec) {
            Inner::Flac(flac::FlacWriter::create(path, sr, channels)?)
        } else {
            Inner::Ffmpeg(ffmpeg::Writer::create(path, sr, channels, codec)?)
        }))
    }

    /// Append interleaved samples (whole frames).
    pub fn write(&mut self, x: &[f32]) -> Result<()> {
        match &mut self.0 {
            Inner::Ffmpeg(w) => w.write(x),
            Inner::Flac(w) => w.write(x),
        }
    }

    /// Flush and close; ffmpeg errors carry its stderr.
    pub fn finish(self) -> Result<()> {
        match self.0 {
            Inner::Ffmpeg(w) => w.finish(),
            Inner::Flac(w) => w.finish(),
        }
    }
}
