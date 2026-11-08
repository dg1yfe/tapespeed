//! ffmpeg / ffprobe subprocess I/O with fixed argument lists.
//! Sample layout everywhere: interleaved f32, frame-major (an `(n_samples, n_channels)` array in C order).
use crate::Error;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::thread::JoinHandle;

type Result<T> = std::result::Result<T, Error>;

/// Whole-file / segment audio, interleaved.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub data: Vec<f32>,
    pub channels: usize,
    pub sr: u32,
}

impl Audio {
    pub fn frames(&self) -> usize {
        self.data.len() / self.channels.max(1)
    }
}

/// Stream and container properties reported by ffprobe.
#[derive(Debug, Clone, PartialEq)]
pub struct Info {
    pub codec: String,
    pub format: String,
    pub sr: u32,
    pub channels: usize,
    /// seconds; NaN if the container gives none
    pub duration: f64,
    /// bit/s, 0 if unknown
    pub bit_rate: i64,
}

fn ffprobe_json(entries: &str, path: &Path) -> Result<serde_json::Value> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            entries,
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|e| Error::Spawn("ffprobe", e))?;
    if !out.status.success() {
        return Err(Error::Failed(
            "ffprobe",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).into(),
        ));
    }
    Ok(serde_json::from_slice(&out.stdout)?)
}

fn first_stream(j: &serde_json::Value) -> Result<&serde_json::Value> {
    j["streams"]
        .get(0)
        .ok_or_else(|| Error::Parse("ffprobe: no audio stream".into()))
}

/// JSON scalar (string or number) as text.
fn text(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn need<T: std::str::FromStr>(v: &serde_json::Value, what: &str) -> Result<T> {
    text(v)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| Error::Parse(format!("ffprobe: bad {what}")))
}

/// `(sample_rate, channels)` of the first audio stream.
pub fn probe(path: impl AsRef<Path>) -> Result<(u32, usize)> {
    let j = ffprobe_json("stream=sample_rate,channels", path.as_ref())?;
    let s = first_stream(&j)?;
    Ok((
        need(&s["sample_rate"], "sample_rate")?,
        need(&s["channels"], "channels")?,
    ))
}

pub fn info(path: impl AsRef<Path>) -> Result<Info> {
    let j = ffprobe_json(
        "stream=codec_name,sample_rate,channels,bit_rate:format=duration,bit_rate,format_name",
        path.as_ref(),
    )?;
    let s = first_stream(&j)?;
    let f = &j["format"];
    // first non-empty of stream / container bit_rate; 0 unless it is all digits
    let br = [&s["bit_rate"], &f["bit_rate"]]
        .into_iter()
        .filter_map(text)
        .find(|t| !t.is_empty());
    let bit_rate = br
        .filter(|t| !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()))
        .and_then(|t| t.parse().ok())
        .unwrap_or(0);
    Ok(Info {
        codec: text(&s["codec_name"]).ok_or_else(|| Error::Parse("ffprobe: codec_name".into()))?,
        format: text(&f["format_name"]).unwrap_or_else(|| "?".into()),
        sr: need(&s["sample_rate"], "sample_rate")?,
        channels: need(&s["channels"], "channels")?,
        duration: text(&f["duration"])
            .and_then(|t| t.parse().ok())
            .unwrap_or(f64::NAN),
        bit_rate,
    })
}

fn drain(mut r: impl Read + Send + 'static) -> JoinHandle<String> {
    std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = r.read_to_end(&mut b);
        String::from_utf8_lossy(&b).into_owned()
    })
}

fn decode_args(path: &Path, start: Option<f64>, dur: Option<f64>) -> Vec<String> {
    let mut a: Vec<String> = ["-v", "error"].map(String::from).into();
    // None and 0 both mean "no option"
    if let Some(s) = start.filter(|&s| s != 0.0) {
        a.extend(["-ss".into(), format!("{s:.3}")]);
    }
    if let Some(d) = dur.filter(|&d| d != 0.0) {
        a.extend(["-t".into(), format!("{d:.3}")]);
    }
    a.extend(["-i".into(), path.to_string_lossy().into_owned()]);
    a.extend(["-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-"].map(String::from));
    a
}

/// Streaming decode: calls `sink` with consecutive interleaved chunks (whole frames, at most
/// `chunk_frames` each). Returns `(sr, channels)`.
pub fn load_chunks(
    path: impl AsRef<Path>,
    start: Option<f64>,
    dur: Option<f64>,
    chunk_frames: usize,
    sink: impl FnMut(&[f32]),
) -> Result<(u32, usize)> {
    let path = path.as_ref();
    let (sr, ch) = probe(path)?;
    decode(path, start, dur, chunk_frames, ch, sink)?;
    Ok((sr, ch))
}

/// The decode loop of [`load_chunks`] for a probed `ch`.
fn decode(
    path: &Path,
    start: Option<f64>,
    dur: Option<f64>,
    chunk_frames: usize,
    ch: usize,
    mut sink: impl FnMut(&[f32]),
) -> Result<()> {
    let mut child = Command::new("ffmpeg")
        .args(decode_args(path, start, dur))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()
        .map_err(|e| Error::Spawn("ffmpeg", e))?;
    let err = drain(child.stderr.take().expect("piped"));
    let mut out = child.stdout.take().expect("piped");
    let frame_bytes = 4 * ch;
    let mut buf = vec![0u8; frame_bytes * chunk_frames.max(1)];
    let mut fill = 0usize;
    let mut samples: Vec<f32> = Vec::with_capacity(ch * chunk_frames.max(1));
    loop {
        let n = out.read(&mut buf[fill..])?;
        fill += n;
        if n == 0 || fill == buf.len() {
            let whole = fill / frame_bytes * frame_bytes;
            if whole > 0 {
                samples.clear();
                samples.extend(
                    buf[..whole]
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                );
                sink(&samples);
            }
            buf.copy_within(whole..fill, 0);
            fill -= whole;
            if n == 0 {
                break;
            }
        }
    }
    let status = child.wait()?;
    let msg = err.join().unwrap_or_default();
    if !status.success() {
        return Err(Error::Failed("ffmpeg", status.code(), msg));
    }
    if fill != 0 {
        return Err(Error::Parse(format!(
            "ffmpeg: {fill} trailing bytes, not a whole frame"
        )));
    }
    Ok(())
}

/// Frames to reserve for `[start, start + dur)` of a file of `total` s (+1 s slack; 0 if unknown).
pub(crate) fn reserve_frames(sr: u32, total: f64, start: Option<f64>, dur: Option<f64>) -> usize {
    let rest = total - start.unwrap_or(0.0).max(0.0);
    let secs = dur.filter(|&d| d != 0.0).map_or(rest, |d| d.min(rest));
    if secs.is_finite() && secs > 0.0 {
        ((secs + 1.0) * sr as f64) as usize
    } else {
        0
    }
}

/// Decode the whole file, or `dur` s from `start` s (input seek). The buffer is reserved
/// from the ffprobe duration, so it does not double while it grows (no 1.5x copy transient).
pub fn load(path: impl AsRef<Path>, start: Option<f64>, dur: Option<f64>) -> Result<Audio> {
    load_tap(path, start, dur, |_, _, _| {}, |_| {})
}

/// [`load`] that also shows every decoded chunk to `tap` as soon as it arrives (after it was
/// appended). `init(sr, channels, expected_frames)` runs once, before the decoder starts
/// (`expected_frames`: from the ffprobe duration + 1 s, 0 if unknown).
pub fn load_tap(
    path: impl AsRef<Path>,
    start: Option<f64>,
    dur: Option<f64>,
    init: impl FnOnce(u32, usize, usize),
    mut tap: impl FnMut(&[f32]),
) -> Result<Audio> {
    let path = path.as_ref();
    let inf = info(path)?;
    let res = reserve_frames(inf.sr, inf.duration, start, dur);
    let mut data = Vec::with_capacity(res * inf.channels);
    init(inf.sr, inf.channels, res);
    decode(path, start, dur, 1 << 16, inf.channels, |c| {
        data.extend_from_slice(c);
        tap(c)
    })?;
    Ok(Audio {
        data,
        channels: inf.channels,
        sr: inf.sr,
    })
}

/// Decode into `sink` with the probed `(sr, channels)` known before the first chunk.
pub fn load_with(
    path: impl AsRef<Path>,
    chunk_frames: usize,
    sr_ch: (u32, usize),
    sink: impl FnMut(&[f32]),
) -> Result<()> {
    decode(path.as_ref(), None, None, chunk_frames, sr_ch.1, sink)
}

/// Lower-case extension of `path` ("" if none).
fn ext_of(path: &Path) -> String {
    path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

/// This ffmpeg build has the `libvorbis` encoder (probed once).
fn has_libvorbis() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        Command::new("ffmpeg")
            .args(["-hide_banner", "-encoders"])
            .output()
            .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains(" libvorbis "))
    })
}

/// Encoder settings for a lossy output container when no codec is given.
/// ffmpeg's own defaults are low (MP3 / AAC 128 kb/s CBR, Opus 96 kb/s, Vorbis
/// q3 ~112 kb/s): `.mp3` -> LAME VBR `-q:a 2` (~190 kb/s stereo; LAME's recommended
/// "transparent" preset, adapts the rate to the content), `.m4a/.aac/.mp4` -> AAC 128 kb/s per
/// channel, `.opus` -> Opus 80 kb/s per channel (Opus always runs at 48 kHz: ffmpeg resamples),
/// `.ogg/.oga` -> Vorbis `-q:a 6` (~192 kb/s) if this ffmpeg has libvorbis (otherwise ffmpeg
/// writes FLAC in Ogg: lossless). Returns (ffmpeg arguments, description); None: not lossy.
pub fn lossy_default(path: &Path, channels: usize) -> Option<(Vec<String>, String)> {
    let ch = channels.clamp(1, 2) as u32;
    let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    match ext_of(path).as_str() {
        "mp3" => Some((
            v(&["-c:a", "libmp3lame", "-q:a", "2"]),
            "MP3, LAME VBR -q:a 2, ~190 kb/s".into(),
        )),
        "m4a" | "aac" | "mp4" | "m4b" => {
            let b = format!("{}k", 128 * ch);
            Some((v(&["-c:a", "aac", "-b:a", &b]), format!("AAC {b}b/s")))
        }
        "opus" => {
            let b = format!("{}k", 80 * ch);
            Some((
                v(&["-c:a", "libopus", "-b:a", &b]),
                format!("Opus {b}b/s, 48 kHz"),
            ))
        }
        "ogg" | "oga" if has_libvorbis() => Some((
            v(&["-c:a", "libvorbis", "-q:a", "6"]),
            "Vorbis -q:a 6, ~192 kb/s".into(),
        )),
        _ => None,
    }
}

/// One-line hint for a lossy output path (None for lossless formats).
pub fn output_note(path: impl AsRef<Path>, channels: usize) -> Option<String> {
    let path = path.as_ref();
    lossy_default(path, channels).map(|(_, what)| {
        format!(
            "note: {} is lossy ({what}); .flac or .wav give a lossless master",
            path.display()
        )
    })
}

/// Encoder pipe: `ffmpeg -f f32le -ar SR -ac CH -i - [-acodec C | lossy defaults] OUT`.
pub struct Writer {
    child: Child,
    stdin: Option<ChildStdin>,
    err: Option<JoinHandle<String>>,
    channels: usize,
    bytes: Vec<u8>,
}

impl Writer {
    /// `codec = None` -> `pcm_f32le` for `.wav` (case-insens.), [`lossy_default`] for lossy
    /// containers, otherwise ffmpeg's default for the container (FLAC: s32 input -> 24 bit).
    pub fn create(
        path: impl AsRef<Path>,
        sr: u32,
        channels: usize,
        codec: Option<&str>,
    ) -> Result<Writer> {
        let path = path.as_ref();
        let is_wav = path.to_string_lossy().to_lowercase().ends_with(".wav");
        let codec = codec.or(if is_wav { Some("pcm_f32le") } else { None });
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-y", "-f", "f32le", "-ar"])
            .arg(sr.to_string());
        cmd.args(["-ac"])
            .arg(channels.to_string())
            .args(["-i", "-"]);
        if let Some(c) = codec {
            cmd.args(["-acodec", c]);
        } else if let Some((a, _)) = lossy_default(path, channels) {
            cmd.args(a);
        }
        cmd.arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| Error::Spawn("ffmpeg", e))?;
        let stdin = child.stdin.take();
        let err = Some(drain(child.stderr.take().expect("piped")));
        Ok(Writer {
            child,
            stdin,
            err,
            channels,
            bytes: Vec::new(),
        })
    }

    /// Append interleaved samples (whole frames).
    pub fn write(&mut self, x: &[f32]) -> Result<()> {
        assert!(x.len().is_multiple_of(self.channels), "partial frame");
        self.bytes.clear();
        self.bytes.extend(x.iter().flat_map(|v| v.to_le_bytes()));
        self.stdin.as_mut().expect("open").write_all(&self.bytes)?;
        Ok(())
    }

    /// Close the pipe and wait for ffmpeg; errors carry its stderr.
    pub fn finish(mut self) -> Result<()> {
        drop(self.stdin.take());
        let status = self.child.wait()?;
        let msg = self
            .err
            .take()
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default();
        if !status.success() {
            return Err(Error::Failed("ffmpeg", status.code(), msg));
        }
        Ok(())
    }
}

/// Encode and write audio: `x` interleaved, `channels` per frame (via [`crate::Writer`]: `.flac` in-process).
pub fn save(
    path: impl AsRef<Path>,
    x: &[f32],
    channels: usize,
    sr: u32,
    codec: Option<&str>,
) -> Result<()> {
    let mut w = crate::Writer::create(path, sr, channels, codec)?;
    for c in x.chunks(channels * (1 << 16)) {
        // a broken pipe means ffmpeg died; report its stderr rather than EPIPE
        if let Err(e) = w.write(c) {
            return match w.finish() {
                Err(f) => Err(f),
                Ok(()) => Err(e),
            };
        }
    }
    w.finish()
}

#[cfg(test)]
mod lossy_tests {
    use super::*;

    #[test]
    fn lossy_defaults_per_container() {
        let a = |p: &str, ch| lossy_default(Path::new(p), ch).map(|(a, _)| a.join(" "));
        assert_eq!(a("x.mp3", 2).as_deref(), Some("-c:a libmp3lame -q:a 2"));
        assert_eq!(a("x.MP3", 1).as_deref(), Some("-c:a libmp3lame -q:a 2"));
        assert_eq!(a("x.m4a", 2).as_deref(), Some("-c:a aac -b:a 256k"));
        assert_eq!(a("x.aac", 1).as_deref(), Some("-c:a aac -b:a 128k"));
        assert_eq!(a("x.opus", 2).as_deref(), Some("-c:a libopus -b:a 160k"));
        for p in ["x.flac", "x.wav", "x", "dir/x.aiff"] {
            assert_eq!(a(p, 2), None, "{p}");
        }
        let n = output_note("o/x.mp3", 2).unwrap();
        assert!(
            n.starts_with("note: o/x.mp3 is lossy (MP3, LAME VBR -q:a 2"),
            "{n}"
        );
        assert!(output_note("x.flac", 2).is_none());
    }
}
