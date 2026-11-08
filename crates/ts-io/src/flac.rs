//! In-process FLAC encoder for `.flac` outputs (24 bit), frames encoded in parallel on the rayon pool.
//!
//! Quantisation is ffmpeg's: swresample `flt -> s32` (`av_clipl_int32(llrintf(x * 2^31))`), then the
//! FLAC encoder's `s32 >> 8` to 24 bit. Decoding the result gives the samples of the old ffmpeg path.
//! Frames: `frame::encode` (own coder, ffmpeg-like choices); stream: `fLaC`, STREAMINFO (patched
//! on finish: frame sizes, total samples, MD5), VORBIS_COMMENT, frames.
//!
//! Pipeline: `write` only copies f32 into a batch; a full batch becomes one rayon job (blocks in
//! parallel: quantise, encode, pack MD5 bytes); an output thread takes the jobs in order, writes the
//! frames and hashes. At most `inflight` batches exist at a time (bounded memory).
use crate::Error;
use md5::{Digest, Md5};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;

type Result<T> = std::result::Result<T, Error>;

mod frame;

/// Frames per FLAC block (libFLAC's and ffmpeg's default).
pub const BLOCK: usize = 4096;
/// Blocks per batch (one rayon job): 64 k frames.
const BATCH_BLOCKS: usize = 16;
const BPS: usize = 24;
const VENDOR: &str = concat!("tapespeed ", env!("CARGO_PKG_VERSION"));

/// ffmpeg's f32 -> 24 bit: round half to even at 2^-31, clip to i32, arithmetic shift by 8
/// (NaN -> 0, ±inf -> clipped).
#[inline]
pub fn q24(x: f32) -> i32 {
    ((x * 2_147_483_648.0f32).round_ties_even() as i32) >> 8
}

struct Batch {
    bytes: Vec<u8>,
    /// encoded size of each frame
    sizes: Vec<u32>,
    /// interleaved 24-bit LE samples (MD5 input)
    pcm: Vec<u8>,
    frames: usize,
}

type Job = Receiver<std::result::Result<Batch, String>>;

struct Done {
    min_frame: u32,
    max_frame: u32,
    total: u64,
    md5: [u8; 16],
    file: BufWriter<File>,
}

pub struct FlacWriter {
    ch: usize,
    sr: u32,
    buf: Vec<f32>,
    next_frame: usize,
    jobs: Option<SyncSender<Job>>,
    out: Option<JoinHandle<Result<Done>>>,
}

fn streaminfo(
    sr: u32,
    ch: usize,
    min_frame: u32,
    max_frame: u32,
    total: u64,
    md5: &[u8; 16],
) -> [u8; 34] {
    let mut s = [0u8; 34];
    s[0..2].copy_from_slice(&(BLOCK as u16).to_be_bytes());
    s[2..4].copy_from_slice(&(BLOCK as u16).to_be_bytes());
    s[4..7].copy_from_slice(&min_frame.to_be_bytes()[1..]);
    s[7..10].copy_from_slice(&max_frame.to_be_bytes()[1..]);
    let packed: u64 = ((sr as u64) << 44)
        | (((ch as u64) - 1) << 41)
        | (((BPS as u64) - 1) << 36)
        | (total & ((1u64 << 36) - 1));
    s[10..18].copy_from_slice(&packed.to_be_bytes());
    s[18..34].copy_from_slice(md5);
    s
}

fn header(sr: u32, ch: usize) -> Vec<u8> {
    let mut h = b"fLaC".to_vec();
    h.extend([0u8, 0, 0, 34]); // STREAMINFO, not last
    h.extend(streaminfo(sr, ch, 0, 0, 0, &[0; 16]));
    let v = VENDOR.as_bytes();
    let len = 4 + v.len() + 4;
    h.push(0x80 | 4); // VORBIS_COMMENT, last
    h.extend(&(len as u32).to_be_bytes()[1..]);
    h.extend((v.len() as u32).to_le_bytes());
    h.extend(v);
    h.extend(0u32.to_le_bytes());
    h
}

fn encode_batch(x: &[f32], ch: usize, sr: u32, first_frame: usize) -> Batch {
    let enc: Vec<(Vec<u8>, Vec<u8>)> = x
        .par_chunks(BLOCK * ch)
        .enumerate()
        .map(|(i, blk)| {
            let q: Vec<i32> = blk.iter().map(|&v| q24(v)).collect();
            let mut pcm = Vec::with_capacity(q.len() * 3);
            for v in &q {
                pcm.extend_from_slice(&v.to_le_bytes()[..3]);
            }
            let f = frame::encode(&q, ch, BPS as u32, sr, (first_frame + i) as u64);
            (f, pcm)
        })
        .collect();
    let mut b = Batch {
        bytes: Vec::with_capacity(enc.iter().map(|e| e.0.len()).sum()),
        sizes: Vec::with_capacity(enc.len()),
        pcm: Vec::with_capacity(x.len() * 3),
        frames: x.len() / ch,
    };
    for (f, p) in enc {
        b.sizes.push(f.len() as u32);
        b.bytes.extend_from_slice(&f);
        b.pcm.extend_from_slice(&p);
    }
    b
}

fn output(mut file: BufWriter<File>, jobs: Receiver<Job>) -> Result<Done> {
    let (mut min_frame, mut max_frame, mut total) = (u32::MAX, 0u32, 0u64);
    let mut md5 = Md5::new();
    for job in jobs {
        let b = job
            .recv()
            .map_err(|_| Error::Parse("flac: encoder job died".into()))?
            .map_err(|e| Error::Parse(format!("flac: {e}")))?;
        file.write_all(&b.bytes)?;
        md5.update(&b.pcm);
        for &s in &b.sizes {
            min_frame = min_frame.min(s);
            max_frame = max_frame.max(s);
        }
        total += b.frames as u64;
    }
    if total == 0 {
        min_frame = 0;
    }
    Ok(Done {
        min_frame,
        max_frame,
        total,
        md5: md5.finalize().into(),
        file,
    })
}

/// On a rayon worker, run other pool jobs (the encode jobs need a worker) instead of blocking.
fn help_pool() {
    if rayon::yield_now() != Some(rayon::Yield::Executed) {
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
}

/// Queue a job for the output thread; blocks while `inflight` jobs are queued (bounded memory).
fn send_job(tx: &SyncSender<Job>, mut job: Job) -> Result<()> {
    let gone = || Error::Parse("flac: output thread stopped".into());
    if rayon::current_thread_index().is_none() {
        return tx.send(job).map_err(|_| gone());
    }
    loop {
        match tx.try_send(job) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(j)) => {
                job = j;
                help_pool();
            }
            Err(TrySendError::Disconnected(_)) => return Err(gone()),
        }
    }
}

impl FlacWriter {
    pub fn create(path: &Path, sr: u32, channels: usize) -> Result<FlacWriter> {
        if !(1..=8).contains(&channels) || sr == 0 || sr >= 1 << 20 {
            return Err(Error::Parse(format!(
                "flac: unsupported format ({sr} Hz, {channels} channels)"
            )));
        }
        let mut file = BufWriter::with_capacity(1 << 20, File::create(path)?);
        file.write_all(&header(sr, channels))?;
        let inflight = (2 * rayon::current_num_threads()).max(4);
        let (tx, rx) = sync_channel::<Job>(inflight);
        let out = std::thread::Builder::new()
            .name("flac-out".into())
            .spawn(move || output(file, rx))?;
        Ok(FlacWriter {
            ch: channels,
            sr,
            buf: Vec::with_capacity(BLOCK * BATCH_BLOCKS * channels),
            next_frame: 0,
            jobs: Some(tx),
            out: Some(out),
        })
    }

    fn submit(&mut self) -> Result<()> {
        let x = std::mem::replace(
            &mut self.buf,
            Vec::with_capacity(BLOCK * BATCH_BLOCKS * self.ch),
        );
        let n_blocks = (x.len() / self.ch).div_ceil(BLOCK);
        let (ch, sr, first) = (self.ch, self.sr, self.next_frame);
        self.next_frame += n_blocks;
        let (rtx, rrx) = sync_channel(1);
        // the output thread gone means it hit an error; finish() reports it
        send_job(self.jobs.as_ref().expect("open"), rrx)?;
        rayon::spawn(move || {
            let r = std::panic::catch_unwind(|| encode_batch(&x, ch, sr, first))
                .map_err(|_| "encoder panicked".to_string());
            let _ = rtx.send(r);
        });
        Ok(())
    }

    /// Append interleaved samples (whole frames).
    pub fn write(&mut self, mut x: &[f32]) -> Result<()> {
        assert!(x.len().is_multiple_of(self.ch), "partial frame");
        let cap = BLOCK * BATCH_BLOCKS * self.ch;
        while !x.is_empty() {
            let n = (cap - self.buf.len()).min(x.len());
            self.buf.extend_from_slice(&x[..n]);
            x = &x[n..];
            if self.buf.len() == cap {
                self.submit()?;
            }
        }
        Ok(())
    }

    /// Encode the rest, wait for all frames, patch STREAMINFO.
    pub fn finish(mut self) -> Result<()> {
        let sub = if self.buf.is_empty() {
            Ok(())
        } else {
            self.submit()
        };
        drop(self.jobs.take());
        let out = self.out.take().expect("open");
        while !out.is_finished() && rayon::current_thread_index().is_some() {
            help_pool();
        }
        let done = out
            .join()
            .map_err(|_| Error::Parse("flac: output thread panicked".into()))??;
        sub?;
        let mut f = done.file.into_inner().map_err(|e| e.into_error())?;
        f.seek(SeekFrom::Start(8))?;
        f.write_all(&streaminfo(
            self.sr,
            self.ch,
            done.min_frame,
            done.max_frame,
            done.total,
            &done.md5,
        ))?;
        f.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q24_matches_swresample_then_shift() {
        let t = 1.0 / 2_147_483_648.0f32;
        for (x, q) in [
            (0.0f32, 0),
            (-0.0, 0),
            (1.0, 8_388_607),
            (-1.0, -8_388_608),
            (7.0, 8_388_607),
            (f32::INFINITY, 8_388_607),
            (f32::NEG_INFINITY, -8_388_608),
            (f32::NAN, 0),
            (255.5 * t, 1), // rint 255.5 -> 256 (ties to even) -> >> 8 = 1
            (254.5 * t, 0), // rint -> 254 -> 0
            (-0.5 * t, 0),  // rint -> -0 -> 0
            (-1.5 * t, -1), // rint -> -2 -> floor(-2 / 256) = -1
            (1.0 / 8_388_608.0, 1),
        ] {
            assert_eq!(q24(x), q, "x = {x:e}");
        }
    }

    #[test]
    fn streaminfo_layout() {
        let s = streaminfo(44100, 2, 20, 18222, 113_660_257, &[7; 16]);
        assert_eq!(&s[..4], &[0x10, 0, 0x10, 0]);
        assert_eq!(&s[4..10], &[0, 0, 20, 0, 0x47, 0x2e]);
        let p = u64::from_be_bytes(s[10..18].try_into().unwrap());
        assert_eq!(
            (p >> 44, (p >> 41) & 7, (p >> 36) & 31, p & 0xf_ffff_ffff),
            (44100, 1, 23, 113_660_257)
        );
        assert_eq!(&s[18..], &[7; 16]);
    }
}
