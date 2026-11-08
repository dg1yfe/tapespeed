//! Decoded audio either in RAM or spilled to an f32 temp file mapped read-only (the page
//! cache then holds it; clean file pages are reclaimable and count neither on the heap nor in
//! the macOS footprint). The temp file is unlinked right after it is created, so it can never
//! outlive the process, not even after SIGKILL; a guard removes it if creation fails halfway.
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::Error;

/// Interleaved f32 frames: owned `Vec` or a read-only mapping of an unlinked spill file.
pub struct Samples {
    buf: Buf,
    pub channels: usize,
    pub sr: u32,
}

enum Buf {
    Ram(Vec<f32>),
    Map(*const f32, usize),
}

// SAFETY: the mapping is read-only, private to this value and unmapped only in Drop.
unsafe impl Send for Samples {}
unsafe impl Sync for Samples {}

impl Samples {
    pub fn ram(data: Vec<f32>, channels: usize, sr: u32) -> Self {
        Samples {
            buf: Buf::Ram(data),
            channels,
            sr,
        }
    }
    pub fn frames(&self) -> usize {
        self.len() / self.channels.max(1)
    }
    pub fn is_spilled(&self) -> bool {
        matches!(self.buf, Buf::Map(..))
    }
}

impl std::ops::Deref for Samples {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match &self.buf {
            Buf::Ram(v) => v,
            // SAFETY: `len` f32 of a live read-only mapping created in `spill_with`
            Buf::Map(p, len) => unsafe { std::slice::from_raw_parts(*p, *len) },
        }
    }
}

impl Drop for Samples {
    fn drop(&mut self) {
        if let Buf::Map(p, len) = self.buf {
            // SAFETY: mapping created by mmap with this length
            unsafe { libc::munmap(p as *mut libc::c_void, len * 4) };
        }
    }
}

/// Removes the path on drop unless disarmed (creation → unlink window, errors, panics).
struct Guard(Option<PathBuf>);

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Temp directory: `dir`, else `$TAPESPEED_TMPDIR`, else the system temp dir.
pub fn tmp_dir(dir: Option<&str>) -> PathBuf {
    dir.map(PathBuf::from)
        .or_else(|| std::env::var_os("TAPESPEED_TMPDIR").map(PathBuf::from))
        .unwrap_or_else(std::env::temp_dir)
}

/// Free bytes on the file system of `dir` (None if unknown).
pub fn free_bytes(dir: &Path) -> Option<u64> {
    let c = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes()).ok()?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: valid C string and out pointer
    (unsafe { libc::statvfs(c.as_ptr(), &mut s) } == 0)
        .then(|| s.f_bavail as u64 * s.f_frsize as u64)
}

/// Spill: `fill` writes the decoded frames through the callback and returns `(sr, channels)`;
/// the file is created in `dir`, unlinked at once, then mapped.
pub fn spill_with(
    dir: &Path,
    fill: impl FnOnce(&mut dyn FnMut(&[f32]) -> std::io::Result<()>) -> Result<(u32, usize), Error>,
) -> Result<Samples, Error> {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let path = dir.join(format!(
        "tapespeed-{}-{}.f32",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let mut guard = Guard(Some(path.clone()));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    std::fs::remove_file(&path)?;
    guard.0 = None;
    let mut w = BufWriter::with_capacity(1 << 20, &file);
    let mut n = 0usize;
    let mut bytes = Vec::new();
    let (sr, ch) = fill(&mut |c: &[f32]| {
        n += c.len();
        bytes.clear();
        bytes.extend(c.iter().flat_map(|v| v.to_le_bytes()));
        w.write_all(&bytes)
    })?;
    w.flush()?;
    drop(w);
    if n == 0 {
        return Ok(Samples::ram(vec![], ch, sr));
    }
    use std::os::fd::AsRawFd;
    // SAFETY: fresh private file of exactly n f32; read-only shared mapping
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            n * 4,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(Samples {
        buf: Buf::Map(p as *const f32, n),
        channels: ch,
        sr,
    })
}

/// Decode `path` into a spill file in `dir`.
pub fn spill(path: impl AsRef<Path>, dir: &Path) -> Result<Samples, Error> {
    spill_tap(path, dir, |_, _, _| {}, |_| {})
}

/// [`spill`] that also shows every decoded chunk to `tap` as soon as it was written; `init` as
/// in [`crate::ffmpeg::load_tap`].
pub fn spill_tap(
    path: impl AsRef<Path>,
    dir: &Path,
    init: impl FnOnce(u32, usize, usize),
    mut tap: impl FnMut(&[f32]),
) -> Result<Samples, Error> {
    let path = path.as_ref();
    spill_with(dir, |put| {
        let inf = crate::info(path)?;
        let (sr, ch) = (inf.sr, inf.channels);
        init(
            sr,
            ch,
            crate::ffmpeg::reserve_frames(sr, inf.duration, None, None),
        );
        let mut err = None;
        crate::ffmpeg::load_with(path, 1 << 16, (sr, ch), |c| {
            if err.is_none() {
                err = put(c).err();
                tap(c);
            }
        })?;
        match err {
            Some(e) => Err(e.into()),
            None => Ok((sr, ch)),
        }
    })
}

/// Whole file in RAM (`load`, exact reservation).
pub fn load_ram(path: impl AsRef<Path>) -> Result<Samples, Error> {
    load_ram_tap(path, |_, _, _| {}, |_| {})
}

/// [`load_ram`] with `init` / `tap` as in [`crate::ffmpeg::load_tap`].
pub fn load_ram_tap(
    path: impl AsRef<Path>,
    init: impl FnOnce(u32, usize, usize),
    tap: impl FnMut(&[f32]),
) -> Result<Samples, Error> {
    let a = crate::ffmpeg::load_tap(path, None, None, init, tap)?;
    Ok(Samples::ram(a.data, a.channels, a.sr))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ts-spill-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn entries(d: &Path) -> usize {
        std::fs::read_dir(d).unwrap().count()
    }

    #[test]
    fn spill_maps_data_and_leaves_no_file() {
        let d = empty_dir("ok");
        let x: Vec<f32> = (0..300_001).map(|i| (i as f32 * 0.01).sin()).collect();
        let s = spill_with(&d, |put| {
            for c in x.chunks(4096) {
                put(c)?;
            }
            Ok((48000, 1))
        })
        .unwrap();
        assert!(s.is_spilled());
        assert_eq!(entries(&d), 0, "unlinked while mapped");
        assert_eq!(&s[..], &x[..]);
        drop(s);
        assert_eq!(entries(&d), 0);
        std::fs::remove_dir(&d).unwrap();
    }

    #[test]
    fn spill_cleans_up_on_error_and_panic() {
        let d = empty_dir("err");
        let r = spill_with(&d, |put| {
            put(&[1.0, 2.0])?;
            Err(Error::Parse("decoder failed".into()))
        });
        assert!(r.is_err());
        assert_eq!(entries(&d), 0);
        let p = std::panic::catch_unwind(|| {
            let _ = spill_with(&d, |put| {
                put(&[1.0; 1000]).unwrap();
                panic!("boom")
            });
        });
        assert!(p.is_err());
        assert_eq!(entries(&d), 0);
        // the guard itself: a file left by a failed create is removed
        let f = d.join("x.f32");
        std::fs::write(&f, b"1").unwrap();
        drop(Guard(Some(f)));
        assert_eq!(entries(&d), 0);
        std::fs::remove_dir(&d).unwrap();
    }

    #[test]
    fn free_bytes_known() {
        assert!(free_bytes(&std::env::temp_dir()).unwrap_or(0) > 0);
    }
}
