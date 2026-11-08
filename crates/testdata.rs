//! Stored test data (`crates/*/tests/data`) are not in git: `tools/fetch-testdata.sh` extracts
//! the archive named in `testdata.lock` and writes `.testdata.stamp` (its sha256). Included by
//! the integration tests with `#[path = ".../testdata.rs"] mod testdata;`.
//!
//! A test that needs the data calls [`dir`] and FAILS (it never skips) when the data are
//! missing or do not match the lock, so CI cannot pass without them.
#![allow(dead_code)]
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn lock_sha256(root: &Path) -> String {
    let lock = root.join("testdata.lock");
    let txt = std::fs::read_to_string(&lock).unwrap_or_else(|e| panic!("{}: {e}", lock.display()));
    txt.lines()
        .find_map(|l| l.trim().strip_prefix("sha256="))
        .unwrap_or_else(|| panic!("{}: no sha256= line", lock.display()))
        .trim()
        .to_string()
}

/// Panic with instructions unless the stored test data match `testdata.lock`.
pub fn require() {
    let root = root();
    let want = lock_sha256(&root);
    let stamp = root.join(".testdata.stamp");
    match std::fs::read_to_string(&stamp) {
        Ok(s) if s.trim() == want => {}
        Ok(s) => panic!(
            "stored test data are outdated (stamp {}, testdata.lock {want}): run tools/fetch-testdata.sh",
            s.trim()
        ),
        Err(_) => panic!(
            "stored test data missing (crates/*/tests/data, {} absent): run tools/fetch-testdata.sh",
            stamp.display()
        ),
    }
}

/// `<this crate>/tests/data[/sub]`, after [`require`].
pub fn dir(sub: &str) -> PathBuf {
    require();
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    if sub.is_empty() { d } else { d.join(sub) }
}
