//! NumPy-compatible `default_rng(0)` bit-exactness (SeedSequence, PCG64 raw, shuffle, integers)
//! against stored reference vectors (tests/data/i/rng) and the two shuffles holdout_fill made on
//! mech_plain.
mod i_common;
use i_common::*;
use ts_core::model::rng::{Pcg64, SeedSequence, default_rng};

fn rng(name: &str) -> std::path::PathBuf {
    data(&format!("rng/{name}"))
}

#[test]
fn seed_sequence() {
    let ss = SeedSequence::new(0);
    assert_eq!(
        ss.generate_state_u32(8),
        npy_u32(&rng("seedseq0_state_u32x8.npy"))
    );
    assert_eq!(
        ss.generate_state_u64(4),
        npy_u64(&rng("seedseq0_state_u64x4.npy"))
    );
}

#[test]
fn pcg64_raw() {
    let mut g = Pcg64::new(0);
    let want = npy_u64(&rng("pcg64_seed0_raw.npy"));
    let got: Vec<u64> = (0..want.len()).map(|_| g.next_u64()).collect();
    assert_eq!(got, want);
}

#[test]
fn shuffle_sequential() {
    let mut g = default_rng(0);
    for (k, n) in [11000i64, 11500, 3, 700].into_iter().enumerate() {
        let mut a: Vec<i64> = (0..n).collect();
        g.shuffle(&mut a);
        let want = npy_i64(&rng(&format!("shuffle_seq_seed0_{}_n{n}.npy", k + 1)));
        assert_eq!(a, want, "shuffle #{}", k + 1);
    }
}

#[test]
fn integers() {
    let mut g = default_rng(0);
    let got: Vec<i64> = [1u64, 2, 3, 7, 10, 1000, 1 << 31, 1 << 40]
        .iter()
        .map(|&k| g.integers(k) as i64)
        .collect();
    assert_eq!(got, npy_i64(&rng("integers_seed0.npy")));
}

/// mech_plain holdout_fill: one generator, two sequential shuffles of the candidate indices.
#[test]
fn holdout_shuffles_mech_plain() {
    let mut g = default_rng(0);
    for k in 0..2 {
        let mut a = npy_i64(&rng(&format!("holdout0_shuffle{k}_before.npy")));
        g.shuffle(&mut a);
        assert_eq!(
            a,
            npy_i64(&rng(&format!("holdout0_shuffle{k}_after.npy"))),
            "shuffle {k}"
        );
    }
}
