//! numpy's default random generator, bit-exact: `np.random.default_rng(seed)` =
//! `Generator(PCG64(SeedSequence(seed)))`, as far as `holdout_fill` uses it
//! (`Generator.shuffle` of a 1-D array) plus `integers(0, k)` for the vector tests.
//!
//! - [`SeedSequence`]: numpy `bit_generator.pyx` (hashmix / mix over a 4-word uint32 pool,
//!   `generate_state` with `INIT_B`/`MULT_B`; uint64 words = little-endian uint32 pairs).
//! - [`Pcg64`]: PCG XSL-RR 128/64 (`pcg64.h`): seeded with `generate_state(4, uint64)`,
//!   `state = val[0] << 64 | val[1]`, `inc = val[2] << 64 | val[3]`, `srandom_r`; `next64`
//!   steps then outputs; `next32` buffers the high half of one 64-bit draw (low half first).
//! - [`Generator::shuffle`]: `_shuffle_raw`: Fisher-Yates from the end, `j =
//!   random_interval(i)` (smallest bit mask ≥ i, rejection; 32-bit draws while `i <=
//!   0xffffffff`).

const MASK32: u64 = 0xffff_ffff;
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

/// `numpy.random.SeedSequence(entropy)` (no spawn key).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedSequence {
    pub pool: [u32; POOL_SIZE],
}

#[inline]
fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

#[inline]
fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

impl SeedSequence {
    /// `SeedSequence(entropy)` for a non-negative integer seed.
    pub fn new(entropy: u128) -> Self {
        // _int_to_uint32_array: little-endian 32-bit words, [0] for 0
        let mut ent = Vec::new();
        let mut n = entropy;
        if n == 0 {
            ent.push(0u32);
        }
        while n > 0 {
            ent.push((n as u64 & MASK32) as u32);
            n >>= 32;
        }
        let mut pool = [0u32; POOL_SIZE];
        let mut hc = INIT_A;
        for (i, p) in pool.iter_mut().enumerate() {
            *p = hashmix(ent.get(i).copied().unwrap_or(0), &mut hc);
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let h = hashmix(pool[i_src], &mut hc);
                    pool[i_dst] = mix(pool[i_dst], h);
                }
            }
        }
        for &e in ent.iter().skip(POOL_SIZE) {
            for p in pool.iter_mut() {
                let h = hashmix(e, &mut hc);
                *p = mix(*p, h);
            }
        }
        SeedSequence { pool }
    }

    /// `generate_state(n_words, np.uint32)`.
    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut hc = INIT_B;
        (0..n_words)
            .map(|i| {
                let mut v = self.pool[i % POOL_SIZE] ^ hc;
                hc = hc.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hc);
                v ^ (v >> XSHIFT)
            })
            .collect()
    }

    /// `generate_state(n_words, np.uint64)` (pairs of uint32 words, low word first).
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        self.generate_state_u32(2 * n_words)
            .chunks_exact(2)
            .map(|w| w[0] as u64 | (w[1] as u64) << 32)
            .collect()
    }
}

/// PCG default 128-bit multiplier.
const PCG_MULT: u128 = (0x2360_ed05_1fc6_5da4u128 << 64) | 0x4385_df64_9fcc_f645u128;

/// `numpy.random.PCG64` bit generator state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcg64 {
    pub state: u128,
    pub inc: u128,
    has_u32: bool,
    u32buf: u32,
}

impl Pcg64 {
    /// `PCG64(seed)` = seeded from `SeedSequence(seed)`.
    pub fn new(seed: u128) -> Self {
        Self::from_seed_seq(&SeedSequence::new(seed))
    }

    pub fn from_seed_seq(ss: &SeedSequence) -> Self {
        let v = ss.generate_state_u64(4);
        let initstate = (v[0] as u128) << 64 | v[1] as u128;
        let initseq = (v[2] as u128) << 64 | v[3] as u128;
        let mut g = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            has_u32: false,
            u32buf: 0,
        };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    /// `random_raw()`: one 64-bit output (XSL-RR of the stepped state).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        let x = ((s >> 64) as u64) ^ (s as u64);
        x.rotate_right((s >> 122) as u32)
    }

    /// `next_uint32`: low half of a fresh 64-bit draw, the high half on the next call.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.has_u32 {
            self.has_u32 = false;
            return self.u32buf;
        }
        let n = self.next_u64();
        self.has_u32 = true;
        self.u32buf = (n >> 32) as u32;
        (n & MASK32) as u32
    }
}

/// `numpy.random.Generator` over [`Pcg64`].
#[derive(Debug, Clone)]
pub struct Generator {
    pub bitgen: Pcg64,
}

/// `np.random.default_rng(seed)`.
pub fn default_rng(seed: u128) -> Generator {
    Generator {
        bitgen: Pcg64::new(seed),
    }
}

impl Generator {
    /// `random_interval(bitgen, max)`: uniform in `[0, max]` by masked rejection.
    pub fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        if max <= MASK32 {
            loop {
                let v = self.bitgen.next_u32() as u64 & mask;
                if v <= max {
                    return v;
                }
            }
        }
        loop {
            let v = self.bitgen.next_u64() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// `Generator.shuffle(x)` for a 1-D array (in place).
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.random_interval(i as u64) as usize;
            x.swap(i, j);
        }
    }

    /// `Generator.integers(0, high)` (int64, scalar): Lemire's bounded draw, 32-bit while
    /// `high - 1 <= 0xffffffff`.
    pub fn integers(&mut self, high: u64) -> u64 {
        assert!(high >= 1, "high <= 0");
        let rng = high - 1;
        if rng == 0 {
            return 0;
        }
        if rng <= MASK32 {
            if rng == MASK32 {
                return self.bitgen.next_u32() as u64;
            }
            let excl = rng as u32 + 1;
            let mut m = self.bitgen.next_u32() as u64 * excl as u64;
            let mut left = m as u32;
            if left < excl {
                let thr = (u32::MAX - rng as u32) % excl;
                while left < thr {
                    m = self.bitgen.next_u32() as u64 * excl as u64;
                    left = m as u32;
                }
            }
            return m >> 32;
        }
        if rng == u64::MAX {
            return self.bitgen.next_u64();
        }
        let excl = rng + 1;
        let mut m = self.bitgen.next_u64() as u128 * excl as u128;
        let mut left = m as u64;
        if left < excl {
            let thr = (u64::MAX - rng) % excl;
            while left < thr {
                m = self.bitgen.next_u64() as u128 * excl as u128;
                left = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed0_state() {
        // numpy 2.5.4: PCG64(0).state (stored reference state)
        let g = Pcg64::new(0);
        assert_eq!(g.state, 35399562948360463058890781895381311971u128);
        assert_eq!(g.inc, 87136372517582989555478159403783844777u128);
    }

    #[test]
    fn small_shuffle() {
        let mut a: Vec<i64> = (0..10).collect();
        default_rng(0).shuffle(&mut a);
        let mut s = a.clone();
        s.sort();
        assert_eq!(s, (0..10).collect::<Vec<_>>());
    }
}
