//! One FLAC frame (RFC 9639): fixed-blocksize header, per channel constant / verbatim / fixed
//! (order 0–4) / LPC (Levinson on a Welch-windowed autocorrelation, order ≤ 8 by libFLAC's
//! expected-bits estimate, 15-bit coefficients) subframes, partitioned Rice (order ≤ 8, RICE2 when a
//! parameter exceeds 14); stereo mode chosen like ffmpeg (order-2 residual sums of L, R, M, S).

/// Highest LPC order tried (ffmpeg's default).
const MAX_LPC_ORDER: usize = 8;
/// Quantised coefficient precision in bits (incl. sign).
const QLP_PRECISION: u32 = 15;
const MAX_PARTITION_ORDER: u32 = 8;

const fn crc8_table() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u8;
        let mut b = 0;
        while b < 8 {
            c = if c & 0x80 != 0 {
                (c << 1) ^ 0x07
            } else {
                c << 1
            };
            b += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

const fn crc16_table() -> [u16; 256] {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = (i as u16) << 8;
        let mut b = 0;
        while b < 8 {
            c = if c & 0x8000 != 0 {
                (c << 1) ^ 0x8005
            } else {
                c << 1
            };
            b += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

/// slice-by-8 tables: `[k][v]` = CRC of byte `v` followed by `k` zero bytes
const fn crc16_slices() -> [[u16; 256]; 8] {
    let mut t = [[0u16; 256]; 8];
    t[0] = crc16_table();
    let mut k = 1;
    while k < 8 {
        let mut v = 0;
        while v < 256 {
            let c = t[k - 1][v];
            t[k][v] = (c << 8) ^ t[0][(c >> 8) as usize];
            v += 1;
        }
        k += 1;
    }
    t
}

static CRC8: [u8; 256] = crc8_table();
static CRC16: [[u16; 256]; 8] = crc16_slices();

fn crc8(b: &[u8]) -> u8 {
    b.iter().fold(0, |c, &x| CRC8[(c ^ x) as usize])
}

fn crc16_bytewise(c: u16, b: &[u8]) -> u16 {
    b.iter().fold(c, |c, &x| {
        (c << 8) ^ CRC16[0][((c >> 8) as u8 ^ x) as usize]
    })
}

fn crc16(b: &[u8]) -> u16 {
    let t = &CRC16;
    let mut c = 0u16;
    let mut it = b.chunks_exact(8);
    for q in &mut it {
        c = t[7][((c >> 8) as u8 ^ q[0]) as usize]
            ^ t[6][(c as u8 ^ q[1]) as usize]
            ^ t[5][q[2] as usize]
            ^ t[4][q[3] as usize]
            ^ t[3][q[4] as usize]
            ^ t[2][q[5] as usize]
            ^ t[1][q[6] as usize]
            ^ t[0][q[7] as usize];
    }
    crc16_bytewise(c, it.remainder())
}

/// MSB-first bit writer.
struct Bits {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    fn new(cap: usize) -> Bits {
        Bits {
            out: Vec::with_capacity(cap),
            acc: 0,
            n: 0,
        }
    }

    /// low `bits` (≤ 32) of `v`; whole 32-bit words go out (`n` < 32 between calls)
    #[inline]
    fn put(&mut self, bits: u32, v: u64) {
        debug_assert!(bits <= 32 && self.n < 32);
        if bits == 0 {
            return;
        }
        self.acc = (self.acc << bits) | (v & ((1u64 << bits) - 1));
        self.n += bits;
        if self.n >= 32 {
            self.n -= 32;
            self.out
                .extend_from_slice(&((self.acc >> self.n) as u32).to_be_bytes());
        }
    }

    /// Emit the complete pending bytes (call when byte-aligned to read `out`).
    fn flush(&mut self) {
        while self.n >= 8 {
            self.n -= 8;
            self.out.push((self.acc >> self.n) as u8);
        }
    }

    #[inline]
    fn signed(&mut self, bits: u32, v: i64) {
        self.put(bits, v as u64);
    }

    /// `q` zeros then a one, then the low `k` bits of `low`
    #[inline]
    fn rice(&mut self, mut q: u64, k: u32, low: u64) {
        while q >= 32 {
            self.put(32, 0);
            q -= 32;
        }
        let q = q as u32;
        if q + 1 + k <= 32 {
            self.put(q + 1 + k, (1u64 << k) | (low & ((1u64 << k) - 1)));
        } else {
            self.put(q + 1, 1);
            self.put(k, low);
        }
    }

    fn align(&mut self) {
        let pad = (8 - self.n % 8) % 8;
        self.put(pad, 0);
        self.flush();
    }
}

#[inline]
fn zigzag(e: i64) -> u64 {
    ((e << 1) ^ (e >> 63)) as u64
}

/// ffmpeg `find_optimal_param`: Rice parameter for `n` values summing to `sum`.
#[inline]
fn rice_param(sum: u64, n: u64) -> u32 {
    if n == 0 || sum <= n >> 1 {
        return 0;
    }
    let m = (sum - (n >> 1)) / n;
    (63 - m.max(1).leading_zeros()).min(30)
}

/// ffmpeg `rice_encode_count`: estimated bits of `n` values summing to `sum` with parameter `k`.
#[inline]
fn rice_bits(sum: u64, n: u64, k: u32) -> u64 {
    n * (k as u64 + 1) + (sum.saturating_sub(n >> 1) >> k)
}

/// Chosen partitioning of a residual.
struct Rice {
    order: u32,
    params: Vec<u32>,
    bits: u64,
}

/// Best partition order (≤ 8, partitions of `n >> po` with the first `pred_order` samples missing).
fn plan_rice(u: &[u64], n: usize, pred_order: usize) -> Rice {
    let mut max_po = 0;
    while max_po < MAX_PARTITION_ORDER
        && n.is_multiple_of(1 << (max_po + 1))
        && (n >> (max_po + 1)) > pred_order
    {
        max_po += 1;
    }
    // sums on the finest level, then merged upward
    let parts = 1usize << max_po;
    let plen = n >> max_po;
    let mut sums: Vec<u64> = (0..parts)
        .map(|p| {
            let a = (p * plen).saturating_sub(pred_order);
            let b = (p + 1) * plen - pred_order;
            u[a..b].iter().sum()
        })
        .collect();
    let mut best: Option<Rice> = None;
    let mut po = max_po;
    loop {
        let parts = 1usize << po;
        let plen = (n >> po) as u64;
        let mut params = Vec::with_capacity(parts);
        let mut bits = 0u64;
        for (p, &s) in sums.iter().enumerate() {
            let len = if p == 0 {
                plen - pred_order as u64
            } else {
                plen
            };
            let k = rice_param(s, len);
            bits += rice_bits(s, len, k);
            params.push(k);
        }
        let pbits = if params.iter().any(|&k| k > 14) { 5 } else { 4 };
        bits += pbits * parts as u64 + 6; // + method (2) + partition order (4)
        if best.as_ref().is_none_or(|b| bits < b.bits) {
            best = Some(Rice {
                order: po,
                params,
                bits,
            });
        }
        if po == 0 {
            break;
        }
        po -= 1;
        sums = sums.chunks(2).map(|c| c[0] + c[1]).collect();
    }
    best.expect("at least order 0")
}

fn write_residual(w: &mut Bits, u: &[u64], n: usize, pred_order: usize, r: &Rice) {
    let rice2 = r.params.iter().any(|&k| k > 14);
    w.put(2, rice2 as u64);
    w.put(4, r.order as u64);
    let plen = n >> r.order;
    let mut i = 0;
    for (p, &k) in r.params.iter().enumerate() {
        w.put(if rice2 { 5 } else { 4 }, k as u64);
        let len = if p == 0 { plen - pred_order } else { plen };
        for &v in &u[i..i + len] {
            w.rice(v >> k, k, v);
        }
        i += len;
    }
}

/// Fixed-predictor residual of order `o` (0–4) for samples `o..n`, zigzagged.
fn fixed_residual(x: &[i32], o: usize, u: &mut Vec<u64>) {
    u.clear();
    u.extend((o..x.len()).map(|i| {
        let a = |j: usize| x[i - j] as i64;
        zigzag(match o {
            0 => a(0),
            1 => a(0) - a(1),
            2 => a(0) - 2 * a(1) + a(2),
            3 => a(0) - 3 * a(1) + 3 * a(2) - a(3),
            _ => a(0) - 4 * a(1) + 6 * a(2) - 4 * a(3) + a(4),
        })
    }));
}

/// Sum of |residual| of fixed orders 0–4 over samples 4..n (libFLAC `compute_best_predictor`).
fn fixed_sums(x: &[i32]) -> [u64; 5] {
    let mut s = [0u64; 5];
    let v = |i: usize| x[i] as i64;
    let (mut e1p, mut e2p, mut e3p) = (
        v(3) - v(2),
        v(3) - 2 * v(2) + v(1),
        v(3) - 3 * v(2) + 3 * v(1) - v(0),
    );
    for i in 4..x.len() {
        let e0 = v(i);
        let e1 = e0 - v(i - 1);
        let e2 = e1 - e1p;
        let e3 = e2 - e2p;
        let e4 = e3 - e3p;
        s[0] += e0.unsigned_abs();
        s[1] += e1.unsigned_abs();
        s[2] += e2.unsigned_abs();
        s[3] += e3.unsigned_abs();
        s[4] += e4.unsigned_abs();
        (e1p, e2p, e3p) = (e1, e2, e3);
    }
    s
}

/// Quantised LPC: coefficients for x[i-1], x[i-2], …, and the right shift.
struct Qlp {
    coef: Vec<i32>,
    shift: u32,
}

/// libFLAC `lpc_quantize_coefficients` (error feedback, 15-bit, shift 0..=15); None if unusable.
fn quantize(lp: &[f64]) -> Option<Qlp> {
    let prec = QLP_PRECISION - 1;
    let (qmin, qmax) = (-(1i64 << prec), (1i64 << prec) - 1);
    let cmax = lp.iter().fold(0.0f64, |m, c| m.max(c.abs()));
    if cmax <= 0.0 || !cmax.is_finite() {
        return None;
    }
    // cmax in [2^l, 2^(l+1))
    let l = cmax.log2().floor() as i32;
    let shift = (prec as i32 - l - 1).min(15);
    if shift < 0 {
        return None;
    }
    let mut err = 0.0f64;
    let coef = lp
        .iter()
        .map(|&c| {
            err += c * (1u64 << shift) as f64;
            let q = (err.round() as i64).clamp(qmin, qmax);
            err -= q as f64;
            q as i32
        })
        .collect();
    Some(Qlp {
        coef,
        shift: shift as u32,
    })
}

/// LPC residual for samples `order..n`, zigzagged; None if a residual leaves i32.
fn lpc_residual(x: &[i32], q: &Qlp, u: &mut Vec<u64>) -> Option<()> {
    fn run<const O: usize>(x: &[i32], coef: &[i32], shift: u32, u: &mut Vec<u64>) -> bool {
        let c: [i64; O] = std::array::from_fn(|j| coef[j] as i64);
        let (mut lo, mut hi) = (0i64, 0i64);
        u.extend(x.windows(O + 1).map(|w| {
            // w[O] is the sample, w[O - 1 - j] = x[i - 1 - j]
            let mut acc = 0i64;
            for j in 0..O {
                acc += c[j] * w[O - 1 - j] as i64;
            }
            let e = w[O] as i64 - (acc >> shift);
            lo = lo.min(e);
            hi = hi.max(e);
            zigzag(e)
        }));
        lo >= i32::MIN as i64 && hi <= i32::MAX as i64
    }
    u.clear();
    let (c, s) = (&q.coef[..], q.shift);
    let ok = match c.len() {
        1 => run::<1>(x, c, s, u),
        2 => run::<2>(x, c, s, u),
        3 => run::<3>(x, c, s, u),
        4 => run::<4>(x, c, s, u),
        5 => run::<5>(x, c, s, u),
        6 => run::<6>(x, c, s, u),
        7 => run::<7>(x, c, s, u),
        8 => run::<8>(x, c, s, u),
        o => unreachable!("lpc order {o}"),
    };
    ok.then_some(())
}

/// Levinson-Durbin on a Welch-windowed autocorrelation: predictors of order 1..=max with their
/// prediction errors (x̂[i] = Σ a[j] x[i-1-j]).
fn lpc_orders(x: &[i32], max: usize) -> Vec<(Vec<f64>, f64)> {
    let n = x.len();
    let c = (n - 1) as f64 / 2.0;
    let inv = 1.0 / (c + 1.0);
    let w: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let t = (i as f64 - c) * inv;
            v as f64 * (1.0 - t * t)
        })
        .collect();
    let r: Vec<f64> = (0..=max).map(|k| dot(&w[k..], &w[..n - k])).collect();
    let mut out = Vec::with_capacity(max);
    let mut a: Vec<f64> = Vec::with_capacity(max);
    let mut err = r[0];
    for m in 1..=max {
        if err <= 0.0 || err.is_nan() {
            break;
        }
        let acc = r[m] - (0..m - 1).map(|j| a[j] * r[m - 1 - j]).sum::<f64>();
        let k = acc / err;
        let prev = a.clone();
        for j in 0..m - 1 {
            a[j] = prev[j] - k * prev[m - 2 - j];
        }
        a.push(k);
        err *= 1.0 - k * k;
        out.push((a.clone(), err));
    }
    out
}

/// Dot product with 8 independent accumulators (vectorisable).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = [0.0f64; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f64 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(p, q)| p * q)
        .sum();
    for (p, q) in ca.zip(cb) {
        for l in 0..8 {
            s[l] += p[l] * q[l];
        }
    }
    s.iter().sum::<f64>() + tail
}

/// libFLAC `compute_best_order`: expected bits by the prediction error, plus coefficient overhead.
fn best_order(orders: &[(Vec<f64>, f64)], n: usize, bps: u32) -> Option<usize> {
    let per = |err: f64, len: usize| {
        if err > 0.0 {
            (0.5 * (0.5 / len as f64 * err).log2()).max(0.0)
        } else if err < 0.0 {
            1e32
        } else {
            0.0
        }
    };
    (0..orders.len())
        .map(|i| {
            let o = i + 1;
            let bits =
                per(orders[i].1, n) * (n - o) as f64 + (o as u32 * (QLP_PRECISION + bps)) as f64;
            (i, bits)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i + 1)
}

enum Kind {
    Verbatim,
    Fixed(usize),
    Lpc(Qlp),
}

/// Encode one subframe of `bps`-bit samples (smallest of the candidates).
fn subframe(w: &mut Bits, x: &[i32], bps: u32) {
    let n = x.len();
    if x.iter().all(|&v| v == x[0]) {
        w.put(8, 0);
        w.signed(bps, x[0] as i64);
        return;
    }
    let mut best = (
        Kind::Verbatim,
        n as u64 * bps as u64,
        Vec::new(),
        None::<Rice>,
    );
    let mut u = Vec::with_capacity(n);
    if n > 4 {
        let s = fixed_sums(x);
        let o = (0..5).min_by_key(|&o| s[o]).expect("5 orders");
        fixed_residual(x, o, &mut u);
        let r = plan_rice(&u, n, o);
        let bits = o as u64 * bps as u64 + r.bits;
        if bits < best.1 {
            best = (Kind::Fixed(o), bits, std::mem::take(&mut u), Some(r));
        }
    }
    let max = MAX_LPC_ORDER.min(n.saturating_sub(1) / 2);
    if max >= 1 && n >= 32 {
        let orders = lpc_orders(x, max);
        if let Some(o) = best_order(&orders, n, bps)
            && let Some(q) = quantize(&orders[o - 1].0)
            && lpc_residual(x, &q, &mut u).is_some()
        {
            let r = plan_rice(&u, n, o);
            let bits = o as u64 * (bps + QLP_PRECISION) as u64 + 9 + r.bits;
            if bits < best.1 {
                best = (Kind::Lpc(q), bits, u, Some(r));
            }
        }
    }
    match best {
        (Kind::Verbatim, ..) => {
            w.put(8, 0b0000_0010);
            for &v in x {
                w.signed(bps, v as i64);
            }
        }
        (Kind::Fixed(o), _, u, Some(r)) => {
            w.put(8, (0b0000_1000 | o as u64) << 1);
            for &v in &x[..o] {
                w.signed(bps, v as i64);
            }
            write_residual(w, &u, n, o, &r);
        }
        (Kind::Lpc(q), _, u, Some(r)) => {
            let o = q.coef.len();
            w.put(8, (0b0010_0000 | (o as u64 - 1)) << 1);
            for &v in &x[..o] {
                w.signed(bps, v as i64);
            }
            w.put(4, (QLP_PRECISION - 1) as u64);
            w.put(5, q.shift as u64);
            for &c in &q.coef {
                w.signed(QLP_PRECISION, c as i64);
            }
            write_residual(w, &u, n, o, &r);
        }
        _ => unreachable!("predictive kinds carry a rice plan"),
    }
}

/// ffmpeg `estimate_stereo_mode`: 0 independent, 1 left/side, 2 right/side, 3 mid/side.
fn stereo_mode(l: &[i32], r: &[i32]) -> u32 {
    let n = l.len();
    if n < 3 {
        return 0;
    }
    let mut s = [0u64; 4];
    for i in 2..n {
        let lt = l[i] as i64 - 2 * l[i - 1] as i64 + l[i - 2] as i64;
        let rt = r[i] as i64 - 2 * r[i - 1] as i64 + r[i - 2] as i64;
        s[2] += ((lt + rt) >> 1).unsigned_abs();
        s[3] += (lt - rt).unsigned_abs();
        s[0] += lt.unsigned_abs();
        s[1] += rt.unsigned_abs();
    }
    let b = s.map(|v| {
        let k = rice_param(2 * v, n as u64);
        rice_bits(2 * v, n as u64, k)
    });
    let score = [b[0] + b[1], b[0] + b[3], b[1] + b[3], b[2] + b[3]];
    (0..4).min_by_key(|&m| score[m]).expect("4 modes") as u32
}

fn sample_rate_code(sr: u32) -> (u64, u32, u64) {
    match sr {
        88200 => (1, 0, 0),
        176400 => (2, 0, 0),
        192000 => (3, 0, 0),
        8000 => (4, 0, 0),
        16000 => (5, 0, 0),
        22050 => (6, 0, 0),
        24000 => (7, 0, 0),
        32000 => (8, 0, 0),
        44100 => (9, 0, 0),
        48000 => (10, 0, 0),
        96000 => (11, 0, 0),
        _ if sr.is_multiple_of(1000) && sr / 1000 <= 255 => (12, 8, (sr / 1000) as u64),
        _ if sr <= 65535 => (13, 16, sr as u64),
        _ if sr.is_multiple_of(10) && sr / 10 <= 65535 => (14, 16, (sr / 10) as u64),
        _ => (0, 0, 0), // from STREAMINFO
    }
}

fn utf8_number(w: &mut Bits, v: u64) {
    if v < 0x80 {
        w.put(8, v);
        return;
    }
    // bytes needed: 2 for 11 payload bits, +1 per 5 more bits
    let n = [0x800u64, 0x1_0000, 0x20_0000, 0x400_0000]
        .iter()
        .position(|&lim| v < lim)
        .map_or(6, |p| p + 2);
    let lead = (0xFF00u64 >> n) & 0xFF;
    w.put(8, lead | (v >> (6 * (n - 1))));
    for i in (0..n - 1).rev() {
        w.put(8, 0x80 | ((v >> (6 * i)) & 0x3F));
    }
}

/// Encode one frame: `x` interleaved `ch`-channel samples of `bps` bits, frame number `num`.
pub fn encode(x: &[i32], ch: usize, bps: u32, sr: u32, num: u64) -> Vec<u8> {
    let n = x.len() / ch;
    assert!((1..=65536).contains(&n) && n * ch == x.len() && (1..=8).contains(&ch));
    let mut chans: Vec<Vec<i32>> = (0..ch)
        .map(|c| x.iter().skip(c).step_by(ch).copied().collect())
        .collect();
    let mut bpss = vec![bps; ch];
    let mut assign = ch as u64 - 1;
    if ch == 2 {
        let mode = stereo_mode(&chans[0], &chans[1]);
        if mode != 0 {
            let side: Vec<i32> = chans[0].iter().zip(&chans[1]).map(|(l, r)| l - r).collect();
            match mode {
                1 => {
                    chans[1] = side;
                    bpss[1] += 1;
                }
                2 => {
                    chans[0] = side;
                    bpss[0] += 1;
                }
                _ => {
                    let mid = chans[0]
                        .iter()
                        .zip(&chans[1])
                        .map(|(&l, &r)| ((l as i64 + r as i64) >> 1) as i32)
                        .collect();
                    chans = vec![mid, side];
                    bpss[1] += 1;
                }
            }
            assign = 7 + mode as u64;
        }
    }
    let mut w = Bits::new(x.len() * 3 + 64);
    w.put(16, 0xFFF8);
    let (bs_code, bs_extra) = match n {
        4096 => (12, 0),
        _ if n <= 256 => (6, 8),
        _ => (7, 16),
    };
    let (sr_code, sr_bits, sr_val) = sample_rate_code(sr);
    w.put(4, bs_code);
    w.put(4, sr_code);
    w.put(4, assign);
    let bps_code = match bps {
        8 => 1,
        12 => 2,
        16 => 4,
        20 => 5,
        24 => 6,
        _ => 0,
    };
    w.put(3, bps_code);
    w.put(1, 0);
    utf8_number(&mut w, num);
    w.put(bs_extra, n as u64 - 1);
    w.put(sr_bits, sr_val);
    w.flush();
    debug_assert_eq!(w.n, 0);
    let h = crc8(&w.out);
    w.put(8, h as u64);
    for (c, b) in chans.iter().zip(&bpss) {
        subframe(&mut w, c, *b);
    }
    w.align();
    let c = crc16(&w.out);
    w.put(16, c as u64);
    w.flush();
    w.out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crcs() {
        assert_eq!(crc8(b"123456789"), 0xF4); // CRC-8/SMBUS check value
        assert_eq!(crc16(b"123456789"), 0xFEE8); // CRC-16/UMTS (BUYPASS)
        let v: Vec<u8> = (0..1000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 999, 1000] {
            assert_eq!(crc16(&v[..n]), crc16_bytewise(0, &v[..n]), "len {n}");
        }
    }

    #[test]
    fn utf8_numbers() {
        for (v, want) in [
            (0u64, vec![0u8]),
            (0x7F, vec![0x7F]),
            (0x80, vec![0xC2, 0x80]),
            (0x7FF, vec![0xDF, 0xBF]),
            (0x800, vec![0xE0, 0xA0, 0x80]),
            (0xFFFF, vec![0xEF, 0xBF, 0xBF]),
            (0x10000, vec![0xF0, 0x90, 0x80, 0x80]),
        ] {
            let mut w = Bits::new(8);
            utf8_number(&mut w, v);
            w.flush();
            assert_eq!(w.out, want, "{v:#x}");
        }
    }

    #[test]
    fn levinson_ar1() {
        // x[i] = 0.9 x[i-1] (+ tiny) -> order-1 coefficient near 0.9
        let mut v = 1_000_000.0f64;
        let x: Vec<i32> = (0..4096)
            .map(|i| {
                v = 0.9 * v + if i % 97 == 0 { 300_000.0 } else { 0.0 };
                v as i32
            })
            .collect();
        let o = lpc_orders(&x, 2);
        assert!((o[0].0[0] - 0.9).abs() < 0.05, "{:?}", o[0].0);
    }
}
