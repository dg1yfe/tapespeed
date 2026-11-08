//! Number formatting with a fixed text form (part of the output formats): shortest round-trip
//! [`repr_shortest`], `%g`-style [`fmt_general`], fixed-point [`fmt_fixed`], decimal rounding
//! [`round_dec`] (correctly rounded, half-even on the exact binary value) and [`np_round`]
//! (`rint(x * 10^n) / 10^n`, the `numpy.round` definition).

/// Shortest round-trip digits and decimal exponent of a finite positive `x`: `x = d1.d2d3... * 10^e`.
/// Where the exact value is a tie between two shortest candidates (e.g. 1162837142450313.25),
/// the even last digit is taken (David Gay's dtoa, mode 0); Rust's `{:e}` may pick the other.
fn shortest_digits(x: f64) -> (String, i32) {
    let s = format!("{x:e}");
    let (m, e) = s.split_once('e').expect("LowerExp has an exponent");
    let d = m.replace('.', "");
    let e: i32 = e.parse().expect("exponent");
    if d.len() >= 15 {
        fix_tie(x, d, e)
    } else {
        (d, e)
    }
}

fn fix_tie(x: f64, d: String, e: i32) -> (String, i32) {
    let n = d.len();
    // exact expansion to n+1 digits; a tie iff digit n+1 is 5 and nothing follows
    let long = format!("{:.*e}", n + 25, x);
    let (m, e2) = long.split_once('e').expect("exponent");
    let m = m.replace('.', "");
    let e2: i32 = e2.parse().expect("exponent");
    if m.as_bytes()[n] != b'5' || m[n + 1..].bytes().any(|c| c != b'0') {
        return (d, e);
    }
    let trunc: u64 = m[..n].parse().expect("digits");
    for c in [trunc, trunc + 1] {
        let ok = format!("{c}e{}", e2 - (n as i32 - 1))
            .parse::<f64>()
            .is_ok_and(|v| v == x);
        if ok && c % 2 == 0 {
            let mut t = c.to_string();
            let ee = if t.len() > n { e2 + 1 } else { e2 };
            t.truncate(t.trim_end_matches('0').len().max(1));
            return (t, ee);
        }
    }
    (d, e)
}

/// Digits (`prec` significant, correctly rounded, half-even on the exact value) and exponent of finite `x >= 0`.
fn rounded_digits(x: f64, prec: usize) -> (String, i32) {
    let s = format!("{:.*e}", prec - 1, x);
    let (m, e) = s.split_once('e').expect("LowerExp has an exponent");
    (m.replace('.', ""), e.parse().expect("exponent"))
}

fn exp_suffix(e: i32) -> String {
    format!("e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}

fn special(x: f64) -> Option<&'static str> {
    if x.is_nan() {
        Some("nan")
    } else if x == f64::INFINITY {
        Some("inf")
    } else if x == f64::NEG_INFINITY {
        Some("-inf")
    } else {
        None
    }
}

/// Shortest round-trip text of a float: positional for decimal exponents -4..=15, always with a
/// fractional part (`30.0`), otherwise scientific with an at least two-digit signed exponent
/// (`1e-05`, `1.5e+300`); `nan`, `inf`, `-inf`, `-0.0`.
pub fn repr_shortest(x: f64) -> String {
    if let Some(s) = special(x) {
        return s.to_string();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    let (d, e) = shortest_digits(x.abs());
    let decpt = e + 1;
    let n = d.len() as i32;
    let body = if (-4..16).contains(&e) {
        if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), d)
        } else if decpt >= n {
            format!("{}{}.0", d, "0".repeat((decpt - n) as usize))
        } else {
            format!("{}.{}", &d[..decpt as usize], &d[decpt as usize..])
        }
    } else {
        let rest = if n > 1 {
            format!(".{}", &d[1..])
        } else {
            String::new()
        };
        format!("{}{}{}", &d[..1], rest, exp_suffix(e))
    };
    format!("{sign}{body}")
}

/// `%.{prec}g`: `prec` significant digits (half-even on the exact value), positional for
/// exponents -4..prec, otherwise scientific (`1e+05`); trailing zeros stripped.
pub fn fmt_general(x: f64, prec: usize) -> String {
    if let Some(s) = special(x) {
        return s.to_string();
    }
    let p = prec.max(1);
    let sign = if x.is_sign_negative() { "-" } else { "" };
    let (d, e) = rounded_digits(x.abs(), p);
    let d = d.trim_end_matches('0');
    let d = if d.is_empty() { "0" } else { d };
    let n = d.len() as i32;
    let body = if e >= -4 && e < p as i32 {
        let decpt = e + 1;
        if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), d)
        } else if decpt >= n {
            format!("{}{}", d, "0".repeat((decpt - n) as usize))
        } else {
            format!("{}.{}", &d[..decpt as usize], &d[decpt as usize..])
        }
    } else {
        let rest = if n > 1 {
            format!(".{}", &d[1..])
        } else {
            String::new()
        };
        format!("{}{}{}", &d[..1], rest, exp_suffix(e))
    };
    format!("{sign}{body}")
}

/// Fixed-point with `n` decimals (exact, half-even on the binary value); `nan`, `inf`, `-inf`.
pub fn fmt_fixed(x: f64, n: usize) -> String {
    match special(x) {
        Some(s) => s.to_string(),
        None => format!("{x:.n$}"),
    }
}

/// Round to `n` decimals (negative `n`: to a multiple of `10^-n`), correctly rounded from the
/// exact binary value, ties to even; non-finite values and zero pass through.
pub fn round_dec(x: f64, n: i32) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    if n > 323 {
        return x;
    }
    if n < -308 {
        return 0.0 * x;
    }
    if n >= 0 {
        return format!("{x:.*}", n as usize).parse().expect("float");
    }
    // negative ndigits: round to a multiple of 10^-n
    let (_, e) = rounded_digits(x.abs(), 1);
    let sig = e + n + 1; // significant digits that survive
    if sig >= 1 {
        format!("{:.*e}", (sig - 1) as usize, x)
            .parse()
            .expect("float")
    } else if sig == 0 && x.abs() > 5.0 * 10f64.powi(-n - 1) {
        (10f64).powi(-n).copysign(x)
    } else {
        0.0f64.copysign(x)
    }
}

fn pow10(n: i32) -> f64 {
    format!("1e{n}").parse().expect("float")
}

/// `numpy.round(x, n)` for float64: `rint(x * 10^n) / 10^n` (half-even); `n < 0`: `rint(x / 10^-n) * 10^-n`.
pub fn np_round(x: f64, n: i32) -> f64 {
    if n >= 0 {
        let f = pow10(n);
        (x * f).round_ties_even() / f
    } else {
        let f = pow10(-n);
        (x / f).round_ties_even() * f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_cases() {
        for (x, s) in [
            (30.0, "30.0"),
            (1e-5, "1e-05"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (1.5e300, "1.5e+300"),
            (f64::NAN, "nan"),
            (f64::NEG_INFINITY, "-inf"),
            (123456789.125, "123456789.125"),
            (5e-324, "5e-324"),
            (1e22, "1e+22"),
        ] {
            assert_eq!(repr_shortest(x), s);
        }
    }

    #[test]
    fn g_cases() {
        assert_eq!(fmt_general(100000.0, 6), "100000");
        assert_eq!(fmt_general(1e6, 6), "1e+06");
        assert_eq!(fmt_general(0.0001234, 3), "0.000123");
        assert_eq!(fmt_general(0.0, 6), "0");
        assert_eq!(fmt_general(9.9999995, 6), "10");
        assert_eq!(fmt_general(8000.0, 6), "8000");
    }

    #[test]
    fn round_cases() {
        assert_eq!(round_dec(2.675, 2), 2.67);
        assert_eq!(round_dec(0.5, 0), 0.0);
        assert_eq!(round_dec(1234.0, -2), 1200.0);
        assert_eq!(round_dec(1250.0, -2), 1200.0);
        assert_eq!(round_dec(1251.0, -2), 1300.0);
        assert_eq!(np_round(2.675, 2), 2.68);
        assert_eq!(np_round(0.5, 0), 0.0);
        assert_eq!(np_round(1250.0, -2), 1200.0);
    }
}
