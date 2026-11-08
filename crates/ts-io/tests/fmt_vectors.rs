#[path = "../../testdata.rs"]
mod testdata;
use ts_io::json::Value;
use ts_io::*;

fn bits(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn fmt_matches_vectors() {
    let path = std::env::var("TS_FMT_VECTORS").unwrap_or_else(|_| {
        testdata::dir("fmt_vectors.tsv")
            .to_string_lossy()
            .into_owned()
    });
    let txt = std::fs::read_to_string(&path).unwrap();
    let (mut n, mut bad) = (0usize, 0usize);
    for line in txt.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let x = bits(c[0]);
        let mut chk = |what: &str, got: String, want: &str| {
            if got != want {
                bad += 1;
                if bad < 30 {
                    eprintln!("{what}({x:e} {}): got {got} want {want}", c[0]);
                }
            }
        };
        chk("repr", repr_shortest(x), c[1]);
        chk("g6", fmt_general(x, 6), c[2]);
        chk("g3", fmt_general(x, 3), c[3]);
        chk("g10", fmt_general(x, 10), c[4]);
        chk("f2", fmt_fixed(x, 2), c[5]);
        chk("f4", fmt_fixed(x, 4), c[6]);
        chk("f6", fmt_fixed(x, 6), c[7]);
        chk("f9", fmt_fixed(x, 9), c[8]);
        for (i, d) in [2, 3, 6].into_iter().enumerate() {
            if !same(round_dec(x, d), bits(c[9 + i])) {
                bad += 1;
                eprintln!(
                    "round_dec({x:e},{d}): got {:e} want {:e}",
                    round_dec(x, d),
                    bits(c[9 + i])
                );
            }
            if !same(np_round(x, d), bits(c[12 + i])) {
                bad += 1;
                eprintln!(
                    "np_round({x:e},{d}): got {:e} want {:e}",
                    np_round(x, d),
                    bits(c[12 + i])
                );
            }
        }
        n += 1;
    }
    eprintln!("{n} vectors, {bad} mismatches");
    assert_eq!(bad, 0);
}

fn doc() -> Value {
    let nest = Value::obj()
        .with(
            "z",
            Value::obj().with(
                "y",
                Value::Arr(vec![
                    Value::Arr(vec![]),
                    Value::obj(),
                    Value::Arr(vec![1i64.into(), Value::Arr(vec![2i64.into()])]),
                ]),
            ),
        )
        .with("x", 123456789012345678i64);
    Value::obj()
        .with("b", 1i64)
        .with(
            "a",
            vec![
                1.0,
                2.5,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                1e-5,
                1e16,
                -0.0,
                30.0,
            ],
        )
        .with("s", "q\"\\/\n\t\u{e4}\u{20ac}\u{1d11e}\u{1}")
        .with("t", true)
        .with("f", false)
        .with("n", Value::Null)
        .with("e", Value::Arr(vec![]))
        .with("o", Value::obj())
        .with("nest", nest)
        .with("mix", vec![0.1, 1.0 / 3.0, 123456.789e10, 5e-324])
}

#[test]
fn json_matches_reference() {
    let d = format!("{}/", testdata::dir("").display());
    assert_eq!(
        doc().dumps(Some(1)),
        std::fs::read_to_string(format!("{d}json_indent1.txt")).unwrap()
    );
    assert_eq!(
        doc().dumps(None),
        std::fs::read_to_string(format!("{d}json_compact.txt")).unwrap()
    );
}
