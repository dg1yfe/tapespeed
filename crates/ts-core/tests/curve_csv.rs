//! Stored reference curve CSVs must be rewritten byte-identically.
#[path = "../../testdata.rs"]
mod testdata;
use ts_core::curve::*;

fn check(path: &std::path::Path) {
    let orig = std::fs::read_to_string(path).unwrap();
    let c = parse_curve(&orig).unwrap();
    assert_eq!(curve_to_string(&c), orig, "{}", path.display());
}

#[test]
fn committed_fixture() {
    check(&testdata::dir("ref_curve.csv"));
}
