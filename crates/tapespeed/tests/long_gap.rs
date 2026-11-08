//! Long reference gaps (>= 20 s without valid samples): detection, edge-linear fill inside, ramp to
//! L1 at the file ends.
use ts_core::model::gaps::{fill_long_gaps_default, long_gaps_default};

fn mean(v: &[f64]) -> f64 {
    ts_dsp::stats::mean(v)
}

#[test]
fn test_long_gap_fallback() {
    let n = (300.0f64 / 0.005).round() as usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.005).collect();
    let r: Vec<f64> = t
        .iter()
        .map(|&t| 0.98 + 0.001 * (std::f64::consts::TAU * 6.8 * t).sin() + 2e-5 * t)
        .collect();
    let mut v: Vec<bool> = t
        .iter()
        .map(|&t| t < 30.0 || (60.0..80.0).contains(&t))
        .collect();
    v[(150.0f64 / 0.005) as usize] = true; // isolated stray 'valid' sample
    let g = long_gaps_default(&t, &v).unwrap();
    assert!(g.len() == 2 && (t[g[1].1 - 1] - 300.0).abs() < 0.01);
    let rin: Vec<f64> = r
        .iter()
        .zip(&v)
        .map(|(&x, &b)| if b { x } else { 5.0 })
        .collect();
    let (rf, vf, l1) = fill_long_gaps_default(&t, &rin, &v, &g).unwrap();
    assert!(rf.iter().all(|x| x.is_finite()));
    assert!(!(0..n).any(|i| vf[i] && ((t[i] > 31.0 && t[i] < 59.0) || t[i] > 81.0)));
    let mid: Vec<f64> = (0..n)
        .filter(|&i| t[i] > 35.0 && t[i] < 55.0)
        .map(|i| rf[i])
        .collect();
    let d: Vec<f64> = mid.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(ts_dsp::stats::ptp(&d).unwrap() < 1e-9); // straight line, no flutter
    let rv: Vec<f64> = (0..n).filter(|&i| v[i]).map(|i| r[i]).collect();
    assert!((l1 - mean(&rv)).abs() < 3e-4);
    assert!(
        (0..n)
            .filter(|&i| t[i] > 120.0)
            .all(|i| (rf[i] - l1).abs() < 1e-12)
    );
}
