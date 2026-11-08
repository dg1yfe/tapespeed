//! Model benchmark: `fit_hier` + `explained` on a curve DIR/{t,r,w}.npy (float64 time, speed
//! ratio and weight on the model grid, e.g. a long mechanical speed curve), as `model_step`
//! calls it (Ta = 5 s, ident window by duration). Prints time, comps, EV;
//! measure RSS with `/usr/bin/time -l`.
//!
//! cargo run --release -p ts-core --example fit_long -- DIR [--comps14]
//!
//! `--comps14`: skip identification, fit 14 given fixed comps × 3 harmonics (the largest L3
//! system `identify` can produce: max_comps = 14) — worst-case size / memory.

use std::time::Instant;
use ts_core::model::{Comp, HierOpts, IdentOpts, explained, fit_hier};

fn ld(p: &std::path::Path) -> Vec<f64> {
    let f = std::fs::File::open(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    npyz::NpyFile::new(std::io::BufReader::new(f))
        .unwrap()
        .into_vec::<f64>()
        .unwrap()
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("DIR"));
    let (t, r, w) = (
        ld(&dir.join("t.npy")),
        ld(&dir.join("r.npy")),
        ld(&dir.join("w.npy")),
    );
    let comps = std::env::args().any(|a| a == "--comps14").then(|| {
        (0..14)
            .map(|k| Comp::new(&format!("c{k}"), 0.37 + 1.13 * k as f64, 0.0, 0.0))
            .collect::<Vec<_>>()
    });
    let o = HierOpts {
        ta: 5.0,
        comps,
        ident: IdentOpts::for_duration(t[t.len() - 1]),
        ..HierOpts::default()
    };
    let t0 = Instant::now();
    let h = fit_hier(&t, &r, &w, &o).unwrap();
    let el = t0.elapsed().as_secs_f64();
    let vv: Vec<f64> = w
        .iter()
        .zip(&h.rejected)
        .map(|(&a, &j)| if a > 0.0 && !j { 1.0 } else { 0.0 })
        .collect();
    let ex = explained(&[("m1", &h.m1), ("m2", &h.m2), ("m3", &h.m3)], &t, &r, &vv);
    let n3: usize = h.m3.amp.len();
    println!(
        "fit_hier {:.1} min curve: {:.2} s ({} threads), {} comps, {} L3 keys, EV {:.6}, explained +{:.2} s",
        t[t.len() - 1] / 60.0,
        el,
        rayon::current_num_threads(),
        h.comps.len(),
        n3,
        ex["expl_m3"],
        t0.elapsed().as_secs_f64() - el
    );
    // bit-level fingerprint of the result (compare across RAYON_NUM_THREADS)
    let mut fp: u64 = ex["expl_m3"].to_bits() ^ ex["expl_m2"].to_bits();
    for m in [&h.m2, &h.m3] {
        for v in m.drift_coef.iter().flatten() {
            fp = fp.rotate_left(7) ^ v.to_bits();
        }
        for a in m.amp.values() {
            let zs: Vec<_> = match a {
                ts_core::model::Amp::L2(z) => vec![*z],
                ts_core::model::Amp::L3(v) => v.clone(),
            };
            for z in zs {
                fp = fp.rotate_left(7) ^ z.re.to_bits() ^ z.im.to_bits().rotate_left(32);
            }
        }
    }
    println!("EV {:?} fingerprint {fp:016x}", ex["expl_m3"]);
    for c in &h.comps {
        print!("{} ", c.name);
    }
    println!();
}
