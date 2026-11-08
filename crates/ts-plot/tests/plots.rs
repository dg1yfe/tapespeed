#[path = "../../testdata.rs"]
mod testdata;
use std::path::PathBuf;
use ts_core::curve::read_curve;
use ts_plot::*;

fn out_dir() -> PathBuf {
    let d = std::env::var("TS_PLOT_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("ts-plot-test"));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn expected() -> serde_json::Value {
    let p = testdata::dir("expected.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn cases() -> Vec<(String, PathBuf)> {
    let mut v = vec![];
    for dir in [
        testdata::dir(""),
        std::env::var("TS_PLOT_REF")
            .map(PathBuf::from)
            .unwrap_or_default(),
    ] {
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                if e.path().join("c.csv").exists() {
                    v.push((e.file_name().to_string_lossy().into_owned(), e.path()));
                }
            }
        }
    }
    v.sort();
    v.dedup_by(|a, b| a.0 == b.0);
    v
}

#[test]
fn numbers_match_reference() {
    let exp = expected();
    let mut n = 0;
    for (name, dir) in cases() {
        let (c1, c, cm) = (
            read_curve(dir.join("c1.csv")).unwrap(),
            read_curve(dir.join("c.csv")).unwrap(),
            read_curve(dir.join("cm.csv")).unwrap(),
        );
        let nm = speed_plot_numbers(&c1, &c, &cm).unwrap();
        for (k, v) in [
            ("wow_pct", nm.wow_pct),
            ("flutter_pct", nm.flutter_pct),
            ("l1_pct", nm.l1_pct),
        ] {
            let e = exp[&name][k].as_f64().unwrap();
            assert!((v - e).abs() <= 1e-9, "{name} {k}: {v} vs {e}");
        }
        n += 1;
    }
    assert!(n >= 3);
}

#[test]
fn speed_png() {
    for (name, dir) in cases() {
        let (c1, c, cm) = (
            read_curve(dir.join("c1.csv")).unwrap(),
            read_curve(dir.join("c.csv")).unwrap(),
            read_curve(dir.join("cm.csv")).unwrap(),
        );
        let p = out_dir().join(format!("{name}_speed.png"));
        plot_speed(&c1, &c, &cm, &[0.4, 1.3, 3.1, 7.7, 12.4, 23.0], &p, &name).unwrap();
        assert!(std::fs::metadata(&p).unwrap().len() > 20_000, "{name}");
    }
}

#[test]
fn model_png() {
    let (_, dir) = cases().into_iter().next().unwrap();
    let (c1, c) = (
        read_curve(dir.join("c1.csv")).unwrap(),
        read_curve(dir.join("c.csv")).unwrap(),
    );
    let n = c.t.len().min(c1.t.len());
    let mut c_in = c1.clone();
    c_in.t.truncate(n);
    c_in.ratio.truncate(n);
    let mut c_out = c.clone();
    c_out.t.truncate(n);
    c_out.ratio.truncate(n);
    c_out.valid.truncate(n);
    c_out.confidence.truncate(n);
    c_in.t = c_out.t.clone();
    let predict: Vec<f64> = c_out
        .ratio
        .iter()
        .enumerate()
        .map(|(i, r)| r + 0.0004 * (i as f64 * 0.01).sin())
        .collect();
    let freq: Vec<f64> = (0..n)
        .map(|i| 1.3 + 0.2 * (i as f64 * 1e-3).sin())
        .collect();
    let fit = ModelFit {
        predict,
        r0: 1.0,
        comps: vec![
            CompTrack {
                chirp: false,
                harmonics: vec![1.0, 2.0, 3.0],
                freq: freq.clone(),
            },
            CompTrack {
                chirp: true,
                harmonics: vec![1.0],
                freq: freq
                    .iter()
                    .enumerate()
                    .map(|(i, f)| f * 4.0 + i as f64 * 1e-3)
                    .collect(),
            },
        ],
    };
    let p = out_dir().join("model.png");
    plot_model(&c_in, &c_out, Some(&fit), &p).unwrap();
    assert!(std::fs::metadata(&p).unwrap().len() > 20_000);
    let p2 = out_dir().join("model_nofit.png");
    plot_model(&c_in, &c_out, None, &p2).unwrap();
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn survey_png() {
    use std::collections::BTreeMap;
    let step = 5e-4;
    let nf = ((19900.0f64 / 1000.0).ln() / step) as usize;
    let f: Vec<f64> = (0..nf).map(|i| 1000.0 * (i as f64 * step).exp()).collect();
    let mut rng = Lcg(7);
    let nb = 12;
    let shape = |fv: f64| {
        if fv < 13000.0 {
            1.0 / (1.0 + fv / 1500.0)
        } else {
            1e-4
        }
    };
    let d: Vec<f64> = f.iter().map(|&x| shape(x)).collect();
    let d_blocks: Vec<Vec<f64>> = (0..nb)
        .map(|_| {
            f.iter()
                .map(|&x| shape(x) * (1.0 + 0.3 * rng.next()))
                .collect()
        })
        .collect();
    let prom: Vec<Vec<f64>> = (0..nb)
        .map(|b| {
            f.iter()
                .map(|&x| {
                    let mut p = 2.0 * rng.next();
                    if (x - 9000.0).abs() < 8.0 && (4..8).contains(&b) {
                        p += 20.0;
                    }
                    if x > 14000.0 && rng.next() > 0.93 {
                        p += 15.0;
                    }
                    p
                })
                .collect()
        })
        .collect();
    let det = |f0: f64, bs: &[usize], cn: f64| -> BTreeMap<usize, SurveyDet> {
        bs.iter()
            .map(|&b| {
                (
                    b,
                    SurveyDet {
                        f: f0 + 0.1 * b as f64,
                        cn0: cn + b as f64,
                    },
                )
            })
            .collect()
    };
    let mut cands = vec![
        SurveyCandidate {
            id: 1,
            f: 9009.0,
            cls: SurveyClass::AmBeat,
            low: false,
            primary: false,
            ch: "mono".into(),
            blocks: det(9009.0, &[6, 7], 20.0),
        },
        SurveyCandidate {
            id: 2,
            f: 4715.0,
            cls: SurveyClass::Unknown,
            low: false,
            primary: true,
            ch: "mono".into(),
            blocks: det(4715.0, &[4], 25.0),
        },
        SurveyCandidate {
            id: 3,
            f: 50.0,
            cls: SurveyClass::Hum,
            low: true,
            primary: false,
            ch: "mono".into(),
            blocks: BTreeMap::new(),
        },
    ];
    for i in 0..18 {
        cands.push(SurveyCandidate {
            id: 10 + i,
            f: 14000.0 + 300.0 * i as f64,
            cls: SurveyClass::Codec,
            low: false,
            primary: false,
            ch: "mono".into(),
            blocks: BTreeMap::new(),
        });
    }
    let inp = SurveyPlotInput {
        file: "synthetic_survey.mp3".into(),
        f,
        d,
        d_blocks,
        prom,
        blocks: (0..nb).map(|i| (i as f64 * 600.0, 20.0)).collect(),
        candidates: cands,
        step,
        lock_db: 17.0,
        margin_db: 22.0,
    };
    let p = out_dir().join("survey.png");
    plot_survey(&inp, &p).unwrap();
    assert!(std::fs::metadata(&p).unwrap().len() > 20_000);
}
