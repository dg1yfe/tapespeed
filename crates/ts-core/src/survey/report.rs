//! Survey output: `report_text`, cleaned JSON (fixed key order, floats rounded to
//! 6 decimals, NaN/inf → null), plot data.

use ts_io::json::Value;
use ts_io::{fmt_fixed, fmt_general, repr_shortest, round_dec};

use super::cands::Cand;
use super::detect::{LOCK_DB, MARGIN_DB, NOMINAL, SCALE_DB, STEP};
use super::plan::{BwVal, Plan, RecItem, RecVal, Recommend};
use super::{Coh, PairCheck, Survey, TrialPair};

/// Fixed-point with explicit sign: `+1.23` / `-0.50` (`p` decimals).
pub fn sign_f(x: f64, p: usize) -> String {
    let s = fmt_fixed(x, p);
    if s.starts_with('-') {
        s
    } else {
        format!("+{s}")
    }
}

/// Text form of an optional float: shortest round-trip repr, or `None`.
pub fn fmt_opt(v: Option<f64>) -> String {
    v.map_or("None".into(), repr_shortest)
}

/// `fmt(v, spec)` of report_text: "-" for None / non-finite.
fn fmt_f(v: Option<f64>, p: usize) -> String {
    match v {
        Some(x) if x.is_finite() => fmt_fixed(x, p),
        _ => "-".into(),
    }
}

fn rj(s: &str, w: usize) -> String {
    format!("{s:>w$}")
}

fn lj(s: &str, w: usize) -> String {
    format!("{s:<w$}")
}

fn short(cls: &str) -> &'static str {
    match cls {
        "FM pilot" => "FM",
        "AM beat" => "AM",
        "TV line" => "TV",
        "hum" => "hum",
        "codec" => "codec",
        _ => "?",
    }
}

/// `Path(p).name`.
fn file_name(p: &str) -> &str {
    p.rsplit('/')
        .find(|s| !s.is_empty() && *s != ".")
        .unwrap_or("")
}

/// `report_text(S, n_show)` (S with recommend and plan).
pub fn report_text(s: &Survey, rec: &Recommend, pl: &Plan, n_show: usize) -> String {
    let (inf, bw) = (&s.info, &s.bandwidth);
    let blocks = s.blocks_rounded();
    let nb = blocks.len();
    let mut dec = 0.0;
    for (_, b) in &blocks {
        dec += b;
    }
    let mut l: Vec<String> = vec![
        format!("SURVEY {}", file_name(&s.file)),
        format!(
            "  {} {} kb/s {} | {} Hz | {} ch | {} s | {nb} blocks x {} s = {} s ({} %) | {} s",
            inf.codec,
            fmt_fixed(inf.bit_rate as f64 / 1000.0, 0),
            if inf.lossy { "lossy" } else { "lossless" },
            inf.sr,
            inf.channels,
            fmt_fixed(inf.duration, 1),
            fmt_fixed(blocks[0].1, 1),
            fmt_fixed(dec, 0),
            fmt_fixed(100.0 * dec / inf.duration, 1),
            fmt_fixed(s.runtime_s, 1)
        ),
        match bw.cliff_hz {
            Some(c) => format!(
                "  bandwidth: Nyquist {} Hz, cliff {} kHz",
                fmt_general(bw.nyquist, 6),
                fmt_fixed(c / 1000.0, 1)
            ),
            None => format!(
                "  bandwidth: Nyquist {} Hz, no cliff (no codec lowpass)",
                fmt_general(bw.nyquist, 6)
            ),
        },
        format!(
            "  level rel. 1-2 kHz [dB]: {}",
            bw.levels_db
                .iter()
                .map(|(k, v)| format!("{k} {}", fmt_fixed(*v, 0)))
                .collect::<Vec<_>>()
                .join(" | ")
        ),
        format!(
            "  C/N0 = excess power in f(1+-w) / median N0 1.2-3 % off (Welch 0.75 s, median of present blocks; survey scale = analyze - {} dB): no lock < {}, marginal < {} dB-Hz",
            fmt_general(SCALE_DB, 6),
            fmt_general(LOCK_DB, 6),
            fmt_general(MARGIN_DB, 6)
        ),
        "  wow = frame (0.09 s) peak std, noise-biased; wcorr = frame-wow correlation with #P (tape >= 0.5, none < 0.2); trk = analyze valid % on 60 s; s25 = noise rms at 25 Hz passband".into(),
        String::new(),
    ];
    let cs = &s.candidates;
    let show: Vec<&Cand> = cs
        .iter()
        .filter(|c| c.cls != "codec")
        .take(n_show)
        .collect();
    l.push("CANDIDATES".into());
    l.push(format!(
        "  {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}  verdict",
        rj("#", 2),
        rj("f Hz", 8),
        lj("class", 5),
        rj("nom", 5),
        rj("off %", 7),
        rj("pres", 5),
        rj("C/N0", 5),
        rj("ch", 4),
        rj("prom", 4),
        rj("p5-p95 Hz", 15),
        rj("%", 5),
        rj("wow %", 5),
        rj("wcorr", 5),
        rj("trk %", 5),
        rj("s25 %", 6)
    ));
    for c in &show {
        let wc = if c.primary {
            "P".to_string()
        } else {
            fmt_f(c.wow_corr.flatten(), 2)
        };
        let hl = c.cls == "hum" || c.cls == "unknown";
        l.push(format!(
            "  {:2} {} {} {} {} {:2}/{:<2} {} {} {} {}-{} {} {} {} {} {}  {}{}{}",
            c.id,
            rj(&fmt_fixed(c.f, 1), 8),
            lj(short(&c.cls), 5),
            rj(&fmt_f(c.nominal, 0), 5),
            rj(
                &match c.offset_pct {
                    Some(o) if o.is_finite() => sign_f(o, 3),
                    _ => "-".into(),
                },
                7
            ),
            c.n,
            c.nb,
            rj(&fmt_fixed(c.cn0, 1), 5),
            rj(&c.ch, 4),
            rj(&fmt_fixed(c.prom, 0), 4),
            rj(&fmt_fixed(c.spread_hz[0], 1), 7),
            lj(&fmt_fixed(c.spread_hz[1], 1), 7),
            rj(&fmt_fixed(c.spread_pct, 2), 5),
            rj(&fmt_f(Some(c.wow_pct), 3), 5),
            rj(&wc, 5),
            rj(&fmt_f(c.trial_valid.map(|v| 100.0 * v), 0), 5),
            rj(&fmt_fixed(c.sigma25_pct, 3), 6),
            if hl { c.label.as_str() } else { "" },
            if hl { " | " } else { "" },
            c.verdict
        ));
    }
    let cod: Vec<&Cand> = cs.iter().filter(|c| c.cls == "codec").collect();
    if !cod.is_empty() {
        let fs: Vec<f64> = cod.iter().map(|c| c.f).collect();
        let mn = fs.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = fs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let cm = cod.iter().map(|c| c.cn0).fold(f64::NEG_INFINITY, f64::max);
        l.push(format!(
            "  codec artifacts: {} lines {}-{} Hz (lossy; >= 15 kHz / ~sr/4 / no shared wow), max C/N0 {} dB-Hz: not usable",
            cod.len(),
            fmt_fixed(mn, 0),
            fmt_fixed(mx, 0),
            fmt_fixed(cm, 0)
        ));
    }
    if !s.coherence.is_empty() {
        l.push(String::new());
        l.push("COHERENCE (trial track, analyze bw auto, 60 s; band coherence of the two pass-1 curves)".into());
        l.push(format!(
            "  {} {} {} {} {} {} {}  same speed modulation",
            rj("a Hz", 8),
            rj("b Hz", 8),
            rj("t0 s", 7),
            rj("common %", 8),
            rj("0-0.5", 6),
            rj("0.5-6", 6),
            rj("6-25", 6)
        ));
        for k in &s.coherence {
            let g = |b: &str| fmt_f(k.get(b), 3);
            l.push(format!(
                "  {} {} {} {} {} {} {}  {}",
                rj(&fmt_fixed(k.a, 1), 8),
                rj(&fmt_fixed(k.b, 1), 8),
                rj(&fmt_fixed(k.t0, 0), 7),
                rj(&fmt_fixed(100.0 * k.common, 0), 8),
                rj(&g("0-0.5"), 6),
                rj(&g("0.5-6"), 6),
                rj(&g("6-25"), 6),
                if k.tape {
                    "yes (tape)"
                } else if k.coh.is_empty() {
                    "no / not measurable"
                } else {
                    "no"
                }
            ));
        }
    }
    if let Some(pr) = &s.pair {
        l.push(String::new());
        l.push(format!(
            "PAIR (--abs pair:{} trial: primary tracked on 60 s, partner on the normalised spectrum, 8 s Welch; true Hz = apparent / s of that trial)",
            fmt_general(pr.hz, 6)
        ));
        l.push(format!(
            "  {} {} {} {} {} {} {} {} {} {}  ambiguity / error",
            rj("t0 s", 6),
            rj("valid", 5),
            rj("s", 9),
            rj("f_lo", 8),
            rj("f_hi", 8),
            rj("d Hz", 6),
            rj("prom", 4),
            rj("cand", 4),
            rj("ok", 3),
            rj("xchk ppm", 8)
        ));
        for t in &pr.trials {
            let x = t
                .sol
                .as_ref()
                .map(|s| {
                    s.xcheck
                        .iter()
                        .map(|(_, v)| sign_f(*v, 0))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "-".into());
            let mid = match &t.sol {
                Some(s) => format!(
                    "{} {} {} {} {} ",
                    rj(&fmt_fixed(s.s, 6), 9),
                    rj(&fmt_fixed(s.f_lo, 2), 8),
                    rj(&fmt_fixed(s.f_hi, 2), 8),
                    rj(&fmt_fixed(s.delta, 2), 6),
                    rj(&fmt_fixed(s.prom, 0), 4)
                ),
                None => format!(
                    "{} {} {} {} {} ",
                    rj("-", 9),
                    rj("-", 8),
                    rj("-", 8),
                    rj("-", 6),
                    rj("-", 4)
                ),
            };
            let tail = match &t.error {
                Some(e) if !e.is_empty() => e.clone(),
                _ => t.ambiguity.clone().unwrap_or_default(),
            };
            l.push(format!(
                "  {} {}% {mid}{:4} {:3} {}  {tail}",
                rj(&fmt_fixed(t.t0, 0), 6),
                rj(&fmt_fixed(100.0 * t.valid, 0), 4),
                t.n_cand,
                t.n_ok,
                rj(&x, 8)
            ));
        }
        l.push(format!(
            "  -> {}",
            if pr.found {
                let f = pr.sol.as_ref().unwrap();
                format!(
                    "found: d {} Hz, beats {} / {} Hz, {}/{} trials, f_true spread {} ppm",
                    fmt_fixed(f.delta, 2),
                    fmt_fixed(f.f_lo, 2),
                    fmt_fixed(f.f_hi, 2),
                    pr.n_agree.unwrap(),
                    pr.n_trials.unwrap(),
                    fmt_fixed(f.f_true_spread_ppm, 0)
                )
            } else {
                format!("not found: {}", pr.why.clone().unwrap_or_default())
            }
        ));
    }
    let top: Vec<&Cand> = cs
        .iter()
        .filter(|c| c.cls != "codec" && !c.low)
        .take(5)
        .collect();
    if !top.is_empty() {
        l.push(String::new());
        l.push("PRESENCE  C/N0 [dB-Hz] per block (best channel), . = not detected".into());
        l.push(format!(
            "  {} {}",
            rj("t s", 7),
            top.iter()
                .map(|c| rj(&format!("#{} {}", c.id, fmt_fixed(c.f, 0)), 11))
                .collect::<Vec<_>>()
                .join(" ")
        ));
        for (i, (t0, d)) in blocks.iter().enumerate() {
            l.push(format!(
                "  {} {}",
                rj(&fmt_fixed(t0 + d / 2.0, 0), 7),
                top.iter()
                    .map(|c| rj(
                        &c.block(i).map_or(".".into(), |e| fmt_f(Some(e.cn0), 0)),
                        11
                    ))
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
        }
    }
    l.push(String::new());
    l.push("LIMITS".into());
    for (_, nom, lab) in NOMINAL {
        let need = 2.0 * nom * (1.0 + s.search) * s.play / 0.98;
        let srf = inf.sr as f64;
        let above = bw.cliff_hz.is_some_and(|c| nom * s.play >= c);
        let ok = srf >= need && !above;
        l.push(format!(
            "  {} {} Hz: needs sr >= {} kHz{}{}{}",
            lj(lab, 38),
            rj(&fmt_fixed(nom, 0), 7),
            fmt_fixed(need / 1000.0, 1),
            if srf >= need {
                ""
            } else {
                " -> NOT at this sr"
            },
            if above {
                format!(
                    ", above cliff {} kHz -> removed by the codec",
                    fmt_fixed(bw.cliff_hz.unwrap() / 1000.0, 1)
                )
            } else {
                String::new()
            },
            if ok { " -> ok" } else { "" }
        ));
    }
    l.push(String::new());
    l.push("RECOMMEND".into());
    let items: [(&str, Option<&RecItem>); 4] = [
        ("ref", Some(&rec.r#ref)),
        ("check-line", None),
        ("channel", rec.channel.as_ref()),
        ("abs", rec.abs.as_ref()),
    ];
    let order = [
        items[0],
        ("search", rec.search.as_ref()),
        items[2],
        items[3],
        ("check-line", rec.check_line.as_ref()),
        ("lowpass", rec.lowpass.as_ref()),
        ("auto-notch", rec.auto_notch.as_ref()),
    ];
    for (k, it) in order {
        if let Some((v, txt)) = it {
            let vs = match v {
                RecVal::None => "-".to_string(),
                RecVal::Float(x) => fmt_general(*x, 6),
                other => other.to_text(),
            };
            l.push(format!("  --{} {} {txt}", lj(k, 11), lj(&vs, 12)));
        }
    }
    if let Some(c) = &rec.cmd {
        l.push(format!("  cmd: {c}"));
    }
    l.push(String::new());
    l.push(match (&pl.resolved, &pl.basis) {
        (Some(r), Some(b)) => format!(
            "PLAN ({b}; --ref {}, {})",
            fmt_general(r.r#ref, 6),
            r.ref_src
        ),
        (Some(r), None) => format!("PLAN (--ref {}, {})", fmt_general(r.r#ref, 6), r.ref_src),
        (None, _) => "PLAN: none".into(),
    });
    for (i, (k, v)) in pl.steps.iter().enumerate() {
        l.push(format!("  {:2} {} {v}", i + 1, lj(k, 10)));
    }
    l.push(String::new());
    l.push(format!(
        "WARNINGS{}",
        if pl.warnings.is_empty() { ": none" } else { "" }
    ));
    for w in &pl.warnings {
        l.push(format!("  ! {w}"));
    }
    l.join("\n")
}

// ------------------------------------------------------------------ JSON

fn fl(x: f64) -> Value {
    if x.is_finite() {
        Value::Float(round_dec(x, 6))
    } else {
        Value::Null
    }
}

fn ofl(x: Option<f64>) -> Value {
    x.map_or(Value::Null, fl)
}

fn obj(kv: Vec<(&str, Value)>) -> Value {
    Value::Obj(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn coh_value(k: &Coh) -> Value {
    obj(vec![
        ("a", fl(k.a)),
        ("b", fl(k.b)),
        ("t0", fl(k.t0)),
        ("dur", fl(k.dur)),
        ("common", fl(k.common)),
        ("n_seg", Value::Int(k.n_seg as i64)),
        (
            "coh",
            Value::Obj(k.coh.iter().map(|(b, v)| (b.clone(), fl(*v))).collect()),
        ),
        ("tape", Value::Bool(k.tape)),
    ])
}

fn cand_value(c: &Cand) -> Value {
    let mut v = vec![
        ("f", fl(c.f)),
        ("cls", Value::Str(c.cls.clone())),
        ("label", Value::Str(c.label.clone())),
        ("nominal", ofl(c.nominal)),
        ("codec_risk", Value::Bool(c.codec_risk)),
        ("ch", Value::Str(c.ch.clone())),
        ("low", Value::Bool(c.low)),
        ("n", Value::Int(c.n as i64)),
        ("nb", Value::Int(c.nb as i64)),
        ("presence", fl(c.presence)),
        ("lock_blocks", Value::Int(c.lock_blocks as i64)),
        ("prom", fl(c.prom)),
        ("cn0", fl(c.cn0)),
        ("pers", fl(c.pers)),
        ("wow_pct", fl(c.wow_pct)),
        (
            "spread_hz",
            Value::Arr(vec![fl(c.spread_hz[0]), fl(c.spread_hz[1])]),
        ),
        ("spread_pct", fl(c.spread_pct)),
        (
            "blocks",
            Value::Obj(
                c.blocks
                    .iter()
                    .map(|(k, e)| {
                        (
                            k.to_string(),
                            obj(vec![
                                ("f", fl(e.f)),
                                ("cn0", fl(e.cn0)),
                                ("prom", fl(e.prom)),
                            ]),
                        )
                    })
                    .collect(),
            ),
        ),
        (
            "channels",
            Value::Obj(
                c.channels
                    .iter()
                    .map(|s| {
                        (
                            s.name.clone(),
                            obj(vec![
                                ("n", Value::Int(s.n as i64)),
                                ("lock", Value::Int(s.lock as i64)),
                                ("cn0", ofl(s.cn0)),
                            ]),
                        )
                    })
                    .collect(),
            ),
        ),
        ("ratio", ofl(c.ratio)),
        ("offset_pct", ofl(c.offset_pct)),
        ("sigma25_pct", fl(c.sigma25_pct)),
    ];
    if let Some(w) = c.wow_corr {
        v.push(("wow_corr", ofl(w)));
    }
    if let Some(t) = c.tape {
        v.push(("tape", t.map_or(Value::Null, Value::Bool)));
    }
    if c.primary {
        v.push(("primary", Value::Bool(true)));
    }
    v.push(("verdict", Value::Str(c.verdict.clone())));
    if let Some(tv) = c.trial_valid {
        v.push(("trial_valid", fl(tv)));
    }
    if let Some(k) = &c.coh {
        v.push(("coh", coh_value(k)));
    }
    v.push(("id", Value::Int(c.id as i64)));
    obj(v)
}

fn trial_value(t: &TrialPair) -> Value {
    let mut v = vec![("t0", fl(t.t0)), ("valid", fl(t.valid))];
    if let Some(a) = &t.ambiguity {
        v.push(("ambiguity", Value::Str(a.clone())));
        v.push(("n_cand", Value::Int(t.n_cand as i64)));
        v.push(("n_ok", Value::Int(t.n_ok as i64)));
    }
    if let Some(e) = &t.error {
        v.push(("error", Value::Str(e.clone())));
    }
    if let Some(s) = &t.sol {
        v.extend([
            ("s", fl(s.s)),
            ("f_true", fl(s.f_true)),
            ("f_lo", fl(s.f_lo)),
            ("f_hi", fl(s.f_hi)),
            ("delta", fl(s.delta)),
            ("side", Value::Str(s.side.clone())),
            ("prom", fl(s.prom)),
            ("pres", fl(s.pres)),
            ("k", fl(s.k)),
            (
                "xcheck",
                Value::Obj(s.xcheck.iter().map(|(k, x)| (k.clone(), fl(*x))).collect()),
            ),
        ]);
    }
    if let Some(cs) = &t.cands {
        v.push((
            "cands",
            Value::Arr(
                cs.iter()
                    .map(|e| {
                        obj(vec![
                            ("f", fl(e.f)),
                            ("prom", fl(e.prom)),
                            ("d", fl(e.d)),
                            ("ok", Value::Bool(e.ok)),
                            ("why", Value::Str(e.why.clone())),
                        ])
                    })
                    .collect(),
            ),
        ));
    }
    obj(v)
}

fn pair_value(p: &PairCheck) -> Value {
    let mut v = vec![
        ("hz", fl(p.hz)),
        (
            "trials",
            Value::Arr(p.trials.iter().map(trial_value).collect()),
        ),
        ("found", Value::Bool(p.found)),
    ];
    if let (Some(a), Some(n)) = (p.n_agree, p.n_trials) {
        v.push(("n_agree", Value::Int(a as i64)));
        v.push(("n_trials", Value::Int(n as i64)));
    }
    if let Some(s) = &p.sol {
        v.extend([
            ("delta", fl(s.delta)),
            ("f_true", fl(s.f_true)),
            ("f_true_spread_ppm", fl(s.f_true_spread_ppm)),
            ("f_lo", fl(s.f_lo)),
            ("f_hi", fl(s.f_hi)),
            ("side", Value::Str(s.side.clone())),
            ("prom", fl(s.prom)),
        ]);
        if let Some(x) = s.xcheck_ppm {
            v.push(("xcheck_ppm", fl(x)));
        }
    }
    if let Some(w) = &p.why {
        v.push(("why", Value::Str(w.clone())));
    }
    obj(v)
}

fn rec_val(v: &RecVal) -> Value {
    match v {
        RecVal::None => Value::Null,
        RecVal::Float(x) => fl(*x),
        RecVal::Int(i) => Value::Int(*i),
        RecVal::Str(s) => Value::Str(s.clone()),
    }
}

fn item(it: &RecItem) -> Value {
    Value::Arr(vec![rec_val(&it.0), Value::Str(it.1.clone())])
}

/// `recommend(S)` as JSON.
pub fn recommend_value(r: &Recommend) -> Value {
    let mut v = vec![("ref", item(&r.r#ref))];
    for (k, it) in [
        ("search", &r.search),
        ("channel", &r.channel),
        ("abs", &r.abs),
        ("check_line", &r.check_line),
        ("lowpass", &r.lowpass),
        ("auto_notch", &r.auto_notch),
    ] {
        if let Some(it) = it {
            v.push((k, item(it)));
        }
    }
    if let Some(c) = &r.cmd {
        v.push(("cmd", Value::Str(c.clone())));
    }
    obj(v)
}

/// `plan(a, S)` as JSON.
pub fn plan_value(p: &Plan) -> Value {
    let res = match &p.resolved {
        None => Value::Obj(vec![]),
        Some(r) => obj(vec![
            ("ref", fl(r.r#ref)),
            ("ref_src", Value::Str(r.ref_src.clone())),
            ("centre", fl(r.centre)),
            ("play", fl(r.play)),
            (
                "search_hz",
                Value::Arr(vec![fl(r.search_hz[0]), fl(r.search_hz[1])]),
            ),
            (
                "bw",
                match r.bw {
                    BwVal::Auto => Value::Str("auto".into()),
                    BwVal::Hz(b) => fl(b),
                },
            ),
            ("channel", Value::Str(r.channel.clone())),
            ("line_hz", ofl(r.line_hz)),
            ("corr_limit", Value::Str(r.corr_limit.clone())),
            ("model", Value::Str(r.model.clone())),
            ("abs_mode", Value::Str(r.abs_mode.clone())),
            ("abs_l1", fl(r.abs_l1)),
            ("true_hz", fl(r.true_hz)),
            ("notch_hz", ofl(r.notch_hz)),
            ("lowpass", ofl(r.lowpass)),
            (
                "extra_notches",
                Value::Arr(r.extra_notches.iter().map(|&h| fl(h)).collect()),
            ),
            ("out_len_s", fl(r.out_len_s)),
            (
                "outputs",
                Value::Obj(
                    r.outputs
                        .iter()
                        .map(|(k, v)| (k.clone(), Value::Str(v.clone())))
                        .collect(),
                ),
            ),
        ]),
    };
    obj(vec![
        (
            "steps",
            Value::Arr(
                p.steps
                    .iter()
                    .map(|(k, v)| Value::Arr(vec![Value::Str(k.clone()), Value::Str(v.clone())]))
                    .collect(),
            ),
        ),
        (
            "warnings",
            Value::Arr(p.warnings.iter().map(|w| Value::Str(w.clone())).collect()),
        ),
        ("resolved", res),
    ])
}

/// Cleaned survey JSON with recommend and plan (what `--survey` writes).
pub fn to_json(s: &Survey, rec: &Recommend, pl: &Plan) -> Value {
    let inf = &s.info;
    let mut v = vec![
        ("file", Value::Str(s.file.clone())),
        (
            "info",
            obj(vec![
                ("codec", Value::Str(inf.codec.clone())),
                ("format", Value::Str(inf.format.clone())),
                ("sr", Value::Int(inf.sr as i64)),
                ("channels", Value::Int(inf.channels as i64)),
                ("duration", fl(inf.duration)),
                ("bit_rate", Value::Int(inf.bit_rate)),
                ("lossy", Value::Bool(inf.lossy)),
            ]),
        ),
        (
            "blocks",
            Value::Arr(
                s.blocks_rounded()
                    .iter()
                    .map(|&(a, b)| Value::Arr(vec![fl(a), fl(b)]))
                    .collect(),
            ),
        ),
        (
            "channels",
            Value::Arr(s.channels.iter().map(|c| Value::Str(c.clone())).collect()),
        ),
        ("search", fl(s.search)),
        ("play", fl(s.play)),
        (
            "coherence",
            Value::Arr(s.coherence.iter().map(coh_value).collect()),
        ),
        (
            "candidates",
            Value::Arr(s.candidates.iter().map(cand_value).collect()),
        ),
        (
            "primary",
            s.primary.map_or(Value::Null, |p| Value::Int(p as i64)),
        ),
    ];
    if let Some(p) = &s.pair {
        v.push(("pair", pair_value(p)));
    }
    let bw = &s.bandwidth;
    v.push((
        "bandwidth",
        obj(vec![
            ("nyquist", fl(bw.nyquist)),
            ("cliff_hz", ofl(bw.cliff_hz)),
            (
                "levels_db",
                Value::Obj(
                    bw.levels_db
                        .iter()
                        .map(|(k, x)| (k.clone(), fl(*x)))
                        .collect(),
                ),
            ),
        ]),
    ));
    v.push(("runtime_s", fl(s.runtime_s)));
    v.push(("recommend", recommend_value(rec)));
    v.push(("plan", plan_value(pl)));
    obj(v)
}

/// Survey plot data, field for field `ts_plot::SurveyPlotInput` (ts-plot depends on ts-core,
/// so the pipeline maps this one-to-one; `cls` as the class label string).
#[derive(Debug, Clone)]
pub struct PlotData {
    pub file: String,
    pub f: Vec<f64>,
    pub d: Vec<f64>,
    pub d_blocks: Vec<Vec<f64>>,
    pub prom: Vec<Vec<f64>>,
    pub blocks: Vec<(f64, f64)>,
    /// (id, f, cls, low, primary, ch, [(block, f, cn0)])
    pub candidates: Vec<PlotCand>,
    pub step: f64,
    pub lock_db: f64,
    pub margin_db: f64,
}

#[derive(Debug, Clone)]
pub struct PlotCand {
    pub id: u32,
    pub f: f64,
    pub cls: String,
    pub low: bool,
    pub primary: bool,
    pub ch: String,
    pub blocks: Vec<(usize, f64, f64)>,
}

/// Plot data of a survey (`S["_spec"]`, ranked candidates, rounded blocks).
pub fn plot_data(s: &Survey) -> PlotData {
    PlotData {
        file: file_name(&s.file).to_string(),
        f: s.spec.f.clone(),
        d: s.spec.d.clone(),
        d_blocks: s.spec.d_blocks.clone(),
        prom: s.spec.prom.clone(),
        blocks: s.blocks_rounded(),
        candidates: s
            .candidates
            .iter()
            .map(|c| PlotCand {
                id: c.id as u32,
                f: c.f,
                cls: c.cls.clone(),
                low: c.low,
                primary: c.primary,
                ch: c.ch.clone(),
                blocks: c.blocks.iter().map(|(k, e)| (*k, e.f, e.cn0)).collect(),
            })
            .collect(),
        step: STEP,
        lock_db: LOCK_DB,
        margin_db: MARGIN_DB,
    }
}
