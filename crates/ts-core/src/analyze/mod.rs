//! analyze: pass 1, the speed curve from a tracked reference line.
//! Stages 1-5: `stage1`, `coarse`, `pll`, `track`. Stage 6 and the public entry: `gaps`,
//! `select`, `api`.
pub mod coarse;
pub mod pll;
pub mod stage1;
pub mod track;

// Stage 6 (grid, gap fill, bw selection) and the public entry.
pub mod api;
pub mod gaps;
pub mod select;

pub use api::{
    AnalyzeDebug, AnalyzeFull, AnalyzeOpts, Bw, Stage6In, analyze, analyze_full, analyze_tracks,
    grid_t,
};
pub use gaps::{DT, Grid, SOFT_CONF, TrackGridIn, gap_fill};
pub use select::{SelIn, Selection, select};
pub use track::BW_AUTO;
