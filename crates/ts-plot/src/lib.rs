//! PNG plots (plotters bitmap backend, embedded DejaVu Sans; no system fonts).
//!
//! Computation is separate from drawing: [`speed_plot_numbers`] is a pure function that
//! computes the numbers the speed plot reports (wow / flutter / L1).
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord
)]
mod draw;
mod model;
mod numbers;
mod speed;
mod survey;

pub use model::{CompTrack, ModelFit, plot_model};
pub use numbers::{
    SpeedNumbers, band_rms, detail_window, meta_f64, ref_gaps, spans, speed_plot_numbers,
};
pub use speed::plot_speed;
pub use survey::{SurveyCandidate, SurveyClass, SurveyDet, SurveyPlotInput, plot_survey};

#[derive(Debug, thiserror::Error)]
pub enum PlotError {
    #[error("plot input: {0}")]
    Input(String),
    #[error("plot drawing: {0}")]
    Draw(String),
    #[error("plot numbers: {0}")]
    Value(#[from] ts_dsp::ValueError),
}
