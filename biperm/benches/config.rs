//! Shared Criterion config for the timed BiPerm benches.
//!
//! Lives apart from `common` so the non-Criterion `phases` bench can share the
//! instance builder without pulling in (and leaving unused) this profiler glue.

use criterion::Criterion;
use pprof::criterion::{Output, PProfProfiler};

/// Criterion configured to emit a flamegraph of the timed closure under
/// `--profile-time`. Sampling at 100 Hz; the profiler is dormant on normal
/// `cargo bench` runs, so timing/report output is unaffected.
pub fn profiled() -> Criterion {
    Criterion::default()
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)))
}
