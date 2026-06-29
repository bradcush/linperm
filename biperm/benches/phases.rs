//! Phase breakdown of `biperm::index` and `biperm::prove`.
//! Non-Criterion harness writing per-phase numbers to `target/`.
//!
//! `index` has no mid-call Fiat-Shamir challenges, so its two phases
//! (`aux_gen`, build the sparse indicators; `commit`, PCS-commit them) are
//! reconstructed externally by re-calling public steps → `index_phases.csv`.
//!
//! `prove` squeezes $\alpha$ and the sumcheck `r` mid-call, so its phases
//! can't be reconstructed without replaying the whole call. It's instrumented
//! in-place with `tracing` spans instead; a capturing layer sums each span's
//! elapsed time by name (PCS opens share `opens` name) → `prove_phases.csv`.
//!
//! Plain `Instant` over a few iterations, enough to read phase
//! ratios, not a rigorous benchmark for absolute timings.
//!
//! Each breakdown carries a `shyrax` (sparse) and a `dhyrax` (dense) row.
//! The `index` row densifies its reconstructed indicators directly; `prove`
//! selects the representation via `biperm::index_with`, wrapper-free.

mod common;

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ark_bn254::{Fr, G1Projective};
use ark_poly::{DenseMultilinearExtension, SparseMultilinearExtension};
use ark_std::rand::RngCore;
use ark_std::test_rng;
use tracing::span;
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

use biperm::permcore::{
    MleRef, MockPcs, Permutation, PolynomialCommitment, Transcript,
};
use biperm::{index_with, prove, IndicatorRepr};
use hyrax::Hyrax;

use common::instance;

const MUS: [usize; 4] = [8, 10, 12, 14];
const ITERS: usize = 10;

// Output paths relative to the crate dir.
const INDEX_CSV_REL_PATH: &str = "/../target/index_phases.csv";
const PROVE_CSV_REL_PATH: &str = "/../target/prove_phases.csv";

// `prove` span names, in report column order. `commit`, `aux`, and
// `sumcheck` time one region each; `opens` aggregates the three PCS opens.
const PROVE_PHASES: [&str; 4] = ["commit", "aux", "sumcheck", "opens"];

/// Mean (aux_gen, commit) durations for `index` under PCS scheme `P`.
/// With `dense`, indicators are densified before commit (the slow `O(N)`
/// path) rather than committed sparse — this bench is the direct Hyrax
/// caller, so it picks the representation, no wrapper needed.
fn time_index<P: PolynomialCommitment<Fr>>(
    perm: &Permutation,
    rng: &mut impl RngCore,
    dense: bool,
) -> (Duration, Duration) {
    let (pk, _vk) = P::setup(perm.num_vars() * 3 / 2, rng).unwrap();
    // Commit both indicators, densifying first on the `dense` path.
    let commit = |il: &SparseMultilinearExtension<Fr>,
                  ir: &SparseMultilinearExtension<Fr>| {
        if dense {
            let dl = MleRef::from(il).to_dense();
            let dr = MleRef::from(ir).to_dense();
            black_box(P::commit(&pk, (&dl).into()).unwrap());
            black_box(P::commit(&pk, (&dr).into()).unwrap());
        } else {
            black_box(P::commit(&pk, il.into()).unwrap());
            black_box(P::commit(&pk, ir.into()).unwrap());
        }
    };
    // Warm up allocator / caches; result discarded.
    let il = perm.half_indicator::<Fr>(true).unwrap();
    let ir = perm.half_indicator::<Fr>(false).unwrap();
    commit(&il, &ir);
    let mut aux = Duration::ZERO;
    let mut com = Duration::ZERO;
    for _ in 0..ITERS {
        let t = Instant::now();
        let il = perm.half_indicator::<Fr>(true).unwrap();
        let ir = perm.half_indicator::<Fr>(false).unwrap();
        aux += t.elapsed();

        let t = Instant::now();
        commit(&il, &ir);
        com += t.elapsed();
    }
    (aux / ITERS as u32, com / ITERS as u32)
}

/// Per-span elapsed time, summed by span name. Spans sharing a
/// name (the three `opens` spans) accumulate into one entry.
#[derive(Default)]
struct Timings(Mutex<HashMap<&'static str, Duration>>);

impl Timings {
    fn add(&self, name: &'static str, d: Duration) {
        *self.0.lock().unwrap().entry(name).or_default() += d;
    }
    fn clear(&self) {
        self.0.lock().unwrap().clear();
    }
    fn snapshot(&self) -> HashMap<&'static str, Duration> {
        self.0.lock().unwrap().clone()
    }
}

/// Span enter instant, stashed in
/// span's extensions until exits.
struct Start(Instant);

/// `tracing` layer timing each span
/// from enter to exit, sum by name.
struct PhaseTimer(Arc<Timings>);

impl<S> Layer<S> for PhaseTimer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(Start(Instant::now()));
        }
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if let Some(Start(t)) = span.extensions_mut().remove::<Start>() {
                self.0.add(span.name(), t.elapsed());
            }
        }
    }
}

/// Mean per-phase durations (in `PROVE_PHASES` order) for `prove` under PCS
/// scheme `P` with indicator representation `repr`, read from the span timer.
fn time_prove<P: PolynomialCommitment<Fr>>(
    perm: &Permutation,
    f: &DenseMultilinearExtension<Fr>,
    g: &DenseMultilinearExtension<Fr>,
    rng: &mut impl RngCore,
    timings: &Timings,
    repr: IndicatorRepr,
) -> [Duration; 4] {
    let pk = P::setup(perm.num_vars() * 3 / 2, rng).unwrap().0;
    let (p_idx, _vk) = index_with::<Fr, P>(&pk, perm, repr).unwrap();
    // Warm up, then drop its timings.
    {
        let mut t = Transcript::new(b"bench");
        black_box(prove(&pk, &p_idx, f, g, &mut t).unwrap());
    }
    timings.clear();
    for _ in 0..ITERS {
        let mut t = Transcript::new(b"bench");
        black_box(prove(&pk, &p_idx, f, g, &mut t).unwrap());
    }
    let map = timings.snapshot();
    PROVE_PHASES.map(|name| {
        map.get(name).copied().unwrap_or(Duration::ZERO) / ITERS as u32
    })
}

/// scheme, $\mu$, and the two `index` phase durations.
type RowIndex = (&'static str, usize, Duration, Duration);

/// scheme, $\mu$, and the `prove` phase durations.
type ProveRow = (&'static str, usize, [Duration; 4]);

/// Write the rows as CSV (raw ms; percentages are
/// derivable) to the workspace `target/` dir.
fn write_index_csv(rows: &[RowIndex]) {
    let path = format!("{}{INDEX_CSV_REL_PATH}", env!("CARGO_MANIFEST_DIR"));
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let mut out = String::from("scheme,mu,aux_gen_ms,commit_ms,total_ms\n");
    for &(scheme, mu, aux, com) in rows {
        let total = aux + com;
        out.push_str(&format!(
            "{scheme},{mu},{:.3},{:.3},{:.3}\n",
            ms(aux),
            ms(com),
            ms(total),
        ));
    }
    match std::fs::write(&path, out) {
        Ok(()) => println!("wrote {path}"),
        Err(e) => eprintln!("failed to write {path}: {e}"),
    }
}

/// Write the `prove` phase rows as CSV (raw ms;
/// percentages derivable) to the workspace `target/` dir.
fn write_prove_csv(rows: &[ProveRow]) {
    let path = format!("{}{PROVE_CSV_REL_PATH}", env!("CARGO_MANIFEST_DIR"));
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let mut out = String::from(
        "scheme,mu,commit_ms,aux_ms,sumcheck_ms,opens_ms,total_ms\n",
    );
    for &(scheme, mu, ph) in rows {
        let total: Duration = ph.iter().sum();
        out.push_str(&format!(
            "{scheme},{mu},{:.3},{:.3},{:.3},{:.3},{:.3}\n",
            ms(ph[0]),
            ms(ph[1]),
            ms(ph[2]),
            ms(ph[3]),
            ms(total),
        ));
    }
    match std::fs::write(&path, out) {
        Ok(()) => println!("wrote {path}"),
        Err(e) => eprintln!("failed to write {path}: {e}"),
    }
}

fn main() {
    let mut rng = test_rng();

    // index breakdown, external reconstruction
    let mut rows: Vec<RowIndex> = Vec::new();
    for mu in MUS {
        // `instance` gives (perm, f, g); index needs only perm.
        // The f/g build is cheap setup, outside the timed region.
        let (perm, _f, _g) = instance(mu, &mut rng);
        let (a, c) = time_index::<MockPcs<Fr>>(&perm, &mut rng, false);
        rows.push(("mock", mu, a, c));
    }
    for mu in MUS {
        let (perm, _f, _g) = instance(mu, &mut rng);
        let (a, c) = time_index::<Hyrax<G1Projective>>(&perm, &mut rng, true);
        rows.push(("dhyrax", mu, a, c));
    }
    for mu in MUS {
        let (perm, _f, _g) = instance(mu, &mut rng);
        let (a, c) = time_index::<Hyrax<G1Projective>>(&perm, &mut rng, false);
        rows.push(("shyrax", mu, a, c));
    }
    write_index_csv(&rows);

    // prove breakdown, in-call spans
    let timings = Arc::new(Timings::default());
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(PhaseTimer(timings.clone())),
    )
    .expect("set tracing subscriber");
    let mut prove_rows: Vec<ProveRow> = Vec::new();
    let sparse = IndicatorRepr::Sparse;
    let dense = IndicatorRepr::Dense;
    for mu in MUS {
        let (perm, f, g) = instance(mu, &mut rng);
        let ph = time_prove::<MockPcs<Fr>>(
            &perm, &f, &g, &mut rng, &timings, sparse,
        );
        prove_rows.push(("mock", mu, ph));
    }
    for mu in MUS {
        let (perm, f, g) = instance(mu, &mut rng);
        let ph = time_prove::<Hyrax<G1Projective>>(
            &perm, &f, &g, &mut rng, &timings, dense,
        );
        prove_rows.push(("dhyrax", mu, ph));
    }
    for mu in MUS {
        let (perm, f, g) = instance(mu, &mut rng);
        let ph = time_prove::<Hyrax<G1Projective>>(
            &perm, &f, &g, &mut rng, &timings, sparse,
        );
        prove_rows.push(("shyrax", mu, ph));
    }
    write_prove_csv(&prove_rows);
}
