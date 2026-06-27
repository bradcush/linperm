//! Phase breakdown of `biperm::index` across PCS schemes.
//!
//! A non-Criterion harness that times the two reconstructable phases
//! of `index`. `aux_gen` (build the two sparse indicators) and
//! `commit` (PCS-commit each), both stable public calls.
//! Writes the raw numbers to `target/phases.csv`.
//!
//! Plain `Instant` over a few iterations, enough to read phase
//! ratios, not a rigorous benchmark for absolute timings.

mod common;

use std::hint::black_box;
use std::time::{Duration, Instant};

use ark_bn254::{Fr, G1Projective};
use ark_std::test_rng;

use biperm::permcore::{MockPcs, Permutation, PolynomialCommitment};
use hyrax::Hyrax;

use common::instance;

const MUS: [usize; 3] = [8, 10, 12];
const ITERS: usize = 10;

// Output path relative to the crate dir.
const CSV_REL_PATH: &str = "/../target/phases.csv";

/// Mean (aux_gen, commit) durations
/// for `index` under PCS scheme `P`.
fn time_index<P: PolynomialCommitment<Fr>>(
    perm: &Permutation,
    rng: &mut impl ark_std::rand::RngCore,
) -> (Duration, Duration) {
    let (pk, _vk) = P::setup(perm.num_vars() * 3 / 2, rng).unwrap();
    // Warm up allocator / caches; result discarded.
    let il = perm.half_indicator::<Fr>(true).unwrap();
    let ir = perm.half_indicator::<Fr>(false).unwrap();
    black_box(P::commit(&pk, (&il).into()).unwrap());
    black_box(P::commit(&pk, (&ir).into()).unwrap());
    let mut aux = Duration::ZERO;
    let mut com = Duration::ZERO;
    for _ in 0..ITERS {
        let t = Instant::now();
        let il = perm.half_indicator::<Fr>(true).unwrap();
        let ir = perm.half_indicator::<Fr>(false).unwrap();
        aux += t.elapsed();

        let t = Instant::now();
        black_box(P::commit(&pk, (&il).into()).unwrap());
        black_box(P::commit(&pk, (&ir).into()).unwrap());
        com += t.elapsed();
    }
    (aux / ITERS as u32, com / ITERS as u32)
}

/// scheme, $\mu$, and the two phase durations.
type Row = (&'static str, usize, Duration, Duration);

/// Write the rows as CSV (raw ms; percentages are
/// derivable) to the workspace `target/` dir.
fn write_csv(rows: &[Row]) {
    let path = format!("{}{CSV_REL_PATH}", env!("CARGO_MANIFEST_DIR"));
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

fn main() {
    let mut rng = test_rng();
    let mut rows: Vec<Row> = Vec::new();
    for mu in MUS {
        // `instance` gives (perm, f, g); index needs only perm.
        // The f/g build is cheap setup, outside the timed region.
        let (perm, _f, _g) = instance(mu, &mut rng);
        let (a, c) = time_index::<MockPcs<Fr>>(&perm, &mut rng);
        rows.push(("mock", mu, a, c));
    }
    for mu in MUS {
        let (perm, _f, _g) = instance(mu, &mut rng);
        let (a, c) = time_index::<Hyrax<G1Projective>>(&perm, &mut rng);
        rows.push(("hyrax", mu, a, c));
    }
    write_csv(&rows);
}
