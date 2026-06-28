# Linear\*-Time Permutation Check

## Background

A recent paper by Benedikt Bünz, Jessica Chen, and Zachary DeStefano proposing
two new permutation arguments for use in modern SNARK protocols. This
repository aims to implement `BiPerm` and `MulPerm` efficiently in Rust. It's a
work in progress which has not gone through a formal audit and is not
recommended for use in production systems. Use at your own risk.

## Paper

- [Cryptology ePrint Archive](https://eprint.iacr.org/2025/1850)
- [Linear\*-Time Permuation Check](2025-ltpc.pdf)

### Relevant sections

- §4.3 Sumcheck protocol: basis for both PIOPs.
- §4.4 Multilinear PCS: drives our PolynomialCommitment trait shape.
- §4.6 Lemma 4: the permutation-to-sumcheck reduction.
- §6 BiPerm + Algorithm 2: first PIOP to implement.
- §7 MulPerm + Algorithms 3-7: second PIOP to implement.
- §7.3 Bucketing: the trick that gets MulPerm to $n \cdot O(\sqrt(log n))$.

## Example

This example comes directly from integration tests in BiPerm and shows what the
full protocol looks like using the Hyrax PCS. For an annotated implementation,
see that source. Integration tests also contain the full protocol using a
`MockPcs`, which are just full polynomials.

``` rs
// biperm/tests/integration.rs
use ark_bn254::{Fr, G1Projective};
use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;
use ark_std::test_rng;

use biperm::permcore::{
    MockPcs, Permutation, PolynomialCommitment, Transcript,
};
use biperm::{index, prove, verify};
use hyrax::Hyrax;

fn instance(
    rng: &mut impl ark_std::rand::RngCore,
) -> (
    Permutation,
    DenseMultilinearExtension<Fr>,
    DenseMultilinearExtension<Fr>,
) {
    let perm = Permutation::new(vec![
        5, 3, 7, 1, 0, 6, 2, 4, 9, 11, 8, 10, 13, 15, 12, 14,
    ])
    .unwrap();
    let num_vars = perm.num_vars();
    let f_evals: Vec<Fr> =
        (0..(1 << num_vars)).map(|_| Fr::rand(rng)).collect();
    let mut g_evals = vec![Fr::from(0u64); perm.size()];
    for x in 0..perm.size() {
        g_evals[perm.apply(x)] = f_evals[x];
    }
    let f = DenseMultilinearExtension::from_evaluations_vec(num_vars, f_evals);
    let g = DenseMultilinearExtension::from_evaluations_vec(num_vars, g_evals);
    (perm, f, g)
}

#[test]
fn biperm_round_trip() {
    let mut rng = test_rng();
    let (perm, f, g) = instance(&mut rng);
    let (pk, vk) =
        Hyrax::<G1Projective>::setup(perm.num_vars() * 3 / 2, &mut rng)
            .unwrap();
    let (p_idx, v_idx) = index::<Fr, Hyrax<G1Projective>>(&pk, &perm).unwrap();
    let mut prover_t = Transcript::new(b"integration");
    let proof = prove(&pk, &p_idx, &f, &g, &mut prover_t).unwrap();
    let mut verifier_t = Transcript::new(b"integration");
    verify(&vk, &v_idx, &proof, &mut verifier_t).unwrap();
}
```

## Workspace

- Commands should be run at the workspace level
- The same commands can be scoped to individual crates

## Building

``` sh
# Optionally w/ --release
cargo build
```

## Testing

Unit, integration, doc tests, and benches:

``` sh
# All but benches
cargo test

# Just benches
cargo test --benches
```

## Linting

lib, bins, tests, examples, and benches:

``` sh
# Treat all warnings as errors
cargo clippy --all-targets -- -D warnings
```

## Benchmarks

Runs full measurement loops:

``` sh
cargo bench
```

### Baselines

Criterion can snapshot a run and diff:

``` sh
# Snapshot the index bench, currently dense
cargo bench --bench index -- --save-baseline dense
```

After a change, compare against it:

``` sh
# Compare current index against baseline
cargo bench --bench index -- --baseline dense
```

### Phase breakdown

A non-Criterion report (`harness = false`) for each phase's share of the call
per PCS scheme and $\mu$, raw ms to `target/`; scripts derive the percentages.

For `index`, the two phases are `aux_gen` (auxiliary generation, building the
sparse indicator polynomials from $\sigma$) and `commit` (PCS-committing them),
"assemble" is ignored. They have no mid-call challenges, so the bench
reconstructs them by re-calling the public steps.

`prove` squeezes $\alpha$ and the sumcheck $r$ mid-call, so its phases can't be
reconstructed externally; it's instrumented in-place with `tracing` spans (no-op
without a subscriber) that the bench's capturing layer times. The phases are
`commit`, `aux`, `sumcheck`, and `opens` (the three PCS opens summed via a
shared span name).

``` sh
cargo bench --bench phases

# Render as tables
scripts/index-phases-table.sh
scripts/prove-phases-table.sh
```

### Flamegraphs

*Width is time, vertical is call depth.*

The benches double as profiling targets via `pprof`. Passing `--profile-time`
skips the timing analysis and samples the timed closure instead, writing an SVG
per benchmark id or filtering to a single id.

``` sh
# Profile all, each 8s
cargo bench -- --profile-time=8

# Profile one id for 8s; single path example
# target/criterion/<group>/<id>/profile/flamegraph.svg
cargo bench --bench index -- --profile-time=8 'biperm_index/hyrax/12'
```

## Documentation

From `///` doc comments. LaTeX in doc comments is rendered by KaTeX, wired in
via `katex-header.html` and `.cargo/config.toml`. Use `$…$` for inline math and
`$$…$$` for display math.

``` sh
# Build and open the API docs
cargo doc --no-deps --open
```

Clean what's generated:

``` sh
cargo clean --doc
```

## Formatting

``` sh
cargo fmt

# Verify only
cargo fmt --check
```

## Organization

- `permcore`: Shared building blocks, library crate
  - permutation type, equality polynomial, Fiat-Shamir transcript, PCS trait
- `biperm`: BiPerm implementation, library crate
- `mulperm`: MulPerm implementation, library crate
- `hyrax`: Hyrax PCS backend, library crate
  - binding-only, dense; for now

## Plan

- Use AI to assist in building a first version
- Walk though decisions, understand why's, what's best
- Separate dependency libs (internal/external) from paper code
- Specify learning in SKILL.md format to run through again

## Development notes

- Permutation check $f(\sigma(x)) = g(x)$ reduces to a sumcheck via Lemma 4:
  - $\Sigma_{x \in B_\mu} f(x) \cdot 1_\sigma(x, \alpha) = g(\alpha)$
- How $1_\sigma (X, Y)$ is arithmetized (eg. BiPerm, MulPerm)
  - BiPerm: indicator polys are $n^{1.5}$, needs a sparse-friendly PCS
    (eg. Hyrax, Dory, KZH) to keep commitment cost linear-time
  - MulPerm: bucketing keeps prover at $n \cdot O(\sqrt(log n))$, any ML PCS
- Both protocols PIOPs (Polynomial Interactive Oracle Proofs)
  - Multi-linear polynomial oracles, instantiated w/ PCS and Fiat-Shamir
- Soundness scales like $polylog(n)/|F|$
  - Authors target $n = 2^{32}$, $|F| = 2^{128}$.
- **No public Rust implementations exist of this.**

### Dependencies

- `arkworks`: multi-linear extensions, fields, curves, serialization
  - [ ] Not sure what serialization is referring to here?
- [ ] HyperPlonk, Jolt, Plonky3, Spartan2, Binius relationships?

## Implementation

- PCS started w/ a `MockPcs` for Trait (introduced Hyrax)
- BN254 field/curve, **pairing friendly**, every PCS in paper needs that
  - [ ] Read more about this curve, details of it, why we choose it
- [ ] Multiple small crates (mirrors Hyperplonk layout)
- [ ] We use (merlin/spongefish), do we want others?

> Ethereum precompiles (EIP-196/197).<br>
> BLS12-381 is the obvious upgrade if we ever need $\geq$ 128-bit security.<br>
> Goldilocks/BabyBear are non-starters without \> extension fields.

``` txt
linperm/
├── permcore/       # Shared building blocks
├── biperm/         # BiPerm prove/verify (indexed)
├── mulperm/        # Currently re-exports permcore
├── hyrax/          # Hyrax PCS backend (dense)
└── scripts/        # Developer tooling
```

## Deferred work

- [x] Sumcheck per MulPerm
- [ ] Prover/verifier for MulPerm
- [ ] PCS backend support (eg. Hyrax, Multi-linear KZG)
  - [x] Hyrax (Dense, trusted-setup, binding-only)
  - [ ] Hyrax (Sparse, transparent, hiding)
- [ ] Benchmarks by itself, w/ HyperPlonk

### Optional

- Prover-provided permutation
- Lookup generalization
- Benchmarks
- Paralellism
- FFT-based speedups

## Skills

`/initialize-linperm`: Bootstrap project
