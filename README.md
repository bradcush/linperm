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

``` rs
// biperm/tests/integration.rs
use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;
use ark_std::test_rng;

use biperm::permcore::{
    MockPcs, Permutation, PolynomialCommitment, Transcript,
};
use biperm::{index, prove, verify};

#[test]
fn biperm_round_trip() {
    let mut rng = test_rng();
    let perm = Permutation::new(vec![
        5, 3, 7, 1, 0, 6, 2, 4, 9, 11, 8, 10, 13, 15, 12, 14,
    ])
    .unwrap();
    let num_vars = perm.num_vars();
    let f_evals: Vec<Fr> =
        (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let mut g_evals = vec![Fr::from(0u64); perm.size()];
    for x in 0..perm.size() {
        g_evals[perm.apply(x)] = f_evals[x];
    }
    let f = DenseMultilinearExtension::from_evaluations_vec(num_vars, f_evals);
    let g = DenseMultilinearExtension::from_evaluations_vec(num_vars, g_evals);
    let (pk, vk) = MockPcs::<Fr>::setup(num_vars * 3 / 2, &mut rng).unwrap();
    let (p_idx, v_idx) = index::<Fr, MockPcs<Fr>>(&pk, &perm).unwrap();
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
cargo build
```

## Running

``` sh
cargo run
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

Criterion-based, runs full measurement loops:

``` sh
cargo bench
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

## Plan

- Use AI to assist in building first version
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

- Both BiPerm/MulPerm need custom variant of sumcheck
  - That's the point of this work, creating these variants
- PCS starting w/ a `MockPCS` for Trait (swap later)
- BN254 field/curve, **pairing friendly**, every PCS in paper needs that
  - [ ] Read more about this curve, details of it, why we choose it
- [ ] Multiple small crates (mirrors Hyperplonk layout)
- [ ] Not sure what transcript (merlin/spongefish) refers to

> Ethereum precompiles (EIP-196/197).<br>
> BLS12-381 is the obvious upgrade if we ever need $\geq$ 128-bit security.<br>
> Goldilocks/BabyBear are non-starters without \> extension fields.

``` txt
linperm/
├── permcore/       # Shared building blocks
├── biperm/         # Currently re-exports permcore
└── mulperm/        # Currently re-exports permcore
```

### Rust specifics

- `no_std` opt-in for some libs, forgot exact meaning
- `eq` stuff builds evaluation table over the boolean hypercube
  - [ ] Layout matches `ark-poly`s little-endian, why LE?
- [ ] A lot of helpers already doing some heavy lifting
- PCS (Setup / Commit / Open / Verify, transcript-aware)

## Deferred work

- Sumcheck per MulPerm
- Prover/verifier for MulPerm
- PCS backend support (eg. Hyrax, Multi-linear KZG)
- Benchmarks by itself, w/ HyperPlonk

### Optional

- Prover-provided permutation
- Lookup generalization
- Benchmarks
- Paralellism
- FFT-based speedups

## Skills

- `/initialize-linperm`: Bootstrap project
