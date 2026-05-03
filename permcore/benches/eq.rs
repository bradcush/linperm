use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::test_rng;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use permcore::eq::eq_eval_table;

fn bench_eq_eval_table(c: &mut Criterion) {
    let mut rng = test_rng();
    let mu = 12;
    let alpha: Vec<Fr> = (0..mu).map(|_| Fr::rand(&mut rng)).collect();
    c.bench_function("eq_eval_table/mu=12", |b| {
        b.iter(|| eq_eval_table(black_box(&alpha)))
    });
}

criterion_group!(benches, bench_eq_eval_table);
criterion_main!(benches);
