use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::test_rng;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use permcore::eq::eq_eval_table;

fn bench(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut group = c.benchmark_group("eq_eval_table");
    for mu in [8usize, 10, 12, 14] {
        let alpha: Vec<Fr> = (0..mu).map(|_| Fr::rand(&mut rng)).collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(mu),
            &alpha,
            |b, alpha| b.iter(|| eq_eval_table(black_box(alpha))),
        );
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
