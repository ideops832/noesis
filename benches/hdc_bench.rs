//! HDC benchmarks using criterion.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::SeedableRng;
use rand::rngs::StdRng;

use noesis::hdc::binary::BinaryHV;
use noesis::hdc::hypervector::HyperVector;

fn bench_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind");
    for &dim in &[1024, 10048] {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryHV::random(dim, &mut rng);
        let b = BinaryHV::random(dim, &mut rng);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| BinaryHV::bind(&a, &b));
        });
    }
    group.finish();
}

fn bench_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");
    for &dim in &[1024, 10048] {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryHV::random(dim, &mut rng);
        let b = BinaryHV::random(dim, &mut rng);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| a.similarity(&b));
        });
    }
    group.finish();
}

fn bench_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("bundle");
    for &dim in &[1024, 10048] {
        let mut rng = StdRng::seed_from_u64(42);
        let vecs: Vec<BinaryHV> = (0..5).map(|_| BinaryHV::random(dim, &mut rng)).collect();
        let refs: Vec<&BinaryHV> = vecs.iter().collect();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            let mut rng2 = StdRng::seed_from_u64(99);
            bench.iter(|| BinaryHV::bundle(&refs, &mut rng2));
        });
    }
    group.finish();
}

fn bench_permute(c: &mut Criterion) {
    let mut group = c.benchmark_group("permute");
    for &dim in &[1024, 10048] {
        let mut rng = StdRng::seed_from_u64(42);
        let v = BinaryHV::random(dim, &mut rng);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| v.permute(1));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_bind, bench_similarity, bench_bundle, bench_permute);
criterion_main!(benches);
