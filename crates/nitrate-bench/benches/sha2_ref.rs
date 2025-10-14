use criterion::{criterion_group, criterion_main, Criterion};
use nitrate_utils::double_sha256;

fn bench_double_sha256(c: &mut Criterion) {
    let data = vec![0u8; 80];
    c.bench_function("double_sha256(80B)", |b| b.iter(|| double_sha256(&data)));
}
criterion_group!(benches, bench_double_sha256);
criterion_main!(benches);