// Requires openblas linking to run

use bola::getrf::getrf;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use std::time::Duration;

#[allow(unused_must_use)]
fn benchmark_getrf(c: &mut Criterion) {
    let mut group = c.benchmark_group("LU Decomposition");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    let sizes = [1024, 2048, 4096];

    for &size in &sizes {
        let m = size;
        let n = size;
        let lda = m;
        let mut rng = StdRng::seed_from_u64(1);
        let a: Vec<f64> = (0..m * n).map(|_| rng.random()).collect();
        let ipiv_template = vec![0; m.min(n)];

        group.bench_with_input(BenchmarkId::new("BOLA", size), &size, |b, &_size| {
            b.iter_batched(
                || {
                    let a_copy = a.clone();
                    let ipiv = ipiv_template.clone();
                    (a_copy, ipiv)
                },
                |(mut a_copy, mut ipiv)| {
                    unsafe { getrf(m, n, &mut a_copy, lda, &mut ipiv) };
                },
                criterion::BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("Lapack", size), &size, |b, &_size| {
            b.iter_batched(
                || {
                    let a_copy = a.clone();
                    let ipiv = ipiv_template.clone();
                    (a_copy, ipiv)
                },
                |(mut a_copy, mut ipiv)| {
                    unsafe {
                        lapack::dgetrf(m as i32, n as i32, &mut a_copy, lda as i32, &mut ipiv, &mut 0);
                    }
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_getrf);
criterion_main!(benches);
