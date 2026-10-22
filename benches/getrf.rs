mod support;

use bola::getrf::getrf;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use std::hint::black_box;

fn benchmark_getrf(c: &mut Criterion) {
    let mut group = c.benchmark_group("LU Decomposition");
    let bola_label = support::bola_label();

    for size in support::matrix_sizes() {
        let mut rng = StdRng::seed_from_u64(1);
        let a: Vec<f64> = (0..size * size).map(|_| rng.random()).collect();
        let setup = || (a.clone(), vec![0; size]);
        let run_bola = |(work, pivots): &mut (Vec<f64>, Vec<i32>)| unsafe {
            getrf(size, size, work, size, pivots).expect("BOLA factorization failed");
            black_box((work, pivots));
        };
        let mut check = setup();
        run_bola(&mut check);
        support::assert_lu_residual(&a, &check.0, &check.1, size);
        drop(check);
        group.bench_with_input(BenchmarkId::new(&bola_label, size), &size, |bencher, _| {
            bencher.iter_batched_ref(setup, run_bola, criterion::BatchSize::LargeInput);
        });
        #[cfg(feature = "external-blas")]
        {
            let run_lapack = |(work, pivots): &mut (Vec<f64>, Vec<i32>)| unsafe {
                let mut info = 0;
                lapack::dgetrf(size as i32, size as i32, work, size as i32, pivots, &mut info);
                assert_eq!(info, 0, "LAPACK factorization failed");
                black_box((work, pivots));
            };
            let mut check = setup();
            run_lapack(&mut check);
            support::assert_lu_residual(&a, &check.0, &check.1, size);
            drop(check);
            group.bench_with_input(BenchmarkId::new("Lapack", size), &size, |bencher, _| {
                bencher.iter_batched_ref(setup, run_lapack, criterion::BatchSize::LargeInput);
            });
        }
    }
    group.finish();
}

criterion_group! { name = benches; config = support::config(); targets = benchmark_getrf }
criterion_main!(benches);
