// Requires openblas linking to run

use bola::gemm::gemm;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use matrixmultiply::dgemm as matrixmultiply_dgemm;
use rand::prelude::*;
use std::time::Duration;

fn benchmark_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("GEMM Comparison");
    group.measurement_time(Duration::from_secs(35));
    group.sample_size(10);

    let sizes = [1024, 2048, 4096];

    for &size in &sizes {
        let m = size;
        let n = size;
        let k = size;
        let lda = m;
        let ldb = k;
        let ldc = m;
        let mut rng = StdRng::seed_from_u64(1);
        let a: Vec<f64> = (0..(m * k)).map(|_| rng.random()).collect();
        let b: Vec<f64> = (0..(k * n)).map(|_| rng.random()).collect();
        let c_init: Vec<f64> = (0..(m * n)).map(|_| rng.random()).collect();

        // // --- Create a golden reference result using CBLAS ---
        // let mut c_golden = c_init.clone();
        // unsafe {
        //     cblas::dgemm(
        //         cblas::Layout::ColumnMajor,
        //         cblas::Transpose::None,
        //         cblas::Transpose::None,
        //         m as i32,
        //         n as i32,
        //         k as i32,
        //         1.0,
        //         &a,
        //         lda as i32,
        //         &b,
        //         ldb as i32,
        //         1.0,
        //         &mut c_golden,
        //         ldc as i32,
        //     );
        // }

        // --- Benchmark cblas::dgemm (our reference) ---
        group.bench_with_input(BenchmarkId::new("CBLAS", size), &size, |bencher, _| {
            bencher.iter_batched(
                || c_init.clone(),
                |mut c_copy| {
                    unsafe {
                        cblas::dgemm(
                            cblas::Layout::ColumnMajor,
                            cblas::Transpose::None,
                            cblas::Transpose::None,
                            m as i32,
                            n as i32,
                            k as i32,
                            1.0,
                            &a,
                            lda as i32,
                            &b,
                            ldb as i32,
                            1.0,
                            &mut c_copy,
                            ldc as i32,
                        );
                    }
                    // Verify the result against the golden copy.
                    // assert_approx_eq(&c_copy, &c_golden, 1e-9);
                },
                criterion::BatchSize::LargeInput,
            );
        });

        // --- Benchmark your custom 'Blast' implementation and verify result ---
        group.bench_with_input(BenchmarkId::new("BOLA", size), &size, |bencher, _| {
            bencher.iter_batched(
                || c_init.clone(),
                |mut c_copy| {
                    unsafe {
                        gemm(
                            'N',
                            'N',
                            m,
                            n,
                            k,
                            1.0,
                            a.as_ptr(),
                            lda,
                            b.as_ptr(),
                            ldb,
                            1.0,
                            c_copy.as_mut_ptr(),
                            ldc,
                        );
                    }
                    // Verify the result against the golden copy.
                    // assert_approx_eq(&c_copy, &c_golden, 1e-9);
                },
                criterion::BatchSize::LargeInput,
            );
        });

        // --- Benchmark matrixmultiply::dgemm and verify result ---
        group.bench_with_input(BenchmarkId::new("MatrixMultiply", size), &size, |bencher, _| {
            bencher.iter_batched(
                || c_init.clone(),
                |mut c_copy| {
                    unsafe {
                        matrixmultiply_dgemm(
                            m,
                            k,
                            n,
                            1.0,
                            a.as_ptr(),
                            1,
                            lda as isize,
                            b.as_ptr(),
                            1,
                            ldb as isize,
                            1.0,
                            c_copy.as_mut_ptr(),
                            1,
                            ldc as isize,
                        );
                    }
                    // Verify the result against the golden copy.
                    // assert_approx_eq(&c_copy, &c_golden, 1e-9);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_gemm);
criterion_main!(benches);
