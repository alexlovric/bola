mod support;

use bola::gemm::gemm;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use matrixmultiply::dgemm as matrixmultiply_dgemm;
use rand::prelude::*;
use std::hint::black_box;

fn benchmark_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("GEMM Comparison");
    let bola_label = support::bola_label();

    let squares = support::matrix_sizes().into_iter().map(|size| (size, size, size));
    for (m, n, k) in squares.chain(std::iter::once((1024, 32, 1024))) {
        let label = if m == n && n == k {
            m.to_string()
        } else {
            format!("{m}x{n}x{k}")
        };
        let mut rng = StdRng::seed_from_u64(1);
        let a: Vec<f64> = (0..m * k).map(|_| rng.random()).collect();
        let b: Vec<f64> = (0..k * n).map(|_| rng.random()).collect();
        let c_init: Vec<f64> = (0..m * n).map(|_| rng.random()).collect();

        let run_cblas = |output: &mut Vec<f64>| unsafe {
            cblas::dgemm(
                cblas::Layout::ColumnMajor,
                cblas::Transpose::None,
                cblas::Transpose::None,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                &a,
                m as i32,
                &b,
                k as i32,
                1.0,
                output,
                m as i32,
            );
            black_box(output);
        };
        let run_bola = |output: &mut Vec<f64>| unsafe {
            gemm(
                'N',
                'N',
                m,
                n,
                k,
                1.0,
                a.as_ptr(),
                m,
                b.as_ptr(),
                k,
                1.0,
                output.as_mut_ptr(),
                m,
            );
            black_box(output);
        };
        let run_matrixmultiply = |output: &mut Vec<f64>| unsafe {
            matrixmultiply_dgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                1,
                m as isize,
                b.as_ptr(),
                1,
                k as isize,
                1.0,
                output.as_mut_ptr(),
                1,
                m as isize,
            );
            black_box(output);
        };

        // Validate each implementation once, outside the measured region.
        let mut golden = c_init.clone();
        run_cblas(&mut golden);
        let mut check = c_init.clone();
        run_bola(&mut check);
        support::assert_close(&check, &golden);
        check.copy_from_slice(&c_init);
        run_matrixmultiply(&mut check);
        support::assert_close(&check, &golden);
        drop((golden, check));

        group.bench_with_input(BenchmarkId::new("CBLAS", &label), &(m, n, k), |bencher, _| {
            bencher.iter_batched_ref(|| c_init.clone(), run_cblas, criterion::BatchSize::LargeInput);
        });
        group.bench_with_input(BenchmarkId::new(&bola_label, &label), &(m, n, k), |bencher, _| {
            bencher.iter_batched_ref(|| c_init.clone(), run_bola, criterion::BatchSize::LargeInput);
        });
        group.bench_with_input(BenchmarkId::new("MatrixMultiply", &label), &(m, n, k), |bencher, _| {
            bencher.iter_batched_ref(|| c_init.clone(), run_matrixmultiply, criterion::BatchSize::LargeInput);
        });
    }
    group.finish();
}

criterion_group! { name = benches; config = support::config(); targets = benchmark_gemm }
criterion_main!(benches);
