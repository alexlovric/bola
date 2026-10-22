use std::time::Duration;

use criterion::Criterion;

pub fn config() -> Criterion {
    // Configure globally so Criterion's CLI can override timing and sample count.
    Criterion::default().measurement_time(Duration::from_secs(5)).sample_size(10)
}

pub fn bola_label() -> String {
    std::env::var("BOLA_BENCH_LABEL").unwrap_or_else(|_| "BOLA".into())
}

/// Select square cases before allocating fixtures or running validation.
pub fn matrix_sizes() -> Vec<usize> {
    match std::env::var("BOLA_BENCH_SIZES") {
        Ok(sizes) => sizes
            .split(',')
            .map(|size| {
                size.trim()
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| n > 0)
                    .expect("BOLA_BENCH_SIZES must contain positive integers separated by commas")
            })
            .collect(),
        Err(_) => vec![32, 64, 128, 256, 512, 1024, 2048, 4096],
    }
}

#[allow(dead_code)]
pub fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() <= 1e-10 * (1.0 + b.abs()), "Mismatch at {i}: {a} != {b}");
    }
}

/// Check P*A*x = L*U*x for three deterministic probes, in O(n²) per probe.
/// This catches incorrect factors without adding another cubic operation to setup.
#[allow(dead_code)]
pub fn assert_lu_residual(original: &[f64], factors: &[f64], pivots: &[i32], n: usize) {
    for probe in 0..3 {
        let x: Vec<_> = (0..n).map(|i| ((i * 17 + probe * 13) % 31) as f64 / 15.0 - 1.0).collect();
        let mut expected = vec![0.0; n];
        let mut ux = vec![0.0; n];
        for j in 0..n {
            for i in 0..n {
                expected[i] += original[i + j * n] * x[j];
                if i <= j {
                    ux[i] += factors[i + j * n] * x[j];
                }
            }
        }
        for (i, &pivot) in pivots.iter().enumerate() {
            assert!((i + 1..=n).contains(&(pivot as usize)));
            expected.swap(i, pivot as usize - 1);
        }
        let mut actual = ux.clone();
        for j in 0..n {
            for i in j + 1..n {
                actual[i] += factors[i + j * n] * ux[j];
            }
        }
        let scale = expected.iter().map(|v| v.abs()).fold(1.0, f64::max);
        for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() <= 1e-8 * scale, "LU residual at {i}: {a} != {b}");
        }
    }
}
