//! Repeatable native-kernel timings without an external BLAS installation.
//! Run with `RAYON_NUM_THREADS=4 cargo run --release --example profile -- [filter]`.
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use bola::{gemm::gemm, getrf::getrf, potrf::potrf, trsm::trsm};

fn matrix(rows: usize, cols: usize) -> Vec<f64> {
    let mut state = 0x1234_5678_9abc_def0_u64;
    (0..rows * cols)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        })
        .collect()
}

fn measure<S>(name: &str, filter: &str, state: &mut S, mut setup: impl FnMut(&mut S), mut run: impl FnMut(&mut S)) {
    if !name.contains(filter) {
        return;
    }
    setup(state);
    run(state); // Warm the worker pool, code and packing buffers.
    #[cfg(feature = "profiling")]
    bola::profiling::reset_counters();
    let mut samples = Vec::new();
    #[cfg(feature = "profiling")]
    let mut total_iterations = 0;
    for _ in 0..7 {
        let mut elapsed = Duration::ZERO;
        let mut iterations = 0;
        while elapsed < Duration::from_millis(30) {
            setup(state);
            let start = Instant::now();
            run(state);
            elapsed += start.elapsed();
            iterations += 1;
        }
        #[cfg(feature = "profiling")]
        {
            total_iterations += iterations;
        }
        samples.push(elapsed.as_secs_f64() * 1e6 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    println!("{name},{:.3},{:.3},{:.3}", samples[3], samples[0], samples[6]);
    #[cfg(feature = "profiling")]
    bola::profiling::print_profile(total_iterations as f64);
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    eprintln!("Rayon threads: {}", rayon::current_num_threads());
    println!("case,median_us,min_us,max_us");
    for (m, n, k) in [
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (32, 1024, 32),
        (1024, 32, 1024),
        (8, 256, 8),
        (4096, 1, 64),
    ] {
        if !format!("gemm/{m}x{n}x{k}").contains(&filter) {
            continue;
        }
        let a = matrix(m, k);
        let b = matrix(k, n);
        let mut c = vec![0.0; m * n];
        measure(
            &format!("gemm/{m}x{n}x{k}"),
            &filter,
            &mut c,
            |_| {},
            |c| unsafe {
                gemm('N', 'N', m, n, k, 1.0, a.as_ptr(), m, b.as_ptr(), k, 0.0, c.as_mut_ptr(), m);
                black_box(&c);
            },
        );
    }
    for n in [512, 1024, 2048] {
        let a = matrix(n, n);
        let mut state = (a.clone(), vec![0; n]);
        measure(
            &format!("getrf/{n}"),
            &filter,
            &mut state,
            |(work, _)| work.copy_from_slice(&a),
            |(work, pivots)| unsafe {
                getrf(n, n, work, n, pivots).unwrap();
                black_box((&work, &pivots));
            },
        );
    }
    for n in [512, 1024] {
        let mut a = matrix(n, n);
        // Symmetric strict diagonal dominance makes this matrix positive definite.
        for j in 0..n {
            a[j + j * n] = n as f64;
            for i in 0..j {
                a[i + j * n] = a[j + i * n];
            }
        }
        for uplo in ['U', 'L'] {
            let mut work = a.clone();
            measure(
                &format!("potrf/{uplo}/{n}"),
                &filter,
                &mut work,
                |work| work.copy_from_slice(&a),
                |work| unsafe {
                    potrf(uplo, n, work.as_mut_ptr(), n).unwrap();
                    black_box(&work);
                },
            );
        }
    }
    for (m, n) in [(32, 1024), (256, 256)] {
        let mut a = matrix(m, m);
        for i in 0..m {
            a[i + i * m] = m as f64;
        }
        let b = matrix(m, n);
        let mut work = b.clone();
        measure(
            &format!("trsm/{m}x{n}"),
            &filter,
            &mut work,
            |work| work.copy_from_slice(&b),
            |work| unsafe {
                trsm('L', 'L', 'N', 'N', m, n, 1.0, a.as_ptr(), m, work.as_mut_ptr(), m);
                black_box(&work);
            },
        );
    }
}
