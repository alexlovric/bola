// Requires openblas linking to run

#![allow(unused, dead_code)]
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::time::Instant;

use bola::{
    gemm::gemm,
    ger::ger,
    getrf::getrf,
    getrf2::getrf2,
    laswp::laswp,
    potrf::potrf,
    potrf2::potrf2,
    utilities::dot_product,
    trsm::trsm,
};

#[cfg(feature = "profiling")]
use bola::profiling::{self, benchmark};

/// Generates a numerically stable Symmetric Positive-Definite (SPD) matrix.
pub fn generate_spd_matrix(n: usize, rng: &mut StdRng) -> Vec<f64> {
    let x: Vec<f64> = (0..(n * n)).map(|_| rng.random_range(-1.0..1.0)).collect();
    let lda = n;

    let mut a = vec![0.0; n * n];

    let x_ptr_addr = x.as_ptr() as usize;
    let a_ptr_addr = a.as_mut_ptr() as usize;

    (0..n).into_par_iter().for_each(|j| {
        let x = x_ptr_addr as *const f64;
        let a = a_ptr_addr as *mut f64;

        for i in j..n {
            let col_i_ptr = unsafe { x.add(i * lda) };
            let col_j_ptr = unsafe { x.add(j * lda) };

            let dot = unsafe { dot_product(n, col_i_ptr, 1, col_j_ptr, 1) };

            unsafe {
                *a.add(i + j * lda) = dot;
            }
        }
    });

    for j in 0..n {
        for i in (j + 1)..n {
            a[j + i * lda] = a[i + j * lda];
        }
    }

    for i in 0..n {
        a[i + i * lda] += 1.0;
    }
    a
}

fn main() -> Result<(), String> {
    // Set the matrix size
    let m = 2048;
    let n = m;
    println!("Testing with a {m}x{n} matrix.");

    // compare_dgemm(m, 10);
    // compare_dger(m, 10);
    // compare_dlaswp(m, 10);
    // compare_dtrsm(m, 10);
    // compare_dgetrf2(m, 10);

    // profiling::reset_counters();

    // compare_getrf(m, 9);

    // Generate a large vector of random f64 values.
    let mut rng = StdRng::seed_from_u64(1);
    // let a: Vec<f64> = (0..(m * n)).map(|_| rng.random()).collect();
    let a = generate_spd_matrix(m, &mut rng);

    let lda = m;
    let ipiv = vec![0; m.min(n)];

    // Bola implementation
    let mut elapsed_time = 0.0;
    let mut my_a = a.clone();
    let mut my_ipiv = ipiv.clone();
    for _ in 0..10 {
        my_a = a.clone();
        my_ipiv = ipiv.clone();
        let start_time = Instant::now();
        unsafe {
            getrf(m, n, &mut my_a, lda, &mut my_ipiv)?;
            // potrf('U', n, my_a.as_mut_ptr(), lda)?;
            // potrf2('U', n, my_a.as_mut_ptr(), lda)?;
        }
        elapsed_time += start_time.elapsed().as_secs_f64();
    }

    let elapsed_time = elapsed_time / 10.;

    println!("--- My getrf Result ---");
    println!("L and U factors: {:.3?}", &my_a[my_a.len() - 10..]);
    println!("Pivots: {:?}", &my_ipiv[my_ipiv.len() - 10..]);
    println!("Elapsed time: {} ms", elapsed_time * 1000.);

    println!("\n");

    // LAPACK implementation

    // This is an FFI call, so it's unsafe.
    let mut elapsed_time = 0.0;
    let mut blas_a = a.clone();
    let mut blas_ipiv = ipiv.clone();
    let mut blas_info = 0;
    for _ in 0..10 {
        blas_a = a.clone();
        blas_ipiv = ipiv.clone();
        blas_info = 0;
        let start_time = Instant::now();
        unsafe {
            lapack::dgetrf(m as i32, n as i32, &mut blas_a, lda as i32, &mut blas_ipiv, &mut blas_info);
            // lapack::dpotrf(b'U', n as i32, &mut blas_a, lda as i32, &mut blas_info);
            // lapack::dpotrf2(b'U', n as i32, &mut blas_a, lda as i32, &mut blas_info);
        }
        elapsed_time += start_time.elapsed().as_secs_f64();
    }
    let elapsed_time = elapsed_time / 10.;

    println!("--- LAPACK dgetrf Result ---");
    if blas_info == 0 {
        // FIX: Print the variables that were actually modified by the lapack call.
        println!("L and U factors: {:.3?}", &blas_a[blas_a.len() - 10..]);
        println!("Pivots: {:?}", &blas_ipiv[blas_ipiv.len() - 10..]);
        println!("Elapsed time: {} ms", elapsed_time * 1000.);
    } else {
        println!("Matrix is singular. First zero pivot at column {blas_info}");
    }

    #[cfg(feature = "profiling")]
    profiling::print_profile(10.);

    Ok(())
}
