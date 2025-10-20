use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

use bola::getrf::getrf;

#[cfg(feature = "profiling")]
use bola::profiling::{self, benchmark};

fn main() -> Result<(), String> {
    // Set the matrix size
    let m = 2048;
    let n = m;
    println!("Testing with a {m}x{n} matrix.");

    // Generate a large vector of random f64 values.
    let mut rng = StdRng::seed_from_u64(1);
    let a: Vec<f64> = (0..(m * n)).map(|_| rng.random()).collect();

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
        }
        elapsed_time += start_time.elapsed().as_secs_f64();
    }

    let elapsed_time = elapsed_time / 10.;

    println!("--- Getrf Result ---");
    println!("L and U factors: {:.3?}", &my_a[my_a.len() - 10..]);
    println!("Pivots: {:?}", &my_ipiv[my_ipiv.len() - 10..]);
    println!("Elapsed time: {} ms", elapsed_time * 1000.);

    println!("\n");

    #[cfg(feature = "profiling")]
    profiling::print_profile(10.);
    
    Ok(())
}
