use crate::syrk::syrk;
use crate::trsm::trsm;

#[cfg(feature = "profiling")]
use crate::profiling;

/// Computes the Cholesky factorisation of a symmetric, positive-definite matrix (unblocked version).
/// - If `uplo` = 'U', then $A = U^T U$, where `U` is an upper triangular matrix.
/// - If `uplo` = 'L', then $A = L L^T$, where `L` is a lower triangular matrix.
///
/// # Arguments
/// * `uplo` - A character specifying which triangular part of `A` is stored and computed:
///   - 'U' or 'u': Upper triangle of `A`.
///   - 'L' or 'l': Lower triangle of `A`.
/// * `n`    - The order of the matrix `A`. `n` must be non-negative.
/// * `a`    - A raw mutable pointer to the first element of the `N`-by-`N` matrix `A`
///   (in column-major order). On exit, the specified `uplo` part of `A`
///   is overwritten with the Cholesky factor.
/// * `lda`  - The leading dimension of the matrix `A`. `lda` must be at least `max(1, n)`.
///
/// # Returns
/// * `Ok(())` - If the factorization completed successfully.
/// * `Err(String)`
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
pub unsafe fn potrf2(uplo: char, n: usize, a: *mut f64, lda: usize) -> Result<(), String> {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("POTRF2");

    match potrf2_recursive(uplo, n, a, lda) {
        Ok(()) => Ok(()),
        Err(info) => Err(format!("Matrix is not positive-definite. Failure at column {}.", info)),
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn potrf2_recursive(uplo: char, n: usize, a: *mut f64, lda: usize) -> Result<(), i32> {
    if n == 0 {
        return Ok(());
    }

    if n == 1 {
        let a11 = *a;
        if a11 > 0.0 {
            *a = a11.sqrt();
            Ok(())
        } else {
            Err(1)
        }
    } else {
        let n1 = n / 2;
        let n2 = n - n1;

        potrf2_recursive(uplo, n1, a, lda)?;

        if uplo == 'U' {
            let a11 = a;
            let a12 = a.add(n1 * lda);
            trsm('L', 'U', 'T', 'N', n1, n2, 1.0, a11, lda, a12, lda);

            let a22 = a.add(n1 + n1 * lda);
            syrk('U', 'T', n2, n1, -1.0, a12, lda, 1.0, a22, lda);

            if let Err(info) = potrf2_recursive(uplo, n2, a22, lda) {
                return Err(info + n1 as i32);
            }
        } else {
            let a11 = a;
            let a21 = a.add(n1);
            trsm('R', 'L', 'T', 'N', n2, n1, 1.0, a11, lda, a21, lda);

            let a22 = a.add(n1 + n1 * lda);
            syrk('L', 'N', n2, n1, -1.0, a21, lda, 1.0, a22, lda);

            if let Err(info) = potrf2_recursive(uplo, n2, a22, lda) {
                return Err(info + n1 as i32);
            }
        }
        Ok(())
    }
}
