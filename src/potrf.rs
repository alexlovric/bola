use std::cmp;

use crate::{gemm::gemm, potrf2::potrf2, syrk::syrk, trsm::trsm};

#[cfg(feature = "profiling")]
use crate::profiling;

const NB: usize = 32;

/// Computes the Cholesky factorisation of a symmetric, positive-definite matrix.
/// - If `uplo` = 'U', then $A = U^T U$, where `U` is an upper triangular matrix.
/// - If `uplo` = 'L', then $A = L L^T$, where `L` is a lower triangular matrix.
///
/// # Arguments
/// * `uplo` - A character specifying which triangular part of `A` is stored:
///   - 'U' or 'u': Upper triangle of `A` is stored.
///   - 'L' or 'l': Lower triangle of `A` is stored.
/// * `n`    - The order of the matrix `A`. `n` must be non-negative.
/// * `a`    - A raw mutable pointer to the first element of the `N`-by-`N` matrix `A`
///   (in column-major order). On exit, the specified `uplo` part of `A`
///   is overwritten with the corresponding factor `U` or `L`.
/// * `lda`  - The leading dimension of the matrix `A`. `lda` must be at least `max(1, n)`.
///
/// # Returns
/// * `Ok(())` - If the factorization completed successfully.
/// * `Err(String)`
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
pub unsafe fn potrf(uplo: char, n: usize, a: *mut f64, lda: usize) -> Result<(), String> {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("POTRF");

    if uplo != 'U' && uplo != 'L' {
        return Err("Argument 1 to potrf had an illegal value".to_string());
    }
    if lda < n.max(1) {
        return Err("Argument 4 to potrf had an illegal value".to_string());
    }

    if NB <= 1 || NB >= n {
        return potrf2(uplo, n, a, lda);
    }

    if uplo == 'U' {
        for j in (0..n).step_by(NB) {
            let jb = cmp::min(NB, n - j);
            let a_jj = a.add(j + j * lda);
            let a_1j = a.add(j * lda);

            syrk('U', 'T', jb, j, -1.0, a_1j, lda, 1.0, a_jj, lda);

            potrf2('U', jb, a_jj, lda)?;

            if j + jb < n {
                let n_rem = n - j - jb;
                let a_1_jb = a.add((j + jb) * lda);
                let a_j_jb = a.add(j + (j + jb) * lda);

                gemm(
                    'T', 'N', jb, n_rem, j, -1.0, a_1j, lda, a_1_jb, lda, 1.0, a_j_jb, lda,
                );

                trsm(
                    'L', 'U', 'T', 'N', jb, n_rem, 1.0, a_jj, lda, a_j_jb, lda,
                );
            }
        }
    } else {
        for j in (0..n).step_by(NB) {
            let jb = cmp::min(NB, n - j);
            let a_jj = a.add(j + j * lda);
            let a_j0 = a.add(j);

            syrk('L', 'N', jb, j, -1.0, a_j0, lda, 1.0, a_jj, lda);

            potrf2('L', jb, a_jj, lda)?;

            if j + jb < n {
                let n_rem = n - j - jb;
                
                let a_jb0 = a.add(j + jb);
                
                let a_jbj = a.add(j + jb + j * lda);

                gemm(
                    'N', 'T', n_rem, jb, j, -1.0, a_jb0, lda, a_j0, lda, 1.0, a_jbj, lda,
                );

                trsm(
                    'R', 'L', 'T', 'N', n_rem, jb, 1.0, a_jj, lda, a_jbj, lda,
                );
            }
        }
    }
    Ok(())
}