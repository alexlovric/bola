use crate::gemm::gemm;
use crate::getrf2::getrf2;
use crate::laswp::laswp;
use crate::trsm::trsm;

#[cfg(feature = "profiling")]
use crate::profiling;

use std::cmp;

const NB: usize = 32;

/// Computes the LU factorisation of a general M-by-N matrix using partial pivoting.
///
/// This function performs an LU decomposition of a matrix `A`, resulting in a
/// factorisation of the form `P * A = L * U`, where:
/// - `P` is a permutation matrix,
/// - `L` is a lower triangular matrix with a unit diagonal,
/// - `U` is an upper triangular matrix.
///
/// # Arguments
/// * `m`    - The number of rows in the matrix `A`.
/// * `n`    - The number of columns in the matrix `A`.
/// * `a`    - A mutable slice representing the `M`-by-`N` matrix `A` in column-major
///   order. On successful exit, this slice is overwritten with the `L` and `U`
///   factors. The unit diagonal of `L` is not stored.
/// * `lda`  - The leading dimension of `A`. It specifies the stride between consecutive
///   columns in memory and must be at least `max(1, m)`.
/// * `ipiv` - A mutable slice that will be filled with the pivot indices. Its length
///   must be at least `min(m, n)`. For each `i` from 0 to `min(m,n)-1`,
///   row `i` was interchanged with row `ipiv[i]`.
///  
/// # Returns
/// * `Ok(())` - If the factorization completed successfully.
/// * `Err(String)`
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
pub unsafe fn getrf(m: usize, n: usize, a: &mut [f64], lda: usize, ipiv: &mut [i32]) -> Result<(), String> {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("GETRF");

    if lda < m.max(1) {
        return Err(format!("Argument 4 to getrf had an illegal value of {lda}"));
    }
    if m == 0 || n == 0 {
        return Ok(());
    }

    let min_mn = m.min(n);

    if NB <= 1 || NB >= min_mn {
        getrf2(m, n, a.as_mut_ptr(), lda, ipiv)?;
    } else {
        for j in (0..min_mn).step_by(NB) {
            let jb = cmp::min(NB, min_mn - j);
            let panel_offset = j + j * lda;

            getrf2(m - j, jb, a.as_mut_ptr().add(panel_offset), lda, &mut ipiv[j..])?;

            ipiv[j..(j + jb)].iter_mut().for_each(|p| *p += j as i32);

            let k1 = (j + 1) as i32;
            let k2 = (j + jb) as i32;

            if j > 0 {
                laswp(j, a, lda, k1, k2, ipiv, 1);
            }

            if j + jb < n {
                let swap_offset = (j + jb) * lda;
                laswp(n - j - jb, &mut a[swap_offset..], lda, k1, k2, ipiv, 1);

                // A22 := A22 - L21 * U12
                let a_ptr = a.as_mut_ptr();
                // --- IMPROVEMENT: Calculate pointers directly for TRSM ---
                let l11_ptr = a_ptr.add(j + j * lda);
                let u12_ptr = a_ptr.add(j + (j + jb) * lda);

                trsm('L', 'L', 'N', 'U', jb, n - j - jb, 1.0, l11_ptr, lda, u12_ptr, lda);

                if j + jb < m {
                    gemm(
                        'N',
                        'N',
                        m - j - jb,
                        n - j - jb,
                        jb,
                        -1.0,
                        a_ptr.add((j + jb) + j * lda),
                        lda,
                        u12_ptr,
                        lda,
                        1.0,
                        a_ptr.add((j + jb) + (j + jb) * lda),
                        lda,
                    );
                }
            }
        }
    }
    Ok(())
}
