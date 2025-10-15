use std::{cmp, slice};

use crate::ger::ger;
use crate::idamax::idamax;
use crate::scal::scal_kernel;
use crate::{gemm::gemm, laswp::laswp, trsm::trsm};

#[cfg(feature = "profiling")]
use crate::profiling;

const RECURSION_STOP_SIZE: usize = 32;

/// Computes the LU factorization of a general M-by-N matrix using partial pivoting (unsafe version).
///
/// This function is an `unsafe` variant of `getrf` that operates on a raw pointer to the
/// matrix data. It performs an LU decomposition of a matrix `A`, resulting in a
/// factorization of the form `P * A = L * U`, where:
/// - `P` is a permutation matrix,
/// - `L` is a lower triangular matrix with a unit diagonal,
/// - `U` is an upper triangular matrix.
///
/// # Arguments
/// * `m`    - The number of rows in the matrix `A`.
/// * `n`    - The number of columns in the matrix `A`.
/// * `a`    - A raw mutable pointer to the first element of the `M`-by-`N` matrix `A`,
///   which is assumed to be in column-major order. On successful exit, the
///   memory region is overwritten with the `L` and `U` factors. The unit
///   diagonal of `L` is not stored.
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
pub unsafe fn getrf2(m: usize, n: usize, a: *mut f64, lda: usize, ipiv: &mut [i32]) -> Result<(), String> {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("GETRF2");

    if lda < m.max(1) {
        return Err(format!("Argument 4 to getrf2 had an illegal value of {}", lda));
    }
    if m == 0 || n == 0 {
        return Ok(());
    }

    if m <= RECURSION_STOP_SIZE || n <= RECURSION_STOP_SIZE {
        return iterative_getrf2(m, n, a, lda, ipiv);
    }

    let n1 = cmp::min(m, n) / 2;
    let n2 = n - n1;

    getrf2(m, n1, a, lda, &mut ipiv[0..n1])?;

    laswp(
        n2,
        slice::from_raw_parts_mut(a.add(n1 * lda), (n - n1) * lda),
        lda,
        1,
        n1 as i32,
        ipiv,
        1,
    );

    trsm('L', 'L', 'N', 'U', n1, n2, 1.0, a, lda, a.add(n1 * lda), lda);

    if n1 < m {
        gemm(
            'N',
            'N',
            m - n1,
            n2,
            n1,
            -1.0,
            a.add(n1),
            lda,
            a.add(n1 * lda),
            lda,
            1.0,
            a.add(n1 + n1 * lda),
            lda,
        );
    }

    if n1 < m {
        getrf2(m - n1, n2, a.add(n1 + n1 * lda), lda, &mut ipiv[n1..])?;
    }

    ipiv.iter_mut().take(cmp::min(m, n)).skip(n1).for_each(|p| *p += n1 as i32);

    laswp(
        n1,
        slice::from_raw_parts_mut(a, n1 * lda),
        lda,
        (n1 + 1) as i32,
        cmp::min(m, n) as i32,
        ipiv,
        1,
    );

    Ok(())
}

/// Computes the LU factorization of a general M-by-N matrix using a tiled, iterative approach (unsafe version).
///
/// This function performs an LU decomposition on a matrix `A`, producing a factorization
/// of the form `P * A = L * U`. It is an `unsafe` variant that operates on raw pointers
/// and employs a tiled (or block-based) algorithm. This iterative approach processes the
/// matrix in smaller blocks, which can improve cache performance and expose more parallelism
/// compared to a standard recursive or right-looking algorithm.
///
/// # Arguments
/// * `m`    - The number of rows in the matrix `A`.
/// * `n`    - The number of columns in the matrix `A`.
/// * `a`    - A raw mutable pointer to the first element of the `M`-by-`N` matrix `A`,
///   which is assumed to be in column-major order. On successful exit, this
///   memory region is overwritten with the `L` and `U` factors.
/// * `lda`  - The leading dimension of `A`, representing the memory stride between
///   consecutive columns. It must be at least `max(1, m)`.
/// * `ipiv` - A mutable slice to be filled with the pivot indices. Its length must be
///   at least `min(m, n)`. Row `i` is swapped with row `ipiv[i]`.
///
/// # Returns
/// * `Ok(())` - If the factorization completed successfully.
/// * `Err(String)`
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
unsafe fn iterative_getrf2(m: usize, n: usize, a: *mut f64, lda: usize, ipiv: &mut [i32]) -> Result<(), String> {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("GETRF2_IT");

    let a_slice = slice::from_raw_parts_mut(a, n * lda);
    let min_mn = m.min(n);

    for j in 0..min_mn {
        let jp = j + idamax(m - j, a.add(j + j * lda));

        ipiv[j] = (jp + 1) as i32;

        if a_slice[jp + j * lda] != 0.0 {
            if jp != j {
                #[cfg(feature = "profiling")]
                let _timer = profiling::ScopedTimer::new("SWAP");
                for k in 0..n {
                    a_slice.swap(j + k * lda, jp + k * lda);
                }
            }
            if j < m - 1 {
                #[cfg(feature = "profiling")]
                let _timer = profiling::ScopedTimer::new("SCAL");

                let ajj = a_slice[j + j * lda];
                let inv_ajj = 1.0 / ajj;
                let m_rem = m - (j + 1);
                let col_ptr = a.add(j + 1 + j * lda);

                let processed = scal_kernel(m_rem, inv_ajj, col_ptr);

                for i in processed..m_rem {
                    *a_slice.get_unchecked_mut(j + 1 + i + j * lda) *= inv_ajj;
                }
            }
        } else {
            return Err(format!("Matrix is singular. The pivot in column {} is zero.", j + 1));
        }

        if j < min_mn - 1 {
            ger(
                m - j - 1,
                n - j - 1,
                -1.0,
                a.add(j + 1 + j * lda),
                a.add(j + (j + 1) * lda),
                lda,
                a.add(j + 1 + (j + 1) * lda),
                lda,
            );
        }
    }

    Ok(())
}
