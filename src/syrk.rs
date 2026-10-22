use std::cmp;

use crate::utilities::dot_product;
use crate::{gemm::gemm, utilities::daxpy};

#[cfg(feature = "profiling")]
use crate::profiling;

const NB: usize = 32; // A reasonable block size

/// Performs a symmetric rank-k update on a matrix.
///
/// This function computes one of the following operations:
/// - `C := alpha * A * A^T + beta * C`  (if `trans` = 'N')
/// - `C := alpha * A^T * A + beta * C`  (if `trans` = 'T')
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn syrk(
    uplo: char,
    trans: char,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("SYRK");

    if n == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0) {
        return;
    }

    // Scale C by beta
    if beta == 0.0 {
        for j in 0..n {
            if uplo == 'U' {
                for i in 0..=j {
                    *c.add(i + j * ldc) = 0.0;
                }
            } else {
                for i in j..n {
                    *c.add(i + j * ldc) = 0.0;
                }
            }
        }
    } else if beta != 1.0 {
        for j in 0..n {
            if uplo == 'U' {
                for i in 0..=j {
                    *c.add(i + j * ldc) *= beta;
                }
            } else {
                for i in j..n {
                    *c.add(i + j * ldc) *= beta;
                }
            }
        }
    }

    if alpha == 0.0 {
        return;
    }

    // --- Main blocked algorithm ---
    if uplo == 'U' {
        for j in (0..n).step_by(NB) {
            let jb = cmp::min(n - j, NB);
            for p in (0..k).step_by(NB) {
                let pb = cmp::min(k - p, NB);
                // Update the diagonal block C(j,j)
                syrk_kernel(
                    'U',
                    trans,
                    jb,
                    pb,
                    alpha,
                    if trans == 'N' {
                        a.add(j + p * lda)
                    } else {
                        a.add(p + j * lda)
                    },
                    lda,
                    1.0,
                    c.add(j + j * ldc),
                    ldc,
                    trans,
                );
                // Update the off-diagonal blocks C(i,j) for i < j
                for i in (0..j).step_by(NB) {
                    let ib = cmp::min(j - i, NB);
                    if trans == 'T' {
                        gemm(
                            'T',
                            'N',
                            ib,
                            jb,
                            pb,
                            alpha,
                            a.add(p + i * lda),
                            lda,
                            a.add(p + j * lda),
                            lda,
                            1.0,
                            c.add(i + j * ldc),
                            ldc,
                        );
                    } else {
                        gemm(
                            'N',
                            'T',
                            ib,
                            jb,
                            pb,
                            alpha,
                            a.add(i + p * lda),
                            lda,
                            a.add(j + p * lda),
                            lda,
                            1.0,
                            c.add(i + j * ldc),
                            ldc,
                        );
                    }
                }
            }
        }
    } else {
        // uplo == 'L'
        for j in (0..n).step_by(NB) {
            let jb = cmp::min(n - j, NB);
            for p in (0..k).step_by(NB) {
                let pb = cmp::min(k - p, NB);
                // Update the diagonal block C(j,j)
                syrk_kernel(
                    'L',
                    trans,
                    jb,
                    pb,
                    alpha,
                    if trans == 'N' {
                        a.add(j + p * lda)
                    } else {
                        a.add(p + j * lda)
                    },
                    lda,
                    1.0,
                    c.add(j + j * ldc),
                    ldc,
                    trans,
                );
                // Update the off-diagonal blocks C(i,j) for i > j
                for i in (j + jb..n).step_by(NB) {
                    let ib = cmp::min(n - i, NB);
                    if trans == 'N' {
                        gemm(
                            'N',
                            'T',
                            ib,
                            jb,
                            pb,
                            alpha,
                            a.add(i + p * lda),
                            lda,
                            a.add(j + p * lda),
                            lda,
                            1.0,
                            c.add(i + j * ldc),
                            ldc,
                        );
                    } else {
                        gemm(
                            'T',
                            'N',
                            ib,
                            jb,
                            pb,
                            alpha,
                            a.add(p + i * lda),
                            lda,
                            a.add(p + j * lda),
                            lda,
                            1.0,
                            c.add(i + j * ldc),
                            ldc,
                        );
                    }
                }
            }
        }
    }
}

/// A simple, non-blocked kernel for the symmetric rank-k update.
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::too_many_arguments)]
unsafe fn syrk_kernel(
    uplo: char,
    trans: char,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    _beta: f64,
    c: *mut f64,
    ldc: usize,
    original_trans: char,
) {
    if trans == 'N' {
        if uplo == 'U' {
            for j in 0..n {
                for l in 0..k {
                    let temp = alpha * *a.add(j + l * lda);
                    if temp != 0.0 {
                        daxpy(j + 1, temp, a.add(l * lda), 1, c.add(j * ldc), 1);
                    }
                }
            }
        } else {
            // Lower
            for j in 0..n {
                for l in 0..k {
                    let temp = alpha * *a.add(j + l * lda);
                    if temp != 0.0 {
                        daxpy(n - j, temp, a.add(j + l * lda), 1, c.add(j + j * ldc), 1);
                    }
                }
            }
        }
    } else {
        // trans == 'T'
        if uplo == 'U' {
            for j in 0..n {
                for i in 0..=j {
                    let row_i_ptr = if original_trans == 'T' { a.add(i * lda) } else { a.add(i) };
                    let row_j_ptr = if original_trans == 'T' { a.add(j * lda) } else { a.add(j) };
                    let lda_a = if original_trans == 'T' { 1 } else { lda };
                    let temp = dot_product(k, row_i_ptr, lda_a, row_j_ptr, lda_a);
                    *c.add(i + j * ldc) += alpha * temp;
                }
            }
        } else {
            // Lower
            for j in 0..n {
                for i in j..n {
                    let row_i_ptr = if original_trans == 'T' { a.add(i * lda) } else { a.add(i) };
                    let row_j_ptr = if original_trans == 'T' { a.add(j * lda) } else { a.add(j) };
                    let lda_a = if original_trans == 'T' { 1 } else { lda };
                    let temp = dot_product(k, row_i_ptr, lda_a, row_j_ptr, lda_a);
                    *c.add(i + j * ldc) += alpha * temp;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::utilities::assert_approx_eq;

    use super::*;

    #[test]
    fn test_syrk_lower_no_transpose() {
        // Test C := A * A^T + C, with uplo = 'L'
        let n = 3;
        let k = 2;
        let lda = 3;
        let ldc = 3;

        // A is 3x2, C is 3x3
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // Column-major

        // Expected result for C's lower triangle
        // A*A^T = [[17, 23, 29], [23, 34, 45], [29, 45, 61]]
        let c_expected = vec![18.0, 23.0, 28.0, 0.0, 30.0, 37.0, 0.0, 0.0, 46.0];

        unsafe {
            syrk('L', 'N', n, k, 1.0, a.as_ptr(), lda, 1.0, c.as_mut_ptr(), ldc);
        }

        // Extract and check the lower triangle
        let mut c_lower = vec![0.0; 9];
        for j in 0..n {
            for i in j..n {
                c_lower[i + j * ldc] = c[i + j * ldc];
            }
        }
        assert_approx_eq(&c_lower, &c_expected, 1e-9);
    }

    #[test]
    fn test_syrk_upper_transpose() {
        // Test C := A^T * A + C, with uplo = 'U'
        let n = 2;
        let k = 3;
        let lda = 3;
        let ldc = 2;

        // A is 3x2, C is 2x2
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0]; // Column-major

        // Expected result for C's upper triangle
        // A^T*A = [[14, 32], [32, 77]]
        let c_expected = vec![15.0, 0.0, 33.0, 78.0];

        unsafe {
            syrk('U', 'T', n, k, 1.0, a.as_ptr(), lda, 1.0, c.as_mut_ptr(), ldc);
        }

        // Extract and check the upper triangle
        let mut c_upper = vec![0.0; 4];
        for j in 0..n {
            for i in 0..=j {
                c_upper[i + j * ldc] = c[i + j * ldc];
            }
        }
        assert_approx_eq(&c_upper, &c_expected, 1e-9);
    }
}
