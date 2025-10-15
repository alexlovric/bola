use std::cmp;

#[cfg(feature = "profiling")]
use crate::profiling;
use crate::utilities::dot_product;
use crate::{gemm::gemm, utilities::daxpy_update};

const NB: usize = 32; // A reasonable block size

/// Performs a symmetric rank-k update on a matrix.
/// - `C := alpha * A * A^T + beta * C`  (if `trans` = 'N')
/// - `C := alpha * A^T * A + beta * C`  (if `trans` = 'T')
///
/// Where `C` is an N-by-N symmetric matrix, `A` is a matrix, and `alpha` and
/// `beta` are scalars. Only the specified `uplo` triangle of `C` is updated.
///
/// # Arguments
/// * `uplo`  - Specifies whether the upper ('U') or lower ('L') triangular part of `C` is to be referenced.
/// * `trans` - Specifies the operation: 'N' for `A * A^T` or 'T' for `A^T * A`.
/// * `n`     - The order of the matrix `C`.
/// * `k`     - The number of columns of `A` if `trans` is 'N', or rows of `A` if `trans` is 'T'.
/// * `alpha` - The scalar multiplier `alpha`.
/// * `a`     - A raw constant pointer to the first element of matrix `A`.
/// * `lda`   - The leading dimension of `A`.
/// * `beta`  - The scalar multiplier `beta`.
/// * `c`     - A raw mutable pointer to the first element of the symmetric matrix `C`.
/// * `ldc`   - The leading dimension of `C`.
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

    // First, scale the entire C matrix by beta.
    if beta == 0.0 {
        for j in 0..n {
            for i in 0..=j {
                if uplo == 'U' {
                    *c.add(i + j * ldc) = 0.0;
                } else {
                    *c.add(j + i * ldc) = 0.0;
                }
            }
        }
    } else if beta != 1.0 {
        for j in 0..n {
            for i in 0..=j {
                if uplo == 'U' {
                    *c.add(i + j * ldc) *= beta;
                } else {
                    *c.add(j + i * ldc) *= beta;
                }
            }
        }
    }

    if alpha == 0.0 {
        return;
    }

    if uplo == 'U' {
        if trans == 'T' {
            // C += alpha * A'*A
            for p in (0..k).step_by(NB) {
                let pb = cmp::min(k - p, NB);
                for j in (0..n).step_by(NB) {
                    let jb = cmp::min(n - j, NB);
                    // Update diagonal block C(j,j)
                    syrk_kernel('U', 'T', jb, pb, alpha, a.add(p + j * lda), lda, 1.0, c.add(j + j * ldc), ldc);
                    // Update off-diagonal blocks C(i,j) for i < j
                    for i in (0..j).step_by(NB) {
                        let ib = cmp::min(j - i, NB);
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
        } else {
            // C += alpha * A*A'
            for p in (0..k).step_by(NB) {
                let pb = cmp::min(k - p, NB);
                for j in (0..n).step_by(NB) {
                    let jb = cmp::min(n - j, NB);
                    // Update diagonal block C(j,j)
                    syrk_kernel('U', 'N', jb, pb, alpha, a.add(j + p * lda), lda, 1.0, c.add(j + j * ldc), ldc);
                    // Update off-diagonal blocks C(i,j) for i < j
                    for i in (0..j).step_by(NB) {
                        let ib = cmp::min(j - i, NB);
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
        // UPLO = 'L'
        if trans == 'N' {
            // C += alpha * A*A'
            for p in (0..k).step_by(NB) {
                let pb = cmp::min(k - p, NB);
                for j in (0..n).step_by(NB) {
                    let jb = cmp::min(n - j, NB);
                    // Update diagonal block C(j,j)
                    syrk_kernel('L', 'N', jb, pb, alpha, a.add(j + p * lda), lda, 1.0, c.add(j + j * ldc), ldc);
                    // Update off-diagonal blocks C(i,j) for i > j
                    for i in (j + jb..n).step_by(NB) {
                        let ib = cmp::min(n - i, NB);
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
        } else {
            // C += alpha * A'*A
            for p in (0..k).step_by(NB) {
                let pb = cmp::min(k - p, NB);
                for j in (0..n).step_by(NB) {
                    let jb = cmp::min(n - j, NB);
                    // Update diagonal block C(j,j)
                    syrk_kernel('L', 'T', jb, pb, alpha, a.add(p + j * lda), lda, 1.0, c.add(j + j * ldc), ldc);
                    // Update off-diagonal blocks C(i,j) for i > j
                    for i in (j + jb..n).step_by(NB) {
                        let ib = cmp::min(n - i, NB);
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

/// A computational kernel for the symmetric rank-k update operation.
/// It calculates one of the following operations:
/// - `C := alpha * A * A^T + C`  (if `trans` = 'N')
/// - `C := alpha * A^T * A + C`  (if `trans` = 'T')
///
/// # Arguments
/// * `uplo`  - Specifies whether the upper ('U') or lower ('L') triangular part of `C` is updated.
/// * `trans` - Specifies the operation: 'N' for `A * A^T` or 'T' for `A^T * A`.
/// * `n`     - The order of the matrix `C` block.
/// * `k`     - The inner dimension of the multiplication.
/// * `alpha` - The scalar multiplier `alpha`.
/// * `a`     - A raw constant pointer to the first element of matrix `A`.
/// * `lda`   - The leading dimension of `A`.
/// * `_beta` - An unused parameter, assumed to be `1.0`. The name `_beta` indicates it is ignored.
/// * `c`     - A raw mutable pointer to the first element of the symmetric matrix `C` block.
/// * `ldc`   - The leading dimension of `C`.
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::too_many_arguments)]
unsafe fn syrk_kernel(
    uplo: char,
    trans: char,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    _beta: f64, // Assumed to be 1.0 for updates from the main syrk function
    c: *mut f64,
    ldc: usize,
) {
    if trans == 'N' {
        if uplo == 'U' {
            for j in 0..n {
                for l in 0..k {
                    let temp = alpha * *a.add(j + l * lda);
                    if temp != 0.0 {
                        daxpy_update(j + 1, temp, a.add(l * lda), 1, c.add(j * ldc), 1);
                    }
                }
            }
        } else {
            // Lower
            for j in 0..n {
                for l in 0..k {
                    let temp = alpha * *a.add(j + l * lda);
                    if temp != 0.0 {
                        daxpy_update(n - j, temp, a.add(j + l * lda), 1, c.add(j + j * ldc), 1);
                    }
                }
            }
        }
    } else if uplo == 'U' {
        for j in 0..n {
            for i in 0..=j {
                let temp = dot_product(k, a.add(i * lda), 1, a.add(j * lda), 1);
                *c.add(i + j * ldc) += alpha * temp;
            }
        }
    } else {
        // Lower
        for j in 0..n {
            for i in j..n {
                let temp = dot_product(k, a.add(i * lda), 1, a.add(j * lda), 1);
                *c.add(i + j * ldc) += alpha * temp;
            }
        }
    }
}
