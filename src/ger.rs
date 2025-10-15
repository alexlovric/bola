use std::arch::x86_64::*;

#[cfg(feature = "profiling")]
use crate::profiling;

/// Performs a general rank-1 update operation on a matrix.
///
/// $A := \alpha \cdot x \cdot y^T + A$
////// Where `A` is an M-by-N matrix, `x` is a vector of length `m`, `y` is a vector
/// of length `n`, and `alpha` is a scalar. This operation adds the outer product
/// of vectors `x` and `y` (scaled by `alpha`) to the matrix `A`.
///
/// # Arguments
/// * `m`     - The number of rows of the matrix `A`, and the length of vector `x`.
/// * `n`     - The number of columns of the matrix `A`, and the number of elements in vector `y`.
/// * `alpha` - The scalar multiplier `alpha`.
/// * `x`     - A raw constant pointer to the first element of the vector `x`. Assumed to have a stride of 1.
/// * `y`     - A raw constant pointer to the first element of the vector `y`.
/// * `ldy`   - The stride between consecutive elements of the vector `y`.
/// * `a`     - A raw mutable pointer to the first element of the matrix `A` (in column-major order).
/// * `lda`   - The leading dimension of the matrix `A`, i.e., the stride between consecutive columns.
#[inline]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ger(m: usize, n: usize, alpha: f64, x: *const f64, y: *const f64, ldy: usize, a: *mut f64, lda: usize) {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("GER");

    let n_unrolled = n - (n % 2);
    for j in (0..n_unrolled).step_by(2) {
        let temp_scalar0 = alpha * *y.add(j * ldy);
        let temp_scalar1 = alpha * *y.add((j + 1) * ldy);

        if temp_scalar0 == 0.0 && temp_scalar1 == 0.0 {
            continue;
        }

        let temp_vec0 = _mm256_set1_pd(temp_scalar0);
        let temp_vec1 = _mm256_set1_pd(temp_scalar1);

        let mut a_ptr0 = a.add(j * lda);
        let mut a_ptr1 = a.add((j + 1) * lda);
        let mut x_ptr = x;

        let m_chunks = m / 4;
        for _ in 0..m_chunks {
            let x_vec = _mm256_loadu_pd(x_ptr);

            let a_vec0 = _mm256_loadu_pd(a_ptr0);
            _mm256_storeu_pd(a_ptr0, _mm256_fmadd_pd(x_vec, temp_vec0, a_vec0));

            let a_vec1 = _mm256_loadu_pd(a_ptr1);
            _mm256_storeu_pd(a_ptr1, _mm256_fmadd_pd(x_vec, temp_vec1, a_vec1));

            x_ptr = x_ptr.add(4);
            a_ptr0 = a_ptr0.add(4);
            a_ptr1 = a_ptr1.add(4);
        }

        for i in (m_chunks * 4)..m {
            *a.add(i + j * lda) += temp_scalar0 * (*x.add(i));
            *a.add(i + (j + 1) * lda) += temp_scalar1 * (*x.add(i));
        }
    }

    if !n.is_multiple_of(2) {
        let j = n - 1;
        let temp_scalar = alpha * *y.add(j * ldy);
        if temp_scalar != 0.0 {
            for i in 0..m {
                *a.add(i + j * lda) += temp_scalar * (*x.add(i));
            }
        }
    }
}
