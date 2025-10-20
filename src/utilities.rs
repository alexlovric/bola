#[cfg(target_arch = "x86_64")]
use std::{arch::x86_64::*, mem};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Computes the dot product of two vectors.
///
/// # Arguments
/// * `n`    - The number of elements in both vectors `x` and `y`.
/// * `x`    - A raw constant pointer to the first element of vector `x`.
/// * `incx` - The stride between consecutive elements of `x`.
/// * `y`    - A raw constant pointer to the first element of vector `y`.
/// * `incy` - The stride between consecutive elements of `y`.
///
/// # Returns
/// The dot product as an `f64` value.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product(n: usize, x: *const f64, incx: usize, y: *const f64, incy: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }

    if incx == 1 && incy == 1 {
        let n_chunks = n / 4;
        let mut sum_vec = _mm256_setzero_pd();
        for i in 0..n_chunks {
            let x_vec = _mm256_loadu_pd(x.add(i * 4));
            let y_vec = _mm256_loadu_pd(y.add(i * 4));
            sum_vec = _mm256_fmadd_pd(x_vec, y_vec, sum_vec);
        }

        // Horizontal sum of the vector register
        let mut temp_sum = 0.0;
        let temp_arr: [f64; 4] = mem::transmute(sum_vec);
        temp_sum += temp_arr.iter().sum::<f64>();

        // Handle remainder
        for i in (n_chunks * 4)..n {
            temp_sum += *x.add(i) * *y.add(i);
        }
        temp_sum
    } else {
        // Scalar fallback for non-unit strides
        let mut temp_sum = 0.0;
        for i in 0..n {
            temp_sum += *x.add(i * incx) * *y.add(i * incy);
        }
        temp_sum
    }
}

/// Computes the dot product of two vectors.
/// ... see above function
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
#[target_feature(enable = "neon")]
pub unsafe fn dot_product(n: usize, x: *const f64, incx: usize, y: *const f64, incy: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }

    if incx == 1 && incy == 1 {
        let n_chunks = n / 2;
        let mut sum_vec = vdupq_n_f64(0.0);
        for i in 0..n_chunks {
            let x_vec = vld1q_f64(x.add(i * 2));
            let y_vec = vld1q_f64(y.add(i * 2));
            sum_vec = vfmaq_f64(sum_vec, x_vec, y_vec);
        }

        // Horizontal sum of the vector register
        let mut temp_sum = vaddvq_f64(sum_vec);

        // Handle remainder
        for i in (n_chunks * 2)..n {
            temp_sum += *x.add(i) * *y.add(i);
        }
        temp_sum
    } else {
        // Scalar fallback for non-unit strides
        let mut temp_sum = 0.0;
        for i in 0..n {
            temp_sum += *x.add(i * incx) * *y.add(i * incy);
        }
        temp_sum
    }
}

/// Performs a SIMD-accelerated AXPY operation.
///
/// # Arguments
/// * `n`     - The number of elements in vectors `x` and `y`.
/// * `alpha` - The scalar multiplier `alpha`.
/// * `x`     - A raw constant pointer to the first element of vector `x`.
/// * `incx`  - The stride between consecutive elements of `x`.
/// * `y`     - A raw mutable pointer to the first element of vector `y`.
/// * `incy`  - The stride between consecutive elements of `y`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn daxpy(n: usize, alpha: f64, x: *const f64, incx: usize, y: *mut f64, incy: usize) {
    if n == 0 || alpha == 0.0 {
        return;
    }

    if incx == 1 && incy == 1 {
        let n_chunks = n / 4;
        let alpha_vec = _mm256_set1_pd(alpha);
        for i in 0..n_chunks {
            let x_ptr = x.add(i * 4);
            let y_ptr = y.add(i * 4);
            _mm256_storeu_pd(
                y_ptr,
                _mm256_fmadd_pd(alpha_vec, _mm256_loadu_pd(x_ptr), _mm256_loadu_pd(y_ptr)),
            );
        }
        for i in (n_chunks * 4)..n {
            *y.add(i) += alpha * (*x.add(i));
        }
    } else {
        // Scalar fallback for non-unit increments
        for i in 0..n {
            *y.add(i * incy) += alpha * (*x.add(i * incx));
        }
    }
}

/// Performs a SIMD-accelerated AXPY operation.
/// ... see above function
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
#[target_feature(enable = "neon")]
pub unsafe fn daxpy(n: usize, alpha: f64, x: *const f64, incx: usize, y: *mut f64, incy: usize) {
    if n == 0 || alpha == 0.0 {
        return;
    }

    if incx == 1 && incy == 1 {
        let n_chunks = n / 2;
        let alpha_vec = vdupq_n_f64(alpha);
        for i in 0..n_chunks {
            let x_ptr = x.add(i * 2);
            let y_ptr = y.add(i * 2);
            let x_vec = vld1q_f64(x_ptr);
            let y_vec = vld1q_f64(y_ptr);
            vst1q_f64(y_ptr, vfmaq_f64(y_vec, alpha_vec, x_vec));
        }
        for i in (n_chunks * 2)..n {
            *y.add(i) += alpha * (*x.add(i));
        }
    } else {
        // Scalar fallback for non-unit increments
        for i in 0..n {
            *y.add(i * incy) += alpha * (*x.add(i * incx));
        }
    }
}

pub fn assert_approx_eq(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "Slices have different lengths");
    for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (val_a - val_b).abs() < tol,
            "Mismatch at index {}: a[{}] = {}, b[{}] = {}",
            i,
            i,
            val_a,
            i,
            val_b
        );
    }
}
