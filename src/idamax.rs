#[cfg(feature = "profiling")]
use crate::profiling;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Finds the index of the element with the largest absolute value in a vector.
///
/// # Arguments
/// * `n` - The number of elements in the vector `x`.
/// * `x` - A raw constant pointer to the first element of the vector.
///
/// # Returns
/// A `usize` representing the 0-based index of the first element with the
/// maximum absolute value. If `n` is 0, this function will return 0.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn idamax(n: usize, x: *const f64) -> usize {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("IDAMAX");

    if n < 16 {
        let mut max_val = -1.0;
        let mut max_idx = 0;
        for i in 0..n {
            let val = (*x.add(i)).abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        return max_idx;
    }

    let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFFFFFFFFFFFFFF));

    let mut max_vals_vec0 = _mm256_setzero_pd();
    let mut max_idxs_vec0 = _mm256_setzero_pd();
    let mut max_vals_vec1 = _mm256_setzero_pd();
    let mut max_idxs_vec1 = _mm256_setzero_pd();
    let mut max_vals_vec2 = _mm256_setzero_pd();
    let mut max_idxs_vec2 = _mm256_setzero_pd();
    let mut max_vals_vec3 = _mm256_setzero_pd();
    let mut max_idxs_vec3 = _mm256_setzero_pd();

    let mut idxs0 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
    let mut idxs1 = _mm256_set_pd(7.0, 6.0, 5.0, 4.0);
    let mut idxs2 = _mm256_set_pd(11.0, 10.0, 9.0, 8.0);
    let mut idxs3 = _mm256_set_pd(15.0, 14.0, 13.0, 12.0);
    let step_vec = _mm256_set1_pd(16.0);

    let n_chunks = n / 16;
    let mut x_ptr = x;
    for _ in 0..n_chunks {
        let vals0 = _mm256_loadu_pd(x_ptr);
        let abs_vals0 = _mm256_and_pd(vals0, abs_mask);
        let mask0 = _mm256_cmp_pd(abs_vals0, max_vals_vec0, _CMP_GT_OQ);
        max_vals_vec0 = _mm256_blendv_pd(max_vals_vec0, abs_vals0, mask0);
        max_idxs_vec0 = _mm256_blendv_pd(max_idxs_vec0, idxs0, mask0);

        let vals1 = _mm256_loadu_pd(x_ptr.add(4));
        let abs_vals1 = _mm256_and_pd(vals1, abs_mask);
        let mask1 = _mm256_cmp_pd(abs_vals1, max_vals_vec1, _CMP_GT_OQ);
        max_vals_vec1 = _mm256_blendv_pd(max_vals_vec1, abs_vals1, mask1);
        max_idxs_vec1 = _mm256_blendv_pd(max_idxs_vec1, idxs1, mask1);

        let vals2 = _mm256_loadu_pd(x_ptr.add(8));
        let abs_vals2 = _mm256_and_pd(vals2, abs_mask);
        let mask2 = _mm256_cmp_pd(abs_vals2, max_vals_vec2, _CMP_GT_OQ);
        max_vals_vec2 = _mm256_blendv_pd(max_vals_vec2, abs_vals2, mask2);
        max_idxs_vec2 = _mm256_blendv_pd(max_idxs_vec2, idxs2, mask2);

        let vals3 = _mm256_loadu_pd(x_ptr.add(12));
        let abs_vals3 = _mm256_and_pd(vals3, abs_mask);
        let mask3 = _mm256_cmp_pd(abs_vals3, max_vals_vec3, _CMP_GT_OQ);
        max_vals_vec3 = _mm256_blendv_pd(max_vals_vec3, abs_vals3, mask3);
        max_idxs_vec3 = _mm256_blendv_pd(max_idxs_vec3, idxs3, mask3);

        x_ptr = x_ptr.add(16);
        idxs0 = _mm256_add_pd(idxs0, step_vec);
        idxs1 = _mm256_add_pd(idxs1, step_vec);
        idxs2 = _mm256_add_pd(idxs2, step_vec);
        idxs3 = _mm256_add_pd(idxs3, step_vec);
    }

    let mask1 = _mm256_cmp_pd(max_vals_vec1, max_vals_vec0, _CMP_GT_OQ);
    max_vals_vec0 = _mm256_blendv_pd(max_vals_vec0, max_vals_vec1, mask1);
    max_idxs_vec0 = _mm256_blendv_pd(max_idxs_vec0, max_idxs_vec1, mask1);

    let mask2 = _mm256_cmp_pd(max_vals_vec2, max_vals_vec0, _CMP_GT_OQ);
    max_vals_vec0 = _mm256_blendv_pd(max_vals_vec0, max_vals_vec2, mask2);
    max_idxs_vec0 = _mm256_blendv_pd(max_idxs_vec0, max_idxs_vec2, mask2);

    let mask3 = _mm256_cmp_pd(max_vals_vec3, max_vals_vec0, _CMP_GT_OQ);
    max_vals_vec0 = _mm256_blendv_pd(max_vals_vec0, max_vals_vec3, mask3);
    max_idxs_vec0 = _mm256_blendv_pd(max_idxs_vec0, max_idxs_vec3, mask3);

    let mut max_vals_arr: [f64; 4] = [0.0; 4];
    let mut max_idxs_arr: [f64; 4] = [0.0; 4];
    _mm256_storeu_pd(max_vals_arr.as_mut_ptr(), max_vals_vec0);
    _mm256_storeu_pd(max_idxs_arr.as_mut_ptr(), max_idxs_vec0);

    let mut max_val = max_vals_arr[0];
    let mut max_idx = max_idxs_arr[0] as usize;

    if max_vals_arr[1] > max_val {
        max_val = max_vals_arr[1];
        max_idx = max_idxs_arr[1] as usize;
    }
    if max_vals_arr[2] > max_val {
        max_val = max_vals_arr[2];
        max_idx = max_idxs_arr[2] as usize;
    }
    if max_vals_arr[3] > max_val {
        max_val = max_vals_arr[3];
        max_idx = max_idxs_arr[3] as usize;
    }

    for i in (n_chunks * 16)..n {
        let val = (*x.add(i)).abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx
}

/// Finds the index of the element with the largest absolute value in a vector.
/// ... see above function
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
#[target_feature(enable = "neon,fma")]
pub unsafe fn idamax(n: usize, x: *const f64) -> usize {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("IDAMAX");

    if n < 8 {
        let mut max_val = -1.0;
        let mut max_idx = 0;
        for i in 0..n {
            let val = (*x.add(i)).abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        return max_idx;
    }

    let mut max_vals_vec0 = vdupq_n_f64(0.0);
    let mut max_idxs_vec0 = vdupq_n_f64(0.0);
    let mut max_vals_vec1 = vdupq_n_f64(0.0);
    let mut max_idxs_vec1 = vdupq_n_f64(0.0);
    let mut max_vals_vec2 = vdupq_n_f64(0.0);
    let mut max_idxs_vec2 = vdupq_n_f64(0.0);
    let mut max_vals_vec3 = vdupq_n_f64(0.0);
    let mut max_idxs_vec3 = vdupq_n_f64(0.0);

    let mut idxs0 = vld1q_f64([0.0, 1.0].as_ptr());
    let mut idxs1 = vld1q_f64([2.0, 3.0].as_ptr());
    let mut idxs2 = vld1q_f64([4.0, 5.0].as_ptr());
    let mut idxs3 = vld1q_f64([6.0, 7.0].as_ptr());
    let step_vec = vdupq_n_f64(8.0);

    let n_chunks = n / 8;
    let mut x_ptr = x;
    for _ in 0..n_chunks {
        let vals0 = vld1q_f64(x_ptr);
        let abs_vals0 = vabsq_f64(vals0);
        let mask0 = vcgtq_f64(abs_vals0, max_vals_vec0);
        max_vals_vec0 = vbslq_f64(mask0, abs_vals0, max_vals_vec0);
        max_idxs_vec0 = vbslq_f64(mask0, idxs0, max_idxs_vec0);

        let vals1 = vld1q_f64(x_ptr.add(2));
        let abs_vals1 = vabsq_f64(vals1);
        let mask1 = vcgtq_f64(abs_vals1, max_vals_vec1);
        max_vals_vec1 = vbslq_f64(mask1, abs_vals1, max_vals_vec1);
        max_idxs_vec1 = vbslq_f64(mask1, idxs1, max_idxs_vec1);

        let vals2 = vld1q_f64(x_ptr.add(4));
        let abs_vals2 = vabsq_f64(vals2);
        let mask2 = vcgtq_f64(abs_vals2, max_vals_vec2);
        max_vals_vec2 = vbslq_f64(mask2, abs_vals2, max_vals_vec2);
        max_idxs_vec2 = vbslq_f64(mask2, idxs2, max_idxs_vec2);

        let vals3 = vld1q_f64(x_ptr.add(6));
        let abs_vals3 = vabsq_f64(vals3);
        let mask3 = vcgtq_f64(abs_vals3, max_vals_vec3);
        max_vals_vec3 = vbslq_f64(mask3, abs_vals3, max_vals_vec3);
        max_idxs_vec3 = vbslq_f64(mask3, idxs3, max_idxs_vec3);

        x_ptr = x_ptr.add(8);
        idxs0 = vaddq_f64(idxs0, step_vec);
        idxs1 = vaddq_f64(idxs1, step_vec);
        idxs2 = vaddq_f64(idxs2, step_vec);
        idxs3 = vaddq_f64(idxs3, step_vec);
    }

    let mask1 = vcgtq_f64(max_vals_vec1, max_vals_vec0);
    max_vals_vec0 = vbslq_f64(mask1, max_vals_vec1, max_vals_vec0);
    max_idxs_vec0 = vbslq_f64(mask1, max_idxs_vec1, max_idxs_vec0);

    let mask2 = vcgtq_f64(max_vals_vec2, max_vals_vec0);
    max_vals_vec0 = vbslq_f64(mask2, max_vals_vec2, max_vals_vec0);
    max_idxs_vec0 = vbslq_f64(mask2, max_idxs_vec2, max_idxs_vec0);

    let mask3 = vcgtq_f64(max_vals_vec3, max_vals_vec0);
    max_vals_vec0 = vbslq_f64(mask3, max_vals_vec3, max_vals_vec0);
    max_idxs_vec0 = vbslq_f64(mask3, max_idxs_vec3, max_idxs_vec0);

    let mut max_vals_arr: [f64; 2] = [0.0; 2];
    let mut max_idxs_arr: [f64; 2] = [0.0; 2];
    vst1q_f64(max_vals_arr.as_mut_ptr(), max_vals_vec0);
    vst1q_f64(max_idxs_arr.as_mut_ptr(), max_idxs_vec0);

    let mut max_val = max_vals_arr[0];
    let mut max_idx = max_idxs_arr[0] as usize;
    if max_vals_arr[1] > max_val {
        max_val = max_vals_arr[1];
        max_idx = max_idxs_arr[1] as usize;
    }

    for i in (n_chunks * 8)..n {
        let val = (*x.add(i)).abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx
}
