#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
#[target_feature(enable = "avx2")]
pub unsafe fn scal_kernel(m: usize, inv_diag: f64, col: *mut f64) -> usize {
    let inv_diag_vec = _mm256_set1_pd(inv_diag);
    let m_chunks = m / 4;
    let mut col_ptr = col;

    for _ in 0..m_chunks {
        let col_vec = _mm256_loadu_pd(col_ptr);
        let res_vec = _mm256_mul_pd(col_vec, inv_diag_vec);
        _mm256_storeu_pd(col_ptr, res_vec);
        col_ptr = col_ptr.add(4);
    }
    m_chunks * 4
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
#[target_feature(enable = "neon,fma")]
pub unsafe fn scal_kernel(m: usize, inv_diag: f64, col: *mut f64) -> usize {
    let inv_diag_vec = vdupq_n_f64(inv_diag);
    let m_chunks = m / 2;
    let mut col_ptr = col;

    for _ in 0..m_chunks {
        let col_vec = vld1q_f64(col_ptr);
        let res_vec = vmulq_f64(col_vec, inv_diag_vec);
        vst1q_f64(col_ptr, res_vec);
        col_ptr = col_ptr.add(2);
    }
    m_chunks * 2
}
