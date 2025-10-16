use rayon::prelude::*;
use std::{ptr, slice};

#[cfg(feature = "profiling")]
use crate::profiling;

const PARALLEL_THRESHOLD: usize = 256;

/// Performs a series of row interchanges on a matrix.
///
/// # Arguments
/// * `n`    - The number of columns of the matrix `A` to which the interchanges will be applied.
/// * `a`    - A mutable slice representing the matrix `A` in column-major order.
/// * `lda`  - The leading dimension of `A`, i.e., the stride between consecutive columns.
/// * `k1`   - The first element of `ipiv` to be used for interchanges (0-based).
/// * `k2`   - The last element of `ipiv` to be used for interchanges (0-based).
/// * `ipiv` - The slice containing the pivot indices. For `i` in `k1..=k2`, row `i` is
///   swapped with row `ipiv[i]`.
/// * `incx` - The increment between successive values of `ipiv`. If `incx` is positive,
///   the loop is forward; if negative, it's backward. If zero, no swaps are performed.
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
pub unsafe fn laswp(n: usize, a: &mut [f64], lda: usize, k1: i32, k2: i32, ipiv: &[i32], incx: i32) {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("LASWP");

    if n == 0 || incx == 0 || k1 > k2 {
        return;
    }

    let (ix0, i1, i2, i_inc) = if incx > 0 {
        ((k1 - 1) as usize, k1, k2, 1)
    } else {
        let ix_start_0based = (k1 - 1) as usize;
        let num_pivots = (k2 - k1) as usize;
        let ix0_final = ix_start_0based + num_pivots * incx.unsigned_abs() as usize;
        (ix0_final, k2, k1, -1)
    };

    if n < PARALLEL_THRESHOLD {
        let a_ptr = a.as_mut_ptr();
        let n_unrolled = n - (n % 4);

        for k_base in (0..n_unrolled).step_by(4) {
            let col0_ptr = a_ptr.add(k_base * lda);
            let col1_ptr = a_ptr.add((k_base + 1) * lda);
            let col2_ptr = a_ptr.add((k_base + 2) * lda);
            let col3_ptr = a_ptr.add((k_base + 3) * lda);

            let mut ix = ix0;
            let mut i = i1;
            while if i_inc > 0 { i <= i2 } else { i >= i2 } {
                let ip = ipiv[ix] as usize;
                if ip != i as usize {
                    let row_i_m1 = i as usize - 1;
                    let row_ip_m1 = ip - 1;
                    ptr::swap(col0_ptr.add(row_i_m1), col0_ptr.add(row_ip_m1));
                    ptr::swap(col1_ptr.add(row_i_m1), col1_ptr.add(row_ip_m1));
                    ptr::swap(col2_ptr.add(row_i_m1), col2_ptr.add(row_ip_m1));
                    ptr::swap(col3_ptr.add(row_i_m1), col3_ptr.add(row_ip_m1));
                }
                ix = (ix as i32 + incx) as usize;
                i += i_inc;
            }
        }

        for k in n_unrolled..n {
            let col_ptr = a_ptr.add(k * lda);
            let mut ix = ix0;
            let mut i = i1;
            while if i_inc > 0 { i <= i2 } else { i >= i2 } {
                let ip = ipiv[ix] as usize;
                if ip != i as usize {
                    ptr::swap(col_ptr.add(i as usize - 1), col_ptr.add(ip - 1));
                }
                ix = (ix as i32 + incx) as usize;
                i += i_inc;
            }
        }
        return;
    }

    let a_addr = a.as_mut_ptr() as usize;
    let ipiv_addr = ipiv.as_ptr() as usize;
    let n32 = (n / 32) * 32;

    if n32 != 0 {
        (0..n32).into_par_iter().step_by(32).for_each(|j_chunk_start| {
            let a_ptr = a_addr as *mut f64;
            let ipiv = slice::from_raw_parts(ipiv_addr as *const i32, ipiv.len());
            for k_base in (j_chunk_start..j_chunk_start + 32).step_by(4) {
                let col0_ptr = a_ptr.add(k_base * lda);
                let col1_ptr = a_ptr.add((k_base + 1) * lda);
                let col2_ptr = a_ptr.add((k_base + 2) * lda);
                let col3_ptr = a_ptr.add((k_base + 3) * lda);
                let mut ix = ix0;
                let mut i = i1;
                while if i_inc > 0 { i <= i2 } else { i >= i2 } {
                    let ip = ipiv[ix] as usize;
                    if ip != i as usize {
                        let row_i_m1 = i as usize - 1;
                        let row_ip_m1 = ip - 1;
                        ptr::swap(col0_ptr.add(row_i_m1), col0_ptr.add(row_ip_m1));
                        ptr::swap(col1_ptr.add(row_i_m1), col1_ptr.add(row_ip_m1));
                        ptr::swap(col2_ptr.add(row_i_m1), col2_ptr.add(row_ip_m1));
                        ptr::swap(col3_ptr.add(row_i_m1), col3_ptr.add(row_ip_m1));
                    }
                    ix = (ix as i32 + incx) as usize;
                    i += i_inc;
                }
            }
        });
    }

    if n32 != n {
        let a_ptr = a.as_mut_ptr();
        let mut ix = ix0;
        let mut i = i1;
        while if i_inc > 0 { i <= i2 } else { i >= i2 } {
            let ip = ipiv[ix] as usize;
            if ip != i as usize {
                let row1_base_ptr = a_ptr.add(i as usize - 1);
                let row2_base_ptr = a_ptr.add(ip - 1);
                for k in n32..n {
                    let offset = k * lda;
                    ptr::swap(row1_base_ptr.add(offset), row2_base_ptr.add(offset));
                }
            }
            ix = (ix as i32 + incx) as usize;
            i += i_inc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laswp_row_swaps() {
        let n_cols_to_swap = 3;
        let lda = 3;
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let ipiv = vec![3, 2];

        let a_expected = vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0, 9.0, 8.0, 7.0];

        unsafe {
            laswp(n_cols_to_swap, &mut a, lda, 1, 2, &ipiv, 1);
        }

        assert_eq!(a.len(), a_expected.len(), "Slices have different lengths");
        for (i, (val_a, val_b)) in a.iter().zip(a_expected.iter()).enumerate() {
            assert!(
                (val_a - val_b).abs() < 1e-8,
                "Mismatch at index {}: evaluated[{}] = {}, expected[{}] = {}",
                i,
                i,
                val_a,
                i,
                val_b
            );
        }
    }
}
