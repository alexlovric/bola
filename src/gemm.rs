use rayon::prelude::*;
use std::cell::Cell;
use std::cmp;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(feature = "profiling")]
use crate::profiling;

// Cache and Kernel Parameters
const KC: usize = 64;
const MC: usize = 32;
const NC: usize = 16;

// Micro-kernel dimensions.
const MR: usize = 8;
const NR: usize = 4;

const PARALLEL_THRESHOLD: usize = 128;

// Thread-Local Buffers
thread_local! {
    static PACKED_A: Cell<Vec<f64>> = Cell::new(Vec::with_capacity(MC * KC));
    static PACKED_B: Cell<Vec<f64>> = Cell::new(Vec::with_capacity(KC * NC));
}

/// Packs a panel of a matrix into a contiguous buffer, handling transposition.
///
/// This function copies a panel from the source matrix `src` into the destination
/// buffer `dst` in a column-major format. This is a crucial step for performance,
/// as it ensures that the data accessed by the computational micro-kernel is
/// contiguous in memory, leading to better cache utilisation.
///
/// # Arguments
/// * `trans`  - A character indicating the transposition of the source panel:
///   - 'N' or 'n': The panel is not transposed.
///   - 'T' or 't': The panel is transposed.
/// * `rows`   - The number of rows in the panel to be packed.
/// * `cols`   - The number of columns in the panel to be packed.
/// * `src`    - A pointer to the top-left element of the source panel.
/// * `ld_src` - The leading dimension (stride between columns) of the source matrix.
/// * `dst`    - A pointer to the destination buffer where the panel will be stored.
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
pub unsafe fn pack(trans: char, rows: usize, cols: usize, src: *const f64, ld_src: usize, dst: *mut f64) {
    if trans == 'N' {
        // Pack the panel without transposing (column-major to column-major).
        for j in 0..cols {
            for i in 0..rows {
                *dst.add(i + j * rows) = *src.add(i + j * ld_src);
            }
        }
    } else {
        // Pack the panel with transposition (row-major access on column-major source).
        for j in 0..cols {
            for i in 0..rows {
                *dst.add(i + j * rows) = *src.add(j + i * ld_src);
            }
        }
    }
}

/// Computes the matrix-matrix product C = alpha * op(A) * op(B) + beta * C.
///
/// This function implements the General Matrix-Matrix Multiplication (GEMM) operation.
/// It is highly optimised using a block-based approach, SIMD instructions,
/// and optional multi-threading via Rayon.
///
/// # Arguments
/// * `transa` - Specifies the operation on matrix A: 'N' for no transpose, 'T' for transpose.
/// * `transb` - Specifies the operation on matrix B: 'N' for no transpose, 'T' for transpose.
/// * `m`      - The number of rows of matrix `op(A)` and matrix C.
/// * `n`      - The number of columns of matrix `op(B)` and matrix C.
/// * `k`      - The number of columns of `op(A)` and rows of `op(B)`.
/// * `alpha`  - The scalar multiplier for the product of `op(A)` and `op(B)`.
/// * `a`      - A pointer to the first element of matrix A.
/// * `lda`    - The leading dimension of matrix A.
/// * `b`      - A pointer to the first element of matrix B.
/// * `ldb`    - The leading dimension of matrix B.
/// * `beta`   - The scalar multiplier for matrix C.
/// * `c`      - A pointer to the first element of matrix C.
/// * `ldc`    - The leading dimension of matrix C.
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm(
    transa: char,
    transb: char,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("GEMM");

    if m == 0 || n == 0 {
        return;
    }

    if beta == 0.0 {
        for j in 0..n {
            for i in 0..m {
                *c.add(i + j * ldc) = 0.0;
            }
        }
    } else if beta != 1.0 {
        for j in 0..n {
            for i in 0..m {
                *c.add(i + j * ldc) *= beta;
            }
        }
    }

    if alpha == 0.0 {
        return;
    }

    let notransa = transa.eq_ignore_ascii_case(&'N');
    let notransb = transb.eq_ignore_ascii_case(&'N');

    // Cast to pass
    let a_addr = a as usize;
    let b_addr = b as usize;
    let c_addr = c as usize;

    // The core logic for processing blocks.
    let core_logic = |jc: usize| {
        let nb = cmp::min(n - jc, NC);
        let a = a_addr as *const f64;
        let b = b_addr as *const f64;
        let c = c_addr as *mut f64;

        PACKED_A.with(|a_cell| {
            PACKED_B.with(|b_cell| {
                let mut a_packed = a_cell.take();
                let mut b_packed = b_cell.take();

                for pc in (0..k).step_by(KC) {
                    let pb = cmp::min(k - pc, KC);

                    let b_panel_ptr = if notransb {
                        b.add(pc + jc * ldb)
                    } else {
                        b.add(jc + pc * ldb)
                    };

                    // {
                    //     #[cfg(feature = "profiling")]
                    //     let _timer = profiling::ScopedTimer::new("Pack B");
                    b_packed.resize(pb * nb, 0.0);
                    pack(transb, pb, nb, b_panel_ptr, ldb, b_packed.as_mut_ptr());
                    // }

                    for ic in (0..m).step_by(MC) {
                        let mb = cmp::min(m - ic, MC);

                        let a_panel_ptr = if notransa {
                            a.add(ic + pc * lda)
                        } else {
                            a.add(pc + ic * lda)
                        };

                        // {
                        //     #[cfg(feature = "profiling")]
                        //     let _timer = profiling::ScopedTimer::new("Pack A");
                        a_packed.resize(mb * pb, 0.0);
                        pack(transa, mb, pb, a_panel_ptr, lda, a_packed.as_mut_ptr());
                        // }

                        // {
                        //     #[cfg(feature = "profiling")]
                        //     let _timer = profiling::ScopedTimer::new("GEMM Kern");

                        for j in (0..nb).step_by(NR) {
                            for i in (0..mb).step_by(MR) {
                                let c_block_ptr = c.add(ic + i + (jc + j) * ldc);
                                let b_pack_ptr = b_packed.as_ptr().add(j * pb);

                                if i + MR > mb || j + NR > nb {
                                    add_scalar_kernel(
                                        cmp::min(mb - i, MR),
                                        cmp::min(nb - j, NR),
                                        pb,
                                        alpha,
                                        a_packed.as_ptr().add(i),
                                        mb,
                                        b_pack_ptr,
                                        pb,
                                        c_block_ptr,
                                        ldc,
                                    );
                                } else {
                                    add_8x4_kernel(pb, alpha, a_packed.as_ptr().add(i), mb, b_pack_ptr, pb, c_block_ptr, ldc);
                                }
                            }
                        }
                        // }
                    }
                }
                a_cell.set(a_packed);
                b_cell.set(b_packed);
            });
        });
    };

    // Conditional Parallelism
    if n > PARALLEL_THRESHOLD {
        (0..n).into_par_iter().step_by(NC).for_each(core_logic);
    } else {
        for jc in (0..n).step_by(NC) {
            core_logic(jc);
        }
    }
}

/// Performs an 8x4 micro-kernel matrix multiplication using AVX2 and FMA instructions.
///
/// This function computes `C += alpha * A * B` for a small 8x4 sub-block of the
/// matrix C, where A is an 8xK panel and B is a Kx4 panel. It operates on packed
/// data for maximum efficiency.
///
/// # Arguments
/// * `k`     - The inner dimension of the multiplication (columns of A, rows of B).
/// * `alpha` - The scalar multiplier.
/// * `a`     - Pointer to the packed 8xK panel of matrix A.
/// * `lda`   - The leading dimension of the packed A panel (which is `MR = 8`).
/// * `b`     - Pointer to the packed Kx4 panel of matrix B.
/// * `ldb`   - The leading dimension of the packed B panel (which is `K`).
/// * `c`     - Pointer to the top-left element of the 8x4 block in the C matrix.
/// * `ldc`   - The leading dimension of the C matrix.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn add_8x4_kernel(
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    c: *mut f64,
    ldc: usize,
) {
    let mut c0 = [_mm256_setzero_pd(); 4];
    let mut c1 = [_mm256_setzero_pd(); 4];

    for l in 0..k {
        let a0_vec = _mm256_loadu_pd(a.add(l * lda));
        let a1_vec = _mm256_loadu_pd(a.add(l * lda + 4));

        let b0 = _mm256_set1_pd(*b.add(l));
        let b1 = _mm256_set1_pd(*b.add(l + ldb));
        let b2 = _mm256_set1_pd(*b.add(l + 2 * ldb));
        let b3 = _mm256_set1_pd(*b.add(l + 3 * ldb));

        c0[0] = _mm256_fmadd_pd(a0_vec, b0, c0[0]);
        c1[0] = _mm256_fmadd_pd(a1_vec, b0, c1[0]);
        c0[1] = _mm256_fmadd_pd(a0_vec, b1, c0[1]);
        c1[1] = _mm256_fmadd_pd(a1_vec, b1, c1[1]);
        c0[2] = _mm256_fmadd_pd(a0_vec, b2, c0[2]);
        c1[2] = _mm256_fmadd_pd(a1_vec, b2, c1[2]);
        c0[3] = _mm256_fmadd_pd(a0_vec, b3, c0[3]);
        c1[3] = _mm256_fmadd_pd(a1_vec, b3, c1[3]);
    }

    let alpha_vec = _mm256_set1_pd(alpha);
    for j in 0..NR {
        let c_ptr = c.add(j * ldc);
        _mm256_storeu_pd(c_ptr, _mm256_fmadd_pd(alpha_vec, c0[j], _mm256_loadu_pd(c_ptr)));
        _mm256_storeu_pd(c_ptr.add(4), _mm256_fmadd_pd(alpha_vec, c1[j], _mm256_loadu_pd(c_ptr.add(4))));
    }
}

/// Performs an 8x4 micro-kernel matrix multiplication using AVX2 and FMA instructions.
/// ... see above function
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::too_many_arguments)]
#[target_feature(enable = "neon,fma")]
pub unsafe fn add_8x4_kernel(
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    c: *mut f64,
    ldc: usize,
) {
    let mut c0 = [vdupq_n_f64(0.0); 4];
    let mut c1 = [vdupq_n_f64(0.0); 4];
    let mut c2 = [vdupq_n_f64(0.0); 4];
    let mut c3 = [vdupq_n_f64(0.0); 4];

    for l in 0..k {
        let a_ptr = a.add(l * lda);
        let a0_vec = vld1q_f64(a_ptr);
        let a1_vec = vld1q_f64(a_ptr.add(2));
        let a2_vec = vld1q_f64(a_ptr.add(4));
        let a3_vec = vld1q_f64(a_ptr.add(6));

        let b0 = vdupq_n_f64(*b.add(l));
        let b1 = vdupq_n_f64(*b.add(l + ldb));
        let b2 = vdupq_n_f64(*b.add(l + 2 * ldb));
        let b3 = vdupq_n_f64(*b.add(l + 3 * ldb));

        c0[0] = vfmaq_f64(c0[0], a0_vec, b0);
        c1[0] = vfmaq_f64(c1[0], a1_vec, b0);
        c2[0] = vfmaq_f64(c2[0], a2_vec, b0);
        c3[0] = vfmaq_f64(c3[0], a3_vec, b0);

        c0[1] = vfmaq_f64(c0[1], a0_vec, b1);
        c1[1] = vfmaq_f64(c1[1], a1_vec, b1);
        c2[1] = vfmaq_f64(c2[1], a2_vec, b1);
        c3[1] = vfmaq_f64(c3[1], a3_vec, b1);

        c0[2] = vfmaq_f64(c0[2], a0_vec, b2);
        c1[2] = vfmaq_f64(c1[2], a1_vec, b2);
        c2[2] = vfmaq_f64(c2[2], a2_vec, b2);
        c3[2] = vfmaq_f64(c3[2], a3_vec, b2);

        c0[3] = vfmaq_f64(c0[3], a0_vec, b3);
        c1[3] = vfmaq_f64(c1[3], a1_vec, b3);
        c2[3] = vfmaq_f64(c2[3], a2_vec, b3);
        c3[3] = vfmaq_f64(c3[3], a3_vec, b3);
    }

    let alpha_vec = vdupq_n_f64(alpha);

    for j in 0..NR {
        let c_ptr = c.add(j * ldc);

        vst1q_f64(c_ptr, vfmaq_f64(vld1q_f64(c_ptr), alpha_vec, c0[j]));
        vst1q_f64(c_ptr.add(2), vfmaq_f64(vld1q_f64(c_ptr.add(2)), alpha_vec, c1[j]));
        vst1q_f64(c_ptr.add(4), vfmaq_f64(vld1q_f64(c_ptr.add(4)), alpha_vec, c2[j]));
        vst1q_f64(c_ptr.add(6), vfmaq_f64(vld1q_f64(c_ptr.add(6)), alpha_vec, c3[j]));
    }
}

/// A scalar micro-kernel for handling matrix multiplication on edge cases.
///
/// This function provides a simple, non-vectorised implementation of matrix
/// multiplication for sub-blocks that are smaller than the main 8x4 kernel size.
/// It operates on the same packed buffers as the optimised kernel.
///
/// # Arguments
/// * `m`     - The number of rows in the A sub-panel (must be <= 8).
/// * `n`     - The number of columns in the B sub-panel (must be <= 4).
/// * `k`     - The inner dimension (columns of A, rows of B).
/// * `alpha` - The scalar multiplier.
/// * `a`     - Pointer to the packed M x K panel of matrix A.
/// * `lda`   - The leading dimension of the packed A panel.
/// * `b`     - Pointer to the packed K x N panel of matrix B.
/// * `ldb`   - The leading dimension of the packed B panel.
/// * `c`     - Pointer to the top-left element of the M x N block in the C matrix.
/// * `ldc`   - The leading dimension of the C matrix.
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn add_scalar_kernel(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    c: *mut f64,
    ldc: usize,
) {
    for j in 0..n {
        for l in 0..k {
            let temp = alpha * (*b.add(l + j * ldb));
            for i in 0..m {
                *c.add(i + j * ldc) += temp * (*a.add(i + l * lda));
            }
        }
    }
}
