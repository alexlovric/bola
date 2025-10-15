use rayon::prelude::*;

use crate::{gemm::gemm, utilities::daxpy_update};

#[cfg(feature = "profiling")]
use crate::profiling;

const TRSM_RECURSION_STOP: usize = 64; // Base case size
const PARALLEL_THRESHOLD: usize = 32;

/// Solves a triangular matrix equation with multiple right-hand sides.
/// - `op(A) * X = alpha * B`  (if `side` = 'L')
/// - `X * op(A) = alpha * B`  (if `side` = 'R')
///
/// Where `X` and `B` are M-by-N matrices, `A` is a triangular matrix, `alpha` is a
/// scalar, and `op(A)` is either `A` or `A^T`. The solution `X` overwrites the
/// input matrix `B` in place.
///
/// # Arguments
/// * `side`   - Specifies if `op(A)` multiplies `X` from the left ('L') or right ('R').
/// * `uplo`   - Specifies if `A` is an upper ('U') or lower ('L') triangular matrix.
/// * `transa` - Specifies the form of `op(A)`: 'N' for `A`, 'T' for `A^T`.
/// * `diag`   - Specifies if `A` is unit triangular ('U') or not ('N').
/// * `m`      - The number of rows of matrix `B`.
/// * `n`      - The number of columns of matrix `B`.
/// * `alpha`  - The scalar multiplier `alpha`. If `alpha` is 0, `B` is set to zero.
/// * `a`      - A raw constant pointer to the first element of matrix `A`.
/// * `lda`    - The leading dimension of `A`.
/// * `b`      - A raw mutable pointer to the first element of matrix `B`. On exit, it is
///   overwritten with the solution matrix `X`.
/// * `ldb`    - The leading dimension of `B`.
#[allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn trsm(
    side: char,
    uplo: char,
    transa: char,
    diag: char,
    m: usize,
    n: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *mut f64,
    ldb: usize,
) {
    #[cfg(feature = "profiling")]
    let _timer = profiling::ScopedTimer::new("TRSM");

    if m == 0 || n == 0 {
        return;
    }

    // Scale B by alpha if necessary.
    if alpha != 1.0 {
        for j in 0..n {
            for i in 0..m {
                *b.add(i + j * ldb) *= alpha;
            }
        }
    }

    let lside = side.eq_ignore_ascii_case(&'L');
    let upper = uplo.eq_ignore_ascii_case(&'U');
    let notrans = transa.eq_ignore_ascii_case(&'N');
    let unit = diag.eq_ignore_ascii_case(&'U');

    // Dispatch to the correct implementation based on the arguments.
    if lside {
        if notrans {
            if upper {
                trsm_lnu(unit, m, n, a, lda, b, ldb);
            } else {
                trsm_lnl(unit, m, n, a, lda, b, ldb);
            }
        } else if upper {
            trsm_ltu(unit, m, n, a, lda, b, ldb);
        } else {
            trsm_ltl(unit, m, n, a, lda, b, ldb);
        }
    } else if notrans {
        if upper {
            trsm_rnu(unit, m, n, a, lda, b, ldb);
        } else {
            trsm_rnl(unit, m, n, a, lda, b, ldb);
        }
    } else if upper {
        trsm_rtu(unit, m, n, a, lda, b, ldb);
    } else {
        trsm_rtl(unit, m, n, a, lda, b, ldb);
    }
}

// Case: SIDE = 'L', TRANS = 'N', UPLO = 'L' (Solve L*X = B)
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn trsm_lnl(unit: bool, m: usize, n: usize, a: *const f64, lda: usize, b: *mut f64, ldb: usize) {
    if m <= TRSM_RECURSION_STOP {
        let core_logic = |j: usize, a: *const f64, b: *mut f64| {
            let b_col = b.add(j * ldb);
            for k in 0..m {
                let b_kj = *b_col.add(k);
                if b_kj != 0.0 {
                    let b_kj_updated = if !unit { b_kj / *a.add(k + k * lda) } else { b_kj };
                    *b_col.add(k) = b_kj_updated;
                    if m > k + 1 {
                        daxpy_update(m - k - 1, -b_kj_updated, a.add(k + 1 + k * lda), 1, b_col.add(k + 1), 1);
                    }
                }
            }
        };
        if n > PARALLEL_THRESHOLD {
            let a_addr = a as usize;
            let b_addr = b as usize;
            (0..n).into_par_iter().for_each(|j| {
                let a = a_addr as *const f64;
                let b = b_addr as *mut f64;
                core_logic(j, a, b);
            });
        } else {
            for j in 0..n {
                core_logic(j, a, b);
            }
        }
        return;
    }
    // Recursive step
    let m1 = m / 2;
    let m2 = m - m1;
    trsm_lnl(unit, m1, n, a, lda, b, ldb);
    gemm('N', 'N', m2, n, m1, -1.0, a.add(m1), lda, b, ldb, 1.0, b.add(m1), ldb);
    trsm_lnl(unit, m2, n, a.add(m1 + m1 * lda), lda, b.add(m1), ldb);
}

// Case: SIDE = 'L', TRANS = 'N', UPLO = 'U' (Solve U*X = B)
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn trsm_lnu(unit: bool, m: usize, n: usize, a: *const f64, lda: usize, b: *mut f64, ldb: usize) {
    if m <= TRSM_RECURSION_STOP {
        let core_logic = |j: usize, a: *const f64, b: *mut f64| {
            let b_col = b.add(j * ldb);
            for k in (0..m).rev() {
                let b_kj = *b_col.add(k);
                if b_kj != 0.0 {
                    let b_kj_updated = if !unit { b_kj / *a.add(k + k * lda) } else { b_kj };
                    *b_col.add(k) = b_kj_updated;
                    if k > 0 {
                        daxpy_update(k, -b_kj_updated, a.add(k * lda), 1, b_col, 1);
                    }
                }
            }
        };
        if n > PARALLEL_THRESHOLD {
            let a_addr = a as usize;
            let b_addr = b as usize;
            (0..n).into_par_iter().for_each(|j| {
                let a = a_addr as *const f64;
                let b = b_addr as *mut f64;
                core_logic(j, a, b);
            });
        } else {
            for j in 0..n {
                core_logic(j, a, b);
            }
        }
        return;
    }
    let m1 = m / 2;
    let m2 = m - m1;
    trsm_lnu(unit, m2, n, a.add(m1 + m1 * lda), lda, b.add(m1), ldb);
    gemm('N', 'N', m1, n, m2, -1.0, a.add(m1 * lda), lda, b.add(m1), ldb, 1.0, b, ldb);
    trsm_lnu(unit, m1, n, a, lda, b, ldb);
}

// Case: SIDE = 'L', TRANS = 'T', UPLO = 'L' (Solve L**T*X = B)
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn trsm_ltl(unit: bool, m: usize, n: usize, a: *const f64, lda: usize, b: *mut f64, ldb: usize) {
    if m <= TRSM_RECURSION_STOP {
        let core_logic = |j: usize, a: *const f64, b: *mut f64| {
            let b_col = b.add(j * ldb);
            for i in (0..m).rev() {
                let mut temp = *b_col.add(i);
                for k in (i + 1)..m {
                    temp -= *a.add(k + i * lda) * *b_col.add(k);
                }
                if !unit {
                    temp /= *a.add(i + i * lda);
                }
                *b_col.add(i) = temp;
            }
        };
        if n > PARALLEL_THRESHOLD {
            let a_addr = a as usize;
            let b_addr = b as usize;
            (0..n).into_par_iter().for_each(|j| {
                let a = a_addr as *const f64;
                let b = b_addr as *mut f64;
                core_logic(j, a, b);
            });
        } else {
            for j in 0..n {
                core_logic(j, a, b);
            }
        }
        return;
    }
    let m1 = m / 2;
    let m2 = m - m1;
    trsm_ltl(unit, m2, n, a.add(m1 + m1 * lda), lda, b.add(m1), ldb);
    gemm('T', 'N', m1, n, m2, -1.0, a.add(m1), lda, b.add(m1), ldb, 1.0, b, ldb);
    trsm_ltl(unit, m1, n, a, lda, b, ldb);
}

// Case: SIDE = 'L', TRANS = 'T', UPLO = 'U' (Solve U**T*X = B)
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn trsm_ltu(unit: bool, m: usize, n: usize, a: *const f64, lda: usize, b: *mut f64, ldb: usize) {
    if m <= TRSM_RECURSION_STOP {
        let core_logic = |j: usize, a: *const f64, b: *mut f64| {
            let b_col = b.add(j * ldb);
            for i in 0..m {
                let mut temp = *b_col.add(i);
                for k in 0..i {
                    temp -= *a.add(k + i * lda) * *b_col.add(k);
                }
                if !unit {
                    temp /= *a.add(i + i * lda);
                }
                *b_col.add(i) = temp;
            }
        };
        if n > PARALLEL_THRESHOLD {
            let a_addr = a as usize;
            let b_addr = b as usize;
            (0..n).into_par_iter().for_each(|j| {
                let a = a_addr as *const f64;
                let b = b_addr as *mut f64;
                core_logic(j, a, b);
            });
        } else {
            for j in 0..n {
                core_logic(j, a, b);
            }
        }
        return;
    }
    let m1 = m / 2;
    let m2 = m - m1;
    trsm_ltu(unit, m1, n, a, lda, b, ldb);
    gemm('T', 'N', m2, n, m1, -1.0, a.add(m1 * lda), lda, b, ldb, 1.0, b.add(m1), ldb);
    trsm_ltu(unit, m2, n, a.add(m1 + m1 * lda), lda, b.add(m1), ldb);
}

// Case: SIDE = 'R', TRANS = 'N', UPLO = 'L' (Solve X*L = B)
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn trsm_rnl(unit: bool, m: usize, n: usize, a: *const f64, lda: usize, b: *mut f64, ldb: usize) {
    if n <= TRSM_RECURSION_STOP {
        let core_logic = |i: usize, a: *const f64, b: *mut f64| {
            let b_row = b.add(i);
            for j in 0..n {
                let mut temp = *b_row.add(j * ldb);
                for k in 0..j {
                    temp -= *b_row.add(k * ldb) * *a.add(j + k * lda);
                }
                if !unit {
                    temp /= *a.add(j + j * lda);
                }
                *b_row.add(j * ldb) = temp;
            }
        };
        if m > PARALLEL_THRESHOLD {
            let a_addr = a as usize;
            let b_addr = b as usize;
            (0..m).into_par_iter().for_each(|i| {
                let a = a_addr as *const f64;
                let b = b_addr as *mut f64;
                core_logic(i, a, b);
            });
        } else {
            for i in 0..m {
                core_logic(i, a, b);
            }
        }
        return;
    }
    let n1 = n / 2;
    let n2 = n - n1;
    trsm_rnl(unit, m, n1, a, lda, b, ldb);
    gemm('N', 'N', m, n2, n1, -1.0, b, ldb, a.add(n1), lda, 1.0, b.add(n1 * ldb), ldb);
    trsm_rnl(unit, m, n2, a.add(n1 + n1 * lda), lda, b.add(n1 * ldb), ldb);
}

// Case: SIDE = 'R', TRANS = 'N', UPLO = 'U' (Solve X*U = B)
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn trsm_rnu(unit: bool, m: usize, n: usize, a: *const f64, lda: usize, b: *mut f64, ldb: usize) {
    if n <= TRSM_RECURSION_STOP {
        let core_logic = |i: usize, a: *const f64, b: *mut f64| {
            let b_row = b.add(i);
            for j in (0..n).rev() {
                let mut temp = *b_row.add(j * ldb);
                for k in (j + 1)..n {
                    temp -= *b_row.add(k * ldb) * *a.add(j + k * lda);
                }
                if !unit {
                    temp /= *a.add(j + j * lda);
                }
                *b_row.add(j * ldb) = temp;
            }
        };
        if m > PARALLEL_THRESHOLD {
            let a_addr = a as usize;
            let b_addr = b as usize;
            (0..m).into_par_iter().for_each(|i| {
                let a = a_addr as *const f64;
                let b = b_addr as *mut f64;
                core_logic(i, a, b);
            });
        } else {
            for i in 0..m {
                core_logic(i, a, b);
            }
        }
        return;
    }
    let n1 = n / 2;
    let n2 = n - n1;
    trsm_rnu(unit, m, n2, a.add(n1 + n1 * lda), lda, b.add(n1 * ldb), ldb);
    gemm('N', 'N', m, n1, n2, -1.0, b.add(n1 * ldb), ldb, a.add(n1), lda, 1.0, b, ldb);
    trsm_rnu(unit, m, n1, a, lda, b, ldb);
}

// Case: SIDE = 'R', TRANS = 'T', UPLO = 'L' (Solve X*L**T = B)
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn trsm_rtl(unit: bool, m: usize, n: usize, a: *const f64, lda: usize, b: *mut f64, ldb: usize) {
    if n <= TRSM_RECURSION_STOP {
        let core_logic = |i: usize, a: *const f64, b: *mut f64| {
            let b_row = b.add(i);
            for j in (0..n).rev() {
                let mut temp = *b_row.add(j * ldb);
                for k in (j + 1)..n {
                    temp -= *b_row.add(k * ldb) * *a.add(k + j * lda);
                }
                if !unit {
                    temp /= *a.add(j + j * lda);
                }
                *b_row.add(j * ldb) = temp;
            }
        };
        if m > PARALLEL_THRESHOLD {
            let a_addr = a as usize;
            let b_addr = b as usize;
            (0..m).into_par_iter().for_each(|i| {
                let a = a_addr as *const f64;
                let b = b_addr as *mut f64;
                core_logic(i, a, b);
            });
        } else {
            for i in 0..m {
                core_logic(i, a, b);
            }
        }
        return;
    }
    let n1 = n / 2;
    let n2 = n - n1;
    trsm_rtl(unit, m, n2, a.add(n1 + n1 * lda), lda, b.add(n1 * ldb), ldb);
    gemm(
        'N',
        'T',
        m,
        n1,
        n2,
        -1.0,
        b.add(n1 * ldb),
        ldb,
        a.add(n1 * lda),
        lda,
        1.0,
        b,
        ldb,
    );
    trsm_rtl(unit, m, n1, a, lda, b, ldb);
}

// Case: SIDE = 'R', TRANS = 'T', UPLO = 'U' (Solve X*U**T = B)
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn trsm_rtu(unit: bool, m: usize, n: usize, a: *const f64, lda: usize, b: *mut f64, ldb: usize) {
    if n <= TRSM_RECURSION_STOP {
        let core_logic = |i: usize, a: *const f64, b: *mut f64| {
            let b_row = b.add(i);
            for j in 0..n {
                let mut temp = *b_row.add(j * ldb);
                for k in 0..j {
                    temp -= *b_row.add(k * ldb) * *a.add(k + j * lda);
                }
                if !unit {
                    temp /= *a.add(j + j * lda);
                }
                *b_row.add(j * ldb) = temp;
            }
        };
        if m > PARALLEL_THRESHOLD {
            let a_addr = a as usize;
            let b_addr = b as usize;
            (0..m).into_par_iter().for_each(|i| {
                let a = a_addr as *const f64;
                let b = b_addr as *mut f64;
                core_logic(i, a, b);
            });
        } else {
            for i in 0..m {
                core_logic(i, a, b);
            }
        }
        return;
    }
    let n1 = n / 2;
    let n2 = n - n1;
    trsm_rtu(unit, m, n1, a, lda, b, ldb);
    gemm('N', 'T', m, n2, n1, -1.0, b, ldb, a.add(n1), lda, 1.0, b.add(n1 * ldb), ldb);
    trsm_rtu(unit, m, n2, a.add(n1 + n1 * lda), lda, b.add(n1 * ldb), ldb);
}
