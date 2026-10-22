//! Public API checks: numerical residuals, storage boundaries, and known factors.
use bola::{gemm::gemm, getrf::getrf, getrf2::getrf2, potrf::potrf, potrf2::potrf2, syrk::syrk, trsm::trsm};

fn value(i: usize, j: usize) -> f64 {
    ((i * 17 + j * 13 + 3) % 31) as f64 / 31.0 - 0.5
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-10 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn gemm_transposes_edges_and_padding() {
    for (m, n, k) in [
        (9, 7, 65),
        (33, 19, 67),
        (65, 131, 33),
        (129, 17, 137),
        (257, 17, 257),
        (137, 139, 263),
    ] {
        for ta in ['N', 'T', 'n', 't'] {
            for tb in ['N', 'T', 'n', 't'] {
                let na = ta.eq_ignore_ascii_case(&'N');
                let nb = tb.eq_ignore_ascii_case(&'N');
                let (ar, ac) = if na { (m, k) } else { (k, m) };
                let (br, bc) = if nb { (k, n) } else { (n, k) };
                let lda = ar + 3;
                let ldb = br + 2;
                let ldc = m + 5;
                let mut a = vec![f64::NAN; lda * ac];
                let mut b = vec![f64::NAN; ldb * bc];
                for j in 0..ac {
                    for i in 0..ar {
                        a[i + j * lda] = value(i, j);
                    }
                }
                for j in 0..bc {
                    for i in 0..br {
                        b[i + j * ldb] = value(i + 3, j);
                    }
                }
                // BLAS permits the last column to end immediately after its
                // logical rows; no trailing leading-dimension padding is needed.
                a.truncate((ac - 1) * lda + ar);
                b.truncate((bc - 1) * ldb + br);
                for beta in [0.0, 1.0, -0.25] {
                    let mut c = vec![12345.0; ldc * n];
                    for j in 0..n {
                        for i in 0..m {
                            c[i + j * ldc] = if beta == 0.0 { f64::NAN } else { value(i, j) };
                        }
                    }
                    c.truncate((n - 1) * ldc + m);
                    unsafe {
                        gemm(
                            ta,
                            tb,
                            m,
                            n,
                            k,
                            -0.75,
                            a.as_ptr(),
                            lda,
                            b.as_ptr(),
                            ldb,
                            beta,
                            c.as_mut_ptr(),
                            ldc,
                        );
                    }
                    for j in 0..n {
                        for i in 0..m {
                            let sum: f64 = (0..k)
                                .map(|p| {
                                    let av = if na { a[i + p * lda] } else { a[p + i * lda] };
                                    let bv = if nb { b[p + j * ldb] } else { b[j + p * ldb] };
                                    av * bv
                                })
                                .sum();
                            close(c[i + j * ldc], -0.75 * sum + beta * value(i, j));
                        }
                        assert!(c[m + j * ldc..((j + 1) * ldc).min(c.len())].iter().all(|&v| v == 12345.0));
                    }
                }
            }
        }
    }
}

#[test]
fn gemm_shared_panels_parallel_transposes_and_fringe() {
    // A(i,p)=f(i)+g(p), B(p,j)=h(p)+q(j). The separable form gives an
    // independent O(m*n+k) reference for a product large enough to use the
    // shared-panel path in a twenty-thread pool, including partial K/M/N tiles.
    let (m, n, k) = (385, 481, 263);
    let f = |i: usize| (i % 7) as f64 / 8.0;
    let g = |p: usize| (p % 5) as f64 / 16.0 - 0.25;
    let h = |p: usize| (p % 11) as f64 / 32.0 - 0.125;
    let q = |j: usize| (j % 13) as f64 / 64.0;
    let sg: f64 = (0..k).map(g).sum();
    let sh: f64 = (0..k).map(h).sum();
    let sgh: f64 = (0..k).map(|p| g(p) * h(p)).sum();
    for ta in ['N', 'T'] {
        for tb in ['N', 'T'] {
            let (ar, ac) = if ta == 'N' { (m, k) } else { (k, m) };
            let (br, bc) = if tb == 'N' { (k, n) } else { (n, k) };
            let (lda, ldb, ldc) = (ar + 3, br + 2, m + 5);
            let mut a = vec![f64::NAN; (ac - 1) * lda + ar];
            let mut b = vec![f64::NAN; (bc - 1) * ldb + br];
            for p in 0..k {
                for i in 0..m {
                    a[if ta == 'N' { i + p * lda } else { p + i * lda }] = f(i) + g(p);
                }
                for j in 0..n {
                    b[if tb == 'N' { p + j * ldb } else { j + p * ldb }] = h(p) + q(j);
                }
            }
            for beta in [0.0, 1.0, -0.25] {
                let mut c = vec![12345.0; (n - 1) * ldc + m];
                for j in 0..n {
                    for i in 0..m {
                        c[i + j * ldc] = if beta == 0.0 { f64::NAN } else { value(i, j) };
                    }
                }
                unsafe {
                    gemm(
                        ta,
                        tb,
                        m,
                        n,
                        k,
                        -0.75,
                        a.as_ptr(),
                        lda,
                        b.as_ptr(),
                        ldb,
                        beta,
                        c.as_mut_ptr(),
                        ldc,
                    );
                }
                for j in 0..n {
                    for i in 0..m {
                        let product = f(i) * sh + f(i) * q(j) * k as f64 + sgh + q(j) * sg;
                        close(c[i + j * ldc], -0.75 * product + beta * value(i, j));
                    }
                    assert!(c[m + j * ldc..((j + 1) * ldc).min(c.len())].iter().all(|&v| v == 12345.0));
                }
            }
        }
    }
}

#[test]
fn gemm_empty_product_only_scales_c() {
    let mut c = [f64::NAN; 6];
    unsafe {
        gemm(
            'N',
            'N',
            2,
            3,
            0,
            f64::NAN,
            std::ptr::null(),
            2,
            std::ptr::null(),
            1,
            0.0,
            c.as_mut_ptr(),
            2,
        );
    }
    assert_eq!(c, [0.0; 6]);
}

#[test]
fn syrk_all_variants_across_blocks() {
    let (n, k) = (67, 39);
    for uplo in ['U', 'L'] {
        for trans in ['N', 'T'] {
            let (rows, cols) = if trans == 'N' { (n, k) } else { (k, n) };
            let lda = rows + 3;
            let ldc = n + 2;
            let mut a = vec![f64::NAN; lda * cols];
            for j in 0..cols {
                for i in 0..rows {
                    a[i + j * lda] = value(i, j);
                }
            }
            let mut c = vec![12345.0; ldc * n];
            unsafe {
                syrk(uplo, trans, n, k, 0.75, a.as_ptr(), lda, -0.5, c.as_mut_ptr(), ldc);
            }
            for j in 0..n {
                for i in 0..ldc {
                    if i < n && ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j)) {
                        let sum: f64 = (0..k)
                            .map(|p| {
                                if trans == 'N' {
                                    a[i + p * lda] * a[j + p * lda]
                                } else {
                                    a[p + i * lda] * a[p + j * lda]
                                }
                            })
                            .sum();
                        close(c[i + j * ldc], 0.75 * sum - 0.5 * 12345.0);
                    } else {
                        assert_eq!(c[i + j * ldc], 12345.0);
                    }
                }
            }
        }
    }
}

#[test]
fn lu_rectangular_residuals_and_minimal_storage() {
    for (m, n) in [(1, 5), (2, 5), (17, 41), (41, 17), (65, 71), (71, 65), (129, 137)] {
        let lda = m + 3;
        let mut original = vec![12345.0; (n - 1) * lda + m];
        for j in 0..n {
            for i in 0..m {
                original[i + j * lda] = value(i, j) + if i == j { 2.0 } else { 0.0 };
            }
        }
        for recursive in [false, true] {
            let mut a = original.clone();
            let mut pivots = vec![0; m.min(n)];
            unsafe {
                if recursive {
                    getrf2(m, n, a.as_mut_ptr(), lda, &mut pivots).unwrap();
                } else {
                    getrf(m, n, &mut a, lda, &mut pivots).unwrap();
                }
            }
            let mut pa = original.clone();
            for (i, &pivot) in pivots.iter().enumerate() {
                assert!((i + 1..=m).contains(&(pivot as usize)));
                for j in 0..n {
                    pa.swap(i + j * lda, pivot as usize - 1 + j * lda);
                }
            }
            for j in 0..n {
                for i in 0..m {
                    let mut product = 0.0;
                    for p in 0..m.min(n) {
                        let l = if i == p {
                            1.0
                        } else if i > p {
                            a[i + p * lda]
                        } else {
                            0.0
                        };
                        let u = if p <= j { a[p + j * lda] } else { 0.0 };
                        product += l * u;
                    }
                    close(product, pa[i + j * lda]);
                }
            }
            for j in 0..n - 1 {
                assert!(a[m + j * lda..(j + 1) * lda].iter().all(|&v| v == 12345.0));
            }
        }
    }
}

#[test]
fn trsm_all_variants_residuals() {
    let (m, n) = (73, 67);
    for side in ['L', 'R'] {
        for uplo in ['U', 'L'] {
            for trans in ['N', 'T'] {
                for diag in ['U', 'N'] {
                    let order = if side == 'L' { m } else { n };
                    let lda = order + 3;
                    let ldb = m + 2;
                    let mut a = vec![f64::NAN; lda * order];
                    for j in 0..order {
                        for i in 0..order {
                            if (uplo == 'U' && i < j) || (uplo == 'L' && i > j) {
                                a[i + j * lda] = value(i, j) / order as f64;
                            }
                            if i == j && diag == 'N' {
                                a[i + j * lda] = 2.0;
                            }
                        }
                    }
                    let mut b = vec![12345.0; ldb * n];
                    for j in 0..n {
                        for i in 0..m {
                            b[i + j * ldb] = value(i, j);
                        }
                    }
                    unsafe {
                        trsm(side, uplo, trans, diag, m, n, -0.75, a.as_ptr(), lda, b.as_mut_ptr(), ldb);
                    }
                    let op = |i: usize, j: usize| {
                        let (i, j) = if trans == 'N' { (i, j) } else { (j, i) };
                        if i == j && diag == 'U' {
                            1.0
                        } else if (uplo == 'U' && i <= j) || (uplo == 'L' && i >= j) {
                            a[i + j * lda]
                        } else {
                            0.0
                        }
                    };
                    for j in 0..n {
                        for i in 0..m {
                            let sum: f64 = (0..order)
                                .map(|p| {
                                    if side == 'L' {
                                        op(i, p) * b[p + j * ldb]
                                    } else {
                                        b[i + p * ldb] * op(p, j)
                                    }
                                })
                                .sum();
                            close(sum, -0.75 * value(i, j));
                        }
                        assert!(b[m + j * ldb..(j + 1) * ldb].iter().all(|&v| v == 12345.0));
                    }
                }
            }
        }
    }
}

#[test]
fn trsm_zero_alpha_clears_nan_rhs() {
    let a = [f64::NAN; 4];
    let mut b = [f64::NAN; 4];
    unsafe {
        trsm('L', 'L', 'N', 'N', 2, 2, 0.0, a.as_ptr(), 2, b.as_mut_ptr(), 2);
    }
    assert_eq!(b, [0.0; 4]);
}

#[test]
fn cholesky_residuals_across_blocks() {
    let n = 67;
    let lda = n + 3;
    let mut original = vec![0.0; n * n];
    let l = |i: usize, j: usize| {
        if i == j {
            2.0
        } else if i > j {
            value(i, j) / n as f64
        } else {
            0.0
        }
    };
    for j in 0..n {
        for i in 0..n {
            original[i + j * n] = (0..n).map(|p| l(i, p) * l(j, p)).sum();
        }
    }
    for (name, factor) in CHOLESKY_ROUTINES {
        for uplo in ['U', 'L', 'u', 'l'] {
            let upper = uplo.eq_ignore_ascii_case(&'U');
            let mut a = vec![f64::NAN; (n - 1) * lda + n];
            for j in 0..n {
                for i in 0..n {
                    if (upper && i <= j) || (!upper && i >= j) {
                        a[i + j * lda] = original[i + j * n];
                    }
                }
            }
            unsafe {
                factor(uplo, n, a.as_mut_ptr(), lda).unwrap_or_else(|error| panic!("{name}({uplo}): {error}"));
            }
            for j in 0..n {
                for i in 0..n {
                    let sum: f64 = (0..=i.min(j))
                        .map(|p| {
                            if upper {
                                a[p + i * lda] * a[p + j * lda]
                            } else {
                                a[i + p * lda] * a[j + p * lda]
                            }
                        })
                        .sum();
                    close(sum, original[i + j * n]);
                    if (upper && i > j) || (!upper && i < j) {
                        assert!(a[i + j * lda].is_nan());
                    }
                }
                assert!(a[n + j * lda..((j + 1) * lda).min(a.len())].iter().all(|v| v.is_nan()));
            }
        }
    }
}

type Cholesky = unsafe fn(char, usize, *mut f64, usize) -> Result<(), String>;
const CHOLESKY_ROUTINES: [(&str, Cholesky); 2] = [("potrf", potrf), ("potrf2", potrf2)];

#[test]
fn cholesky_validates_arguments_before_access() {
    for (name, factor) in CHOLESKY_ROUTINES {
        for uplo in ['?', ' ', '\0', 'Ü'] {
            // Even an empty matrix must reject an invalid triangle flag.
            let result = unsafe { factor(uplo, 0, std::ptr::null_mut(), 0) };
            assert_eq!(result, Err(format!("Argument 1 to {name} had an illegal value")));
            let mut a = [4.0, 0.0, 0.0, 4.0];
            let result = unsafe { factor(uplo, 2, a.as_mut_ptr(), 2) };
            assert_eq!(result, Err(format!("Argument 1 to {name} had an illegal value")));
            assert_eq!(a, [4.0, 0.0, 0.0, 4.0]);
        }
        for uplo in ['U', 'L', 'u', 'l'] {
            for (n, lda) in [(0, 0), (1, 0), (2, 1)] {
                let mut a = [4.0, 0.0, 0.0, 4.0];
                let result = unsafe { factor(uplo, n, a.as_mut_ptr(), lda) };
                assert_eq!(result, Err(format!("Argument 4 to {name} had an illegal value")));
                assert_eq!(a, [4.0, 0.0, 0.0, 4.0]);
            }
            assert_eq!(unsafe { factor(uplo, 0, std::ptr::null_mut(), 1) }, Ok(()));
        }
    }
}

#[test]
fn cholesky_reports_global_failure_column() {
    for (name, factor) in CHOLESKY_ROUTINES {
        for uplo in ['U', 'L', 'u', 'l'] {
            for (n, failed_column) in [(1, 0), (3, 1), (40, 35), (67, 31), (67, 32), (67, 63), (67, 64), (67, 66)] {
                for pivot in [-1.0, 0.0, f64::NAN] {
                    let mut a = vec![0.0; n * n];
                    for i in 0..n {
                        a[i + i * n] = 1.0;
                    }
                    a[failed_column + failed_column * n] = pivot;
                    let result = unsafe { factor(uplo, n, a.as_mut_ptr(), n) };
                    assert_eq!(
                        result,
                        Err(format!(
                            "Matrix is not positive-definite. Failure at column {}.",
                            failed_column + 1
                        )),
                        "{name}({uplo}), n={n}, pivot={pivot}"
                    );
                }
            }
        }
    }
}

#[test]
fn lu_rejects_short_slices_and_overflow() {
    unsafe {
        assert!(getrf(2, 2, &mut [1.0; 3], 2, &mut [0; 2]).is_err());
        assert!(getrf(2, 2, &mut [1.0; 4], 2, &mut [0; 1]).is_err());
        assert!(getrf(2, usize::MAX, &mut [1.0; 4], 2, &mut [0; 2]).is_err());
        assert!(getrf2(2, 2, [1.0; 4].as_mut_ptr(), 2, &mut [0; 1]).is_err());
    }
}

#[test]
fn idamax_returns_first_tied_maximum() {
    for n in [16, 17, 32, 65] {
        for first in 0..n {
            for second in first + 1..n {
                let mut x = vec![0.0; n];
                x[first] = -2.0;
                x[second] = 2.0;
                assert_eq!(unsafe { bola::idamax::idamax(n, x.as_ptr()) }, first);
            }
        }
    }
}

#[test]
fn lu_handles_subnormal_pivots() {
    let mut a = [1e-320, 5e-321, 0.0, 2.0];
    unsafe {
        getrf(2, 2, &mut a, 2, &mut [0; 2]).unwrap();
    }
    close(a[1], 0.5);
    close(a[3], 2.0);
}

// Small, independently known factors complement the larger residual checks.
mod known_answers {
    use bola::getrf::getrf;
    use bola::potrf::potrf;
    use bola::utilities::assert_approx_eq;

    #[test]
    fn test_potrf_cholesky_factorisation_upper() {
        let n = 3;
        let lda = 3;
        let mut a = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];

        let u_expected = vec![2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0];

        unsafe {
            potrf('U', n, a.as_mut_ptr(), lda).expect("potrf failed for UPLO='U'");
        }

        let mut u_result = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..=j {
                u_result[i + j * lda] = a[i + j * lda];
            }
        }

        assert_approx_eq(&u_result, &u_expected, 1e-9);
    }

    #[test]
    fn test_potrf_cholesky_factorisation_lower() {
        let n = 3;
        let lda = 3;
        let mut a = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];

        let l_expected = vec![2.0, 6.0, -8.0, 0.0, 1.0, 5.0, 0.0, 0.0, 3.0];

        unsafe {
            potrf('L', n, a.as_mut_ptr(), lda).expect("potrf failed for UPLO='L'");
        }

        let mut l_result = vec![0.0; 9];
        for j in 0..n {
            for i in j..n {
                l_result[i + j * lda] = a[i + j * lda];
            }
        }

        assert_approx_eq(&l_result, &l_expected, 1e-9);
    }

    #[test]
    fn test_getrf_lu_decomposition() {
        let m = 3;
        let n = 3;
        let lda = 3;
        let mut a = vec![1.80, 5.25, 1.58, 2.88, -2.95, -2.69, 2.05, -0.95, -2.90];
        let mut ipiv = vec![0; 3];

        let lu_expected = vec![5.25, 0.3429, 0.301, -2.95, 3.8914, -0.4631, -0.95, 2.3757, -1.5139];
        let ipiv_expected = vec![2, 2, 3];

        unsafe {
            getrf(m, n, &mut a, lda, &mut ipiv).expect("getrf failed on non-singular matrix");
        }

        assert_approx_eq(&a, &lu_expected, 1e-4);
        assert_eq!(ipiv, ipiv_expected);
    }
}
