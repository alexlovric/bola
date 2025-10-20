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
