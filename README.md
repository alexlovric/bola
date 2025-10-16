# BOLA

BOLA is a native Rust linear algebra backend inspired by BLAS/LAPACK. The purpose of this project is to provide a drop in replacement for many of BLAS/LAPACK functions with a tiny memory footprint, ideal for embedded systems.

## Features

* The library currently provides the following key functions:

    * GEMM: General matrix-matrix multiplication.
    * GETRF: LU factorisation of a general matrix.
    * POTRF: Cholesky factorisation of a symmetric positive-definite matrix.
    * TRSM: Triangular solve with multiple right-hand sides.
    * Other functions they depend on (LASWP, IDAMAX, SYRK, etc.)

* SIMD Accelerated as well as Rayon parallelised.

* Cross-Platform Support: Bola is written in stable Rust and has been tested on the following architectures:

    * x86_64 (with AVX2)
    * AArch64 (with NEON)

## Examples

```rust
...
use blast::gemm::gemm;

let m = 128;
let n = 128;
let k = 128;
let lda = m;
let ldb = k;
let ldc = m;

let mut rng = rand::rng();
let a: Vec<f64> = (0..(m * k)).map(|_| rng.random()).collect();
let b: Vec<f64> = (0..(k * n)).map(|_| rng.random()).collect();
let c_init: Vec<f64> = (0..(m * n)).map(|_| rng.random()).collect();

let mut my_c = c_init.clone();

unsafe {
    gemm(
        'N', 'N', m, n, k, 1.0, a.as_ptr(), 
        lda, b.as_ptr(), ldb, 1.0, my_c.as_mut_ptr(), ldc
    );
}
...
```

## Comparison
Comparisons run on a computer with Intel i5-14600K × 20 and 64GB RAM. Sizes of the matrices are 1024x1024, 2048x2048 and 4096x4096.

### GEMM
The general matrix-matrix multiplication function is compared against MatrixMultiply and CBLAS. MatrixMultiply with threading feature enabled. The average time in milliseconds is shown.

| Size | BOLA (ms) | MatrixMultiply (ms) | CBLAS (ms) |
|------|------|----------------|-------|
| 1024 | 4.8205 | 8.7194 | 5.4872 |
| 2048 | 38.331 | 70.590 | 40.052 |
| 4096 | 334.57 | 570.56 | 290.42 |

### GETRF
The LU factorisation function is compared against Lapack. The average time in milliseconds is shown.

| Size | BOLA (ms) | Lapack (ms) |
|------|------|--------|
| 1024 | 6.7720 | 3.9243 |
| 2048 | 30.617 | 23.097 |
| 4096 | 186.47 | 129.05 |

## License
MIT License - See [LICENSE](LICENSE) for details.

## Support
If you'd like to support the project consider:
- Identifying the features you'd like to see implemented or bugs you'd like to fix and open an issue.
- Contributing to the code by resolving existing issues, I'm happy to have you.
- Donating to help me continue development, [Buy Me a Coffee](https://coff.ee/alexlovric)