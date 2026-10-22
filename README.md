# BOLA

Double-precision linear algebra in native Rust, with SIMD kernels and Rayon
parallelism. BOLA provides BLAS/LAPACK-style operations on column-major `f64`
matrices without requiring an installed BLAS or LAPACK library.

## Operations

| Function | Operation |
|---|---|
| `gemm` | General matrix multiplication |
| `getrf`, `getrf2` | LU factorization with partial pivoting |
| `potrf`, `potrf2` | Cholesky factorization |
| `trsm` | Triangular solve with multiple right-hand sides |
| `syrk`, `ger`, `laswp`, `idamax` | Supporting matrix updates, row swaps and pivot selection |

Rayon is the only direct runtime dependency. The implementation uses packed
panels, cache blocking and architecture-specific SIMD kernels. The APIs expose
raw pointers and BLAS-style dimensions and strides; callers are responsible for
valid storage and the safety requirements of each function.

## Why use BOLA?

- **Small dependency footprint.** BOLA focuses on the `f64` operations listed
  above, with Rayon as its only direct runtime dependency. Benchmark libraries
  are development dependencies.
- **Cargo-only numerical builds.** Cargo compiles the Rust kernels directly,
  with no separate BLAS/LAPACK installation or C/Fortran numerical library build.
- **Shared Rust parallelism.** The kernels use Rayon, allowing applications to
  run them in the same thread pool as their other Rayon work.
- **Fast for selected workloads.** On the Linux machine below, BOLA beats
  OpenBLAS for the tested narrow-batch GEMM and twenty-thread large square GEMM,
  and LAPACK + OpenBLAS for 32- and 64-square LU. The tables show where each
  library performs best.

## Platform status

Tests and benchmarks were run on **CachyOS Linux** (Arch-based), kernel
**7.2.11-1-cachyos**, with an **Intel Core i5-14600K**: six performance cores,
eight efficiency cores and twenty logical CPUs.

**macOS and Windows are untested as of this review.** The AArch64/NEON path passes
a cross-target compile check, but has not been executed or benchmarked on ARM
hardware in this review. The x86-64 kernels require **AVX2 and FMA**; complete
runtime CPU-feature dispatch is not implemented yet.

## Usage

Add `bola` to your project:

```sh
cargo add bola
```

This multiplies two 2×2 matrices. Elements are stored column by column, and each
leading dimension is the number of rows in its matrix.

```rust
use bola::gemm::gemm;

fn main() {
    let (m, n, k) = (2, 2, 2);
    let a = [1.0, 3.0, 2.0, 4.0];
    let b = [5.0, 7.0, 6.0, 8.0];
    let mut c = [0.0; 4];

    // The buffers have the required extents, and C does not overlap A or B.
    unsafe {
        gemm(
            'N', 'N', m, n, k,
            1.0, a.as_ptr(), m,
            b.as_ptr(), k,
            0.0, c.as_mut_ptr(), m,
        );
    }

    assert_eq!(c, [19.0, 43.0, 22.0, 50.0]);
}
```

## Performance

Measured on the Linux machine above on **2026-10-22**, after updating the Rust
toolchain and dependencies. The comparisons use Rust/Cargo **1.99.1**,
Criterion **0.8.2** and matrixmultiply **0.3.11**.

The external numerical backends used for the recorded results are:

- **GEMM:** OpenBLAS **0.3.34**, built for **HASWELL** with GCC/GFortran
  **16.2.1**, **pthreads**, a twenty-thread limit and automatic CPU affinity
  disabled.
- **LU:** **Reference LAPACK**, built separately from its **v3.12.1 source
  tag** and linked to that same OpenBLAS build for its BLAS operations. The LU
  tables label this combination **Reference LAPACK + OpenBLAS**.

Values are Criterion **mean times**, with lower values better. Each case uses
ten samples, a one-second warmup and a three-second measurement target, extended
for larger cases. Matrix generation, cloning, numerical validation and destruction
are excluded from timing. The optional `profiling` feature is disabled.

Four-thread runs use separate performance cores, CPUs `2,4,6,8`. Twenty-thread
runs allow all logical CPUs, `0-19`. **matrixmultiply supports at most four
threads**, so its twenty-thread rows still use a maximum of four. Libraries may
use fewer workers for small problems; the mixed-core results are also sensitive
to OS thread placement.

Large GEMM shares packed panels across workers. A 4096-square product uses about
16 MiB of reusable scratch storage per calling thread; small and narrow products
keep a lighter path.

### Small GEMM

Four performance cores; times in **microseconds**. Size means `M = N = K`.

| Size | BOLA | matrixmultiply | OpenBLAS |
|---|---:|---:|---:|
| 32 | 1.167 | 1.165 | 1.076 |
| 64 | 6.872 | 8.806 | 7.979 |
| 128 | 24.625 | 41.120 | 20.021 |
| 256 | 142.370 | 155.839 | 124.551 |
| 512 | 948.831 | 1129.665 | 915.494 |

### Large and rectangular GEMM

Times in **milliseconds**. A single size means `M = N = K`; rectangular cases
show `M × N × K`.

| Threads | Size | BOLA | matrixmultiply | OpenBLAS |
|---|---|---:|---:|---:|
| 4 | 1024 | 7.128 | 8.683 | 6.833 |
| 4 | 2048 | 62.792 | 69.700 | 54.116 |
| 4 | 4096 | 487.671 | 572.324 | 426.059 |
| 4 | 1024 × 32 × 1024 | 0.312 | 0.581 | 0.443 |
| 20 | 1024 | 3.861 | 8.839 | 5.929 |
| 20 | 2048 | 29.682 | 71.247 | 34.570 |
| 20 | 4096 | 224.302 | 571.588 | 271.737 |
| 20 | 1024 × 32 × 1024 | 0.196 | 0.584 | 0.399 |

At four threads, BOLA is faster than matrixmultiply on these large square cases
and remains behind OpenBLAS. At twenty threads, the measured 4096-square GEMM
is 224.302 ms for BOLA and 271.737 ms for OpenBLAS. The narrow product benefits
from splitting work across rows as well as columns.

### Why the narrow-batch result matters

The `1024 × 32 × 1024` case multiplies a **1024×1024 matrix by a 1024×32
matrix**, producing 32 output vectors. BOLA takes **0.312 ms versus 0.443 ms**
for OpenBLAS with four threads (about **1.4× faster**), and **0.196 ms versus
0.399 ms** with twenty (about **2.0× faster**). It is also about **1.9×** and
**3.0×** faster than matrixmultiply at those respective thread settings;
matrixmultiply uses at most four threads.

This shape occurs when applying the same dense operator to a small batch of
vectors: [small-batch dense-layer projections](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html),
transforming several signals at once, or advancing a small ensemble of
simulation states. Batching lets those vectors reuse the operator's
coefficients. Lower latency here can matter when the application cannot wait to
collect a much larger batch.

BOLA distributes this narrow output across rows as well as columns, exposing
more parallel work than splitting its 32 columns alone. The measured advantage
applies to this shape and machine; other aspect ratios and batch sizes need
their own measurements.

### Small LU

Times in **microseconds**. Size is the order of a square matrix.

| Threads | Size | BOLA | Reference LAPACK + OpenBLAS |
|---|---|---:|---:|
| 4 | 32 | 1.989 | 4.994 |
| 4 | 64 | 11.947 | 16.869 |
| 4 | 128 | 88.524 | 61.756 |
| 4 | 256 | 385.427 | 323.142 |
| 4 | 512 | 1518.893 | 1533.799 |
| 20 | 32 | 1.975 | 5.492 |
| 20 | 64 | 11.919 | 31.190 |
| 20 | 128 | 432.383 | 106.263 |
| 20 | 256 | 2326.905 | 455.947 |
| 20 | 512 | 1686.047 | 1785.608 |

### Large LU

Times in **milliseconds**.

| Threads | Size | BOLA | Reference LAPACK + OpenBLAS |
|---|---|---:|---:|
| 4 | 1024 | 6.900 | 7.550 |
| 4 | 2048 | 38.953 | 42.142 |
| 4 | 4096 | 315.810 | 279.975 |
| 20 | 1024 | 5.966 | 13.920 |
| 20 | 2048 | 25.901 | 38.177 |
| 20 | 4096 | 212.141 | 250.408 |

More threads do not help every shape. In particular, the 128- and 256-square LU
cases are slower with twenty threads than with four on this machine. Threading
and triangular-solve granularity remain areas for improvement.

## Tests and profiling

The native tests and profiling example need no external BLAS installation:

```sh
cargo test --release
RAYON_NUM_THREADS=4 cargo run --release --example profile -- gemm
RAYON_NUM_THREADS=4 cargo run --release --example profile -- getrf
```

`tests/linear_algebra.rs` combines known-answer factorization checks with
regressions for transposes, pivot behavior, block boundaries, padding, minimal
storage and numerical edge cases. Small kernel checks also live beside the
implementation.

The profiling example reports the median, minimum and maximum of seven timing
windows in microseconds. Input generation and restoration are outside the timed
region. Enable `--features profiling` for function-level timers; instrumentation
adds overhead, so keep that feature disabled for benchmark comparisons.

## Running benchmarks

The LU benchmark runs natively without BLAS:

```sh
RAYON_NUM_THREADS=4 cargo bench --bench getrf
```

Enable `external-blas` to add LAPACK to the LU comparison or to run the GEMM
comparison against OpenBLAS and matrixmultiply:

```sh
RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MATMUL_NUM_THREADS=4 \
  cargo bench --features external-blas --bench gemm -- \
  --measurement-time 3 --warm-up-time 1 --sample-size 10 --noplot

RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  cargo bench --features external-blas --bench getrf
```

`BOLA_BENCH_SIZES=32,64,128,256,512` selects square cases before allocating or
validating inputs. The default includes those sizes plus 1024, 2048 and 4096.
GEMM also includes `1024 × 32 × 1024`; use Criterion's filter to select individual
cases. Match the libraries' thread settings when comparing them.

| Variable | Purpose |
|---|---|
| `BOLA_BLAS_LIB` | BLAS library name; defaults to `openblas`. Cargo link syntax such as `static=openblas` is accepted. |
| `BOLA_LAPACK_LIB` | Optional separate LAPACK library, such as `lapack`; otherwise use the LAPACK bundled with OpenBLAS. |
| `BOLA_BLAS_DIR` | Native library search directory for installations outside the standard paths. |

For shared libraries outside standard runtime paths, also configure your system's
loader search path, such as `LD_LIBRARY_PATH` on Linux.

Detailed reports, raw exports and archived experiments are kept locally under
`docs/`, which is ignored by Git. The README tables provide the versioned
summary.

## Contributing

Bug reports, numerical test cases and reproducible benchmarks are welcome.
You can also [support development](https://coff.ee/alexlovric).

## License

MIT. See [LICENSE](LICENSE).

## Support
If you'd like to support the project consider:

* Identifying the features you'd like to see implemented or bugs you'd like to fix and open an issue.
* Contributing to the code by resolving existing issues, I'm happy to have you.
* Donating to help me continue development, [Buy Me a Coffee](https://buymeacoffee.com/alexlovric)
