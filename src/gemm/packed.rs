//! Shared K panels in micro-kernel order. Each input is packed once per call,
//! and each Rayon task owns a disjoint tile of C during a K-panel update.
//! A is stored as [8-row panel][K][row], B as [6-column panel][K][column].
//! Completed packing and output updates form barriers before buffers are reused.
use rayon::prelude::*;
use std::cell::Cell;

const MR: usize = 8;
const NR: usize = 6;
const KC: usize = 256;
const MC: usize = 64;
const NC: usize = 96;
const _: () = assert!(MC.is_multiple_of(MR) && NC.is_multiple_of(NR));

pub(super) fn enough_tiles(m: usize, n: usize, threads: usize) -> bool {
    // Leave room for work stealing, especially across unequal P/E cores.
    m.div_ceil(MC).saturating_mul(n.div_ceil(NC)) >= threads.saturating_mul(2)
}

thread_local! {
    static PANELS: Cell<(Vec<f64>, Vec<f64>)> = const { Cell::new((Vec::new(), Vec::new())) };
}

#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
pub(super) unsafe fn multiply(
    na: bool,
    nb: bool,
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
    threads: usize,
) {
    PANELS.with(|cell| {
        let (mut a_buffer, mut b_buffer) = cell.take();
        let padded_m = m.next_multiple_of(MR);
        let padded_n = n.next_multiple_of(NR);
        // Reserve up to seven extra doubles to align both panels to a cache line.
        a_buffer.resize(padded_m * k.min(KC) + 8, 0.0);
        b_buffer.resize(padded_n * k.min(KC) + 8, 0.0);
        let a_offset = a_buffer.as_ptr().align_offset(64);
        let b_offset = b_buffer.as_ptr().align_offset(64);
        let ap = &mut a_buffer[a_offset..];
        let bp = &mut b_buffer[b_offset..];
        let column_tiles = n.div_ceil(NC);
        let tasks = m.div_ceil(MC) * column_tiles;
        let c_addr = c as usize;

        for pc in (0..k).step_by(KC) {
            let depth = (k - pc).min(KC);
            let a_addr = a as usize;
            let b_addr = b as usize;
            let pack_a = || {
                let pack_chunk = |(block, dst): (usize, &mut [f64])| {
                    let row = block * MC;
                    pack_left(na, m, depth, pc, row, a_addr, lda, dst);
                };
                if threads > 1 && padded_m * depth >= 32768 {
                    ap[..padded_m * depth]
                        .par_chunks_mut(MC * depth)
                        .enumerate()
                        .for_each(pack_chunk);
                } else {
                    ap[..padded_m * depth].chunks_mut(MC * depth).enumerate().for_each(pack_chunk);
                }
            };
            let pack_b = || {
                let pack_chunk = |(block, dst): (usize, &mut [f64])| {
                    let col = block * NR;
                    pack_right(nb, n, depth, pc, col, b_addr, ldb, dst);
                };
                if threads > 1 && padded_n * depth >= 32768 {
                    bp[..padded_n * depth]
                        .par_chunks_mut(NR * depth)
                        .enumerate()
                        .for_each(pack_chunk);
                } else {
                    bp[..padded_n * depth].chunks_mut(NR * depth).enumerate().for_each(pack_chunk);
                }
            };
            if threads > 1 {
                rayon::join(pack_a, pack_b);
            } else {
                let mut pack_a = pack_a;
                let mut pack_b = pack_b;
                pack_a();
                pack_b();
            }
            let update = |task: usize| {
                let row = (task / column_tiles) * MC;
                let col = (task % column_tiles) * NC;
                let rows = (m - row).min(MC);
                let cols = (n - col).min(NC);
                let a_tile = ap.as_ptr().add(row * depth);
                let b_tile = bp.as_ptr().add(col * depth);
                let c_tile = (c_addr as *mut f64).add(row + col * ldc);
                tile(
                    rows,
                    cols,
                    depth,
                    alpha,
                    a_tile,
                    b_tile,
                    if pc == 0 { beta } else { 1.0 },
                    c_tile,
                    ldc,
                );
            };
            if threads > 1 && m.saturating_mul(n).saturating_mul(depth) >= threads.saturating_mul(super::MIN_WORK_PER_THREAD) {
                (0..tasks).into_par_iter().for_each(update);
            } else {
                for task in 0..tasks {
                    update(task);
                }
            }
        }
        cell.set((a_buffer, b_buffer));
    });
}

#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn pack_left(na: bool, m: usize, k: usize, pc: usize, row: usize, a: usize, lda: usize, dst: &mut [f64]) {
    use std::arch::x86_64::*;
    let a = a as *const f64;
    let d = dst.as_mut_ptr();
    for offset in (0..dst.len() / k).step_by(MR) {
        let i = row + offset;
        if na && i + MR <= m {
            for p in 0..k {
                let src = a.add(i + (pc + p) * lda);
                let dest = d.add(offset * k + p * MR);
                _mm256_store_pd(dest, _mm256_loadu_pd(src));
                _mm256_store_pd(dest.add(4), _mm256_loadu_pd(src.add(4)));
            }
        } else {
            for p in 0..k {
                for r in 0..MR {
                    *d.add(offset * k + p * MR + r) = if i + r < m {
                        *a.add(if na { i + r + (pc + p) * lda } else { pc + p + (i + r) * lda })
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn pack_right(nb: bool, n: usize, k: usize, pc: usize, col: usize, b: usize, ldb: usize, dst: &mut [f64]) {
    let b = b as *const f64;
    let d = dst.as_mut_ptr();
    for p in 0..k {
        for s in 0..NR {
            *d.add(p * NR + s) = if col + s < n {
                *b.add(if nb {
                    pc + p + (col + s) * ldb
                } else {
                    col + s + (pc + p) * ldb
                })
            } else {
                0.0
            };
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn tile(m: usize, n: usize, k: usize, alpha: f64, a: *const f64, b: *const f64, beta: f64, c: *mut f64, ldc: usize) {
    use std::arch::x86_64::*;
    for j in (0..n).step_by(NR) {
        for i in (0..m).step_by(MR) {
            let mut lo = [_mm256_setzero_pd(); NR];
            let mut hi = [_mm256_setzero_pd(); NR];
            for p in 0..k {
                // MR is a cache line; every packed A micro-panel stays aligned.
                let av0 = _mm256_load_pd(a.add(i * k + p * MR));
                let av1 = _mm256_load_pd(a.add(i * k + p * MR + 4));
                for s in 0..NR {
                    let bv = _mm256_set1_pd(*b.add(j * k + p * NR + s));
                    lo[s] = _mm256_fmadd_pd(av0, bv, lo[s]);
                    hi[s] = _mm256_fmadd_pd(av1, bv, hi[s]);
                }
            }
            let scale = _mm256_set1_pd(alpha);
            if i + MR <= m && j + NR <= n {
                for s in 0..NR {
                    let dst = c.add(i + (j + s) * ldc);
                    if beta == 0.0 {
                        _mm256_storeu_pd(dst, _mm256_mul_pd(scale, lo[s]));
                        _mm256_storeu_pd(dst.add(4), _mm256_mul_pd(scale, hi[s]));
                    } else {
                        let cv0 = _mm256_loadu_pd(dst);
                        let cv1 = _mm256_loadu_pd(dst.add(4));
                        let (cv0, cv1) = if beta == 1.0 {
                            (cv0, cv1)
                        } else {
                            let bv = _mm256_set1_pd(beta);
                            (_mm256_mul_pd(bv, cv0), _mm256_mul_pd(bv, cv1))
                        };
                        _mm256_storeu_pd(dst, _mm256_fmadd_pd(scale, lo[s], cv0));
                        _mm256_storeu_pd(dst.add(4), _mm256_fmadd_pd(scale, hi[s], cv1));
                    }
                }
            } else {
                // Never read or write C padding, including a partial last column.
                let mut edge = [[0.0; MR]; NR];
                for s in 0..NR {
                    _mm256_storeu_pd(edge[s].as_mut_ptr(), lo[s]);
                    _mm256_storeu_pd(edge[s].as_mut_ptr().add(4), hi[s]);
                }
                for (s, column) in edge.iter().take((n - j).min(NR)).enumerate() {
                    for (r, &value) in column.iter().take((m - i).min(MR)).enumerate() {
                        let dst = c.add(i + r + (j + s) * ldc);
                        let previous = if beta == 0.0 { 0.0 } else { beta * *dst };
                        *dst = alpha.mul_add(value, previous);
                    }
                }
            }
        }
    }
}
