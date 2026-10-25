# reference material: https://netlib.org/lapack/explore-html/d4/d6e/group__herk_ga09ec411a6b845c47efb0bc8bd3283b03.html?utm_source=chatgpt.com

import os
import sys
import time

# Set single-threaded execution for external BLAS libraries to ensure reproducible benchmarks
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg.blas import dsyrk
from numba import njit, prange, set_num_threads, get_num_threads

BS = 64
L2 = 64
L1 = 64


# ---------------------------------------------------------
# Baseline 0: Pure Python
# ---------------------------------------------------------
def v00_python(X, C, alpha, beta, upper):
    n, k = X.shape
    for i in range(n):
        for j in range(i, n) if upper else range(i + 1):
            s = 0.0
            for l in range(k):
                s += X[i, l] * X[j, l]
            C[i, j] = alpha * s + beta * C[i, j]


# ---------------------------------------------------------
# Baseline 1: Naive SYRK in Numba
# ---------------------------------------------------------
@njit
def v01_numba(X, C, a, b, u):
    n, k = X.shape
    for i in range(n):
        for j in range(i, n) if u else range(i + 1):
            s = 0.0
            for l in range(k):
                s += X[i, l] * X[j, l]
            C[i, j] = a * s + b * C[i, j]


# ---------------------------------------------------------
# Baseline 2: Loop Order Permutations (6 explicit functions)
# IJL and JIL are theoretically the fastest
# as the innermost loop accesses contiguous rows of X
# ---------------------------------------------------------
@njit
def v02_ijl(X, C, a, b, u):
    n, k = X.shape
    for i in range(n):
        for j in range(i, n) if u else range(i + 1):
            s = 0.0
            for l in range(k):
                s += X[i, l] * X[j, l]
            C[i, j] = a * s + b * C[i, j]


@njit
def v02_ilj(X, C, a, b, u):
    n, k = X.shape
    # Scale only the selected triangle before accumulating partial products.
    for row in range(n):
        for col in range(row, n) if u else range(row + 1):
            C[row, col] *= b
    for i in range(n):
        for l in range(k):
            ai = a * X[i, l]
            for j in range(i, n) if u else range(i + 1):
                C[i, j] += ai * X[j, l]


@njit
def v02_jil(X, C, a, b, u):
    n, k = X.shape
    for j in range(n):
        for i in range(j + 1) if u else range(j, n):
            s = 0.0
            for l in range(k):
                s += X[i, l] * X[j, l]
            C[i, j] = a * s + b * C[i, j]


@njit
def v02_jli(X, C, a, b, u):
    n, k = X.shape
    # Scale only the selected triangle before accumulating partial products.
    for row in range(n):
        for col in range(row, n) if u else range(row + 1):
            C[row, col] *= b
    for j in range(n):
        for l in range(k):
            aj = a * X[j, l]
            for i in range(j + 1) if u else range(j, n):
                C[i, j] += X[i, l] * aj


@njit
def v02_lij(X, C, a, b, u):
    n, k = X.shape
    # Scale only the selected triangle before accumulating partial products.
    for row in range(n):
        for col in range(row, n) if u else range(row + 1):
            C[row, col] *= b
    for l in range(k):
        for i in range(n):
            ai = a * X[i, l]
            for j in range(i, n) if u else range(i + 1):
                C[i, j] += ai * X[j, l]


@njit
def v02_lji(X, C, a, b, u):
    n, k = X.shape
    # Scale only the selected triangle before accumulating partial products.
    for row in range(n):
        for col in range(row, n) if u else range(row + 1):
            C[row, col] *= b
    for l in range(k):
        for j in range(n):
            aj = a * X[j, l]
            for i in range(j + 1) if u else range(j, n):
                C[i, j] += X[i, l] * aj


# ---------------------------------------------------------
# Baseline 3: Manual loop unrolling (Lecture 05)
# ---------------------------------------------------------
@njit
def v03_unroll4(X, C, a, b, u):
    n, k = X.shape
    lim = k - (k % 4)
    for i in range(n):
        for j in range(i, n) if u else range(i + 1):
            s0 = s1 = s2 = s3 = 0.0
            for l in range(0, lim, 4):
                s0 += X[i, l] * X[j, l]
                s1 += X[i, l + 1] * X[j, l + 1]
                s2 += X[i, l + 2] * X[j, l + 2]
                s3 += X[i, l + 3] * X[j, l + 3]
            s = s0 + s1 + s2 + s3
            for l in range(lim, k):
                s += X[i, l] * X[j, l]
            C[i, j] = a * s + b * C[i, j]


# ---------------------------------------------------------
# Baseline 4: Manual unrolling + Fastmath
# ---------------------------------------------------------
@njit(fastmath=True)
def v04_unroll4_fastmath(X, C, a, b, u):
    n, k = X.shape
    lim = k - (k % 4)
    for i in range(n):
        for j in range(i, n) if u else range(i + 1):
            s0 = s1 = s2 = s3 = 0.0
            for l in range(0, lim, 4):
                s0 += X[i, l] * X[j, l]
                s1 += X[i, l + 1] * X[j, l + 1]
                s2 += X[i, l + 2] * X[j, l + 2]
                s3 += X[i, l + 3] * X[j, l + 3]
            s = s0 + s1 + s2 + s3
            for l in range(lim, k):
                s += X[i, l] * X[j, l]
            C[i, j] = a * s + b * C[i, j]


# ---------------------------------------------------------
# Baseline 5: Fastmath only (original IJL reduction)
# Fastmath may already enable compiler-level loop unrolling and vectorization,
# so additional manual unrolling can reduce performance
# ---------------------------------------------------------
@njit(fastmath=True)
def v05_fastmath(X, C, a, b, u):
    n, k = X.shape
    for i in range(n):
        for j in range(i, n) if u else range(i + 1):
            s = 0.0
            for l in range(k):
                s += X[i, l] * X[j, l]
            C[i, j] = a * s + b * C[i, j]


# ---------------------------------------------------------
# Baseline 6: Parallel Loop Versions
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def v06_parallel_i(X, C, a, b, u):
    n, k = X.shape
    for i in prange(n):
        for j in range(i, n) if u else range(i + 1):
            s = 0.0
            for l in range(k):
                s += X[i, l] * X[j, l]
            C[i, j] = a * s + b * C[i, j]


@njit(parallel=True, fastmath=True)
def v06_parallel_j(X, C, a, b, u):
    n, k = X.shape
    for j in prange(n):
        for i in range(j + 1) if u else range(j, n):
            s = 0.0
            for l in range(k):
                s += X[i, l] * X[j, l]
            C[i, j] = a * s + b * C[i, j]


@njit(parallel=True, fastmath=True)
def v06_parallel_l(X, C, a, b, u):
    n, k = X.shape
    for i in range(n):
        for j in range(i, n) if u else range(i + 1):
            s = 0.0
            for l in prange(k):
                s += X[i, l] * X[j, l]
            C[i, j] = a * s + b * C[i, j]


# ---------------------------------------------------------
# Baseline 7: Blocked (Tiled) Parallel I Code
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def v07_blocked_scalar(X, C, a, b, u):
    n, k = X.shape
    nb = (n + BS - 1) // BS
    for bi in prange(nb):
        is_ = bi * BS
        ie = min(n, is_ + BS)
        for bj in range(nb):
            if (u and bj < bi) or ((not u) and bj > bi):
                continue
            js = bj * BS
            je = min(n, js + BS)
            for i in range(is_, ie):
                for j in range(max(i, js) if u else js, je if u else min(i + 1, je)):
                    s = 0.0
                    for lb in range(0, k, BS):
                        for l in range(lb, min(lb + BS, k)):
                            s += X[i, l] * X[j, l]
                    C[i, j] = a * s + b * C[i, j]


# ---------------------------------------------------------
# Baseline 8: Remove scalar K blocking (V07 + one change)
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def v08_full_k_scalar(X, C, a, b, u):
    n, k = X.shape
    nb = (n + BS - 1) // BS
    for bi in prange(nb):
        is_ = bi * BS
        ie = min(n, is_ + BS)
        for bj in range(nb):
            if (u and bj < bi) or ((not u) and bj > bi):
                continue
            js = bj * BS
            je = min(n, js + BS)
            for i in range(is_, ie):
                for j in range(max(i, js) if u else js, je if u else min(i + 1, je)):
                    s = 0.0
                    for l in range(k):
                        s += X[i, l] * X[j, l]
                    C[i, j] = a * s + b * C[i, j]


# Map a flat task ID to its corresponding triangular block coordinates.
# Example (nb=4): result C decomposed to 4*4 blocks, for upper case
# 0 1 2 3
# x 4 5 6
# x x 7 8
# x x x 9
# task=2 -> (0, 2)
# task=7 -> (2, 2)
@njit
def triangular_task_coords(task, nb):
    row = int((2 * nb + 1 - np.sqrt((2 * nb + 1)**2 - 8 * task)) / 2)
    col = row + task - row * (2 * nb - row + 1) // 2
    return row, col


# ---------------------------------------------------------
# Baseline 9: Balanced flat triangular tasks (V08 + one change)
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def v09_flat_tri_tasks(X, C, a, b, u):
    n, k = X.shape
    nb = (n + BS - 1) // BS
    nt = nb * (nb + 1) // 2  # total number of blocks from C
    for task in prange(nt):
        br, bc = triangular_task_coords(task, nb)
        bi = int(br if u else bc)
        bj = int(bc if u else br)
        is_ = bi * BS
        ie = min(n, is_ + BS)
        js = bj * BS
        je = min(n, js + BS)
        for i in range(is_, ie):
            for j in range(max(i, js) if u else js, je if u else min(i + 1, je)):
                s = 0.0
                for l in range(k):
                    s += X[i, l] * X[j, l]
                C[i, j] = a * s + b * C[i, j]


# ---------------------------------------------------------
# Baseline 10: Full-K block np.dot (V09 + one change)
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def v10_blocked_dot(X, C, a, b, u):
    n, k = X.shape
    nb = (n + BS - 1) // BS
    nt = nb * (nb + 1) // 2
    for task in prange(nt):
        br, bc = triangular_task_coords(task, nb)
        bi = int(br if u else bc)
        bj = int(bc if u else br)
        is_ = bi * BS
        ie = min(n, is_ + BS)
        js = bj * BS
        je = min(n, js + BS)
        tmp = np.dot(X[is_:ie, :], X[js:je, :].T)
        for i in range(is_, ie):
            for j in range(max(i, js) if u else js, je if u else min(i + 1, je)):
                C[i, j] = a * tmp[i - is_, j - js] + b * C[i, j]


# ---------------------------------------------------------
# Baseline 11: Diagonal scalar fast path (V10 + one change)
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def v11_diagonal_fastpath(X, C, a, b, u):
    n, k = X.shape
    nb = (n + BS - 1) // BS
    nt = nb * (nb + 1) // 2
    for task in prange(nt):
        br, bc = triangular_task_coords(task, nb)
        bi = int(br if u else bc)
        bj = int(bc if u else br)
        is_ = bi * BS
        ie = min(n, is_ + BS)
        js = bj * BS
        je = min(n, js + BS)
        if bi == bj:
            for i in range(is_, ie):
                for j in range(i, je) if u else range(js, i + 1):
                    s = 0.0
                    for l in range(k):
                        s += X[i, l] * X[j, l]
                    C[i, j] = a * s + b * C[i, j]
        else:
            tmp = np.dot(X[is_:ie, :], X[js:je, :].T)
            for i in range(is_, ie):
                for j in range(js, je):
                    C[i, j] = a * tmp[i - is_, j - js] + b * C[i, j]


# ---------------------------------------------------------
# Baseline 12: Temporary C copy-in/out (V11 + one change)
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def v12_temp_copy(X, C, a, b, u):
    n, k = X.shape
    nb = (n + BS - 1) // BS
    nt = nb * (nb + 1) // 2
    for task in prange(nt):
        br, bc = triangular_task_coords(task, nb)
        bi = int(br if u else bc)
        bj = int(bc if u else br)
        is_ = bi * BS
        ie = min(n, is_ + BS)
        js = bj * BS
        je = min(n, js + BS)
        if bi == bj:
            # Keep V10's diagonal fast path: calculate only the selected triangle.
            temp = C[is_:ie, js:je].copy()
            for i in range(is_, ie):
                for j in range(i, je) if u else range(js, i + 1):
                    s = 0.0
                    for l in range(k):
                        s += X[i, l] * X[j, l]
                    temp[i - is_, j - js] = a * s + b * temp[i - is_, j - js]
            for i in range(is_, ie):
                for j in range(i, je) if u else range(js, i + 1):
                    C[i, j] = temp[i - is_, j - js]
        else:
            # Off-diagonal: copy C in, compute full-K dot, then copy out.
            temp = C[is_:ie, js:je].copy()
            temp *= b
            temp += a * np.dot(X[is_:ie, :], X[js:je, :].T)
            for i in range(is_, ie):
                for j in range(js, je):
                    C[i, j] = temp[i - is_, j - js]


# ---------------------------------------------------------
# Baseline 13: Two-level output tiling (V11 + one change)
# Branch directly from V11: no temporary C copy-in/out.
# Keep V11's diagonal scalar path and full-K dot.
# Only subdivide off-diagonal output tiles into L1 subtiles.
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def v13_two_level(X, C, a, b, u):
    n, k = X.shape
    nb = (n + L2 - 1) // L2
    nt = nb * (nb + 1) // 2
    for task in prange(nt):
        br, bc = triangular_task_coords(task, nb)
        bi = int(br if u else bc)
        bj = int(bc if u else br)
        is2 = bi * L2
        ie2 = min(n, is2 + L2)
        js2 = bj * L2
        je2 = min(n, js2 + L2)
        if bi == bj:
            # V11 diagonal scalar fast path: update C directly.
            for i in range(is2, ie2):
                for j in range(i, je2) if u else range(js2, i + 1):
                    s = 0.0
                    for l in range(k):
                        s += X[i, l] * X[j, l]
                    C[i, j] = a * s + b * C[i, j]
        else:
            # V11 full-K dot and direct C update, with L1 output subtiles.
            for is1 in range(is2, ie2, L1):
                ie1 = min(ie2, is1 + L1)
                for js1 in range(js2, je2, L1):
                    je1 = min(je2, js1 + L1)
                    tmp = np.dot(X[is1:ie1, :], X[js1:je1, :].T)
                    for i in range(is1, ie1):
                        for j in range(js1, je1):
                            C[i, j] = a * tmp[i - is1, j - js1] + b * C[i, j]


# 14 Reference BLAS: SciPy DSYRK updates only the selected triangle.
def v14_blas_dsyrk(X, C, a, b, u):
    C[:, :] = dsyrk(alpha=a, a=np.asfortranarray(X), beta=b,
                    c=np.array(C, order='F', copy=True),
                    trans=0, lower=0 if u else 1, overwrite_c=1)


FUNCTIONS = [
    ('00_python_naive', v00_python), ('01_numba_naive', v01_numba),
    ('02_ijl', v02_ijl), ('02_ilj', v02_ilj), ('02_jil', v02_jil),
    ('02_jli', v02_jli), ('02_lij', v02_lij), ('02_lji', v02_lji),
    ('03_unroll4_OWN', v03_unroll4),
    ('04_unroll4_fastmath_OWN', v04_unroll4_fastmath),
    ('05_fastmath', v05_fastmath),
    ('06_parallel_i', v06_parallel_i), ('06_parallel_j', v06_parallel_j),
    ('06_parallel_l', v06_parallel_l), ('07_blocked_scalar', v07_blocked_scalar),
    ('08_full_k_scalar_OWN', v08_full_k_scalar), ('09_flat_tri_tasks_OWN', v09_flat_tri_tasks),
    ('10_blocked_dot', v10_blocked_dot),
    ('11_diag_fastpath_OWN', v11_diagonal_fastpath),
    ('12_temp_copy', v12_temp_copy), ('13_two_level_OWN', v13_two_level),
    ('14_blas_dsyrk', v14_blas_dsyrk),
]


def prepare(A, C, trans='N', uplo='U'):
    if not isinstance(trans, str) or trans.upper() not in ('N', 'T', 'C'):
        raise ValueError('trans must be N, T, or C')
    if not isinstance(uplo, str) or uplo.upper() not in ('U', 'L'):
        raise ValueError('uplo must be U or L')
    A = np.asarray(A)
    if A.dtype != np.float64 or C.dtype != np.float64:
        raise TypeError('This real SYRK benchmark requires float64 arrays')
    if A.ndim != 2 or C.ndim != 2:
        raise ValueError('A and C must be 2-D')
    X = A if trans.upper() == 'N' else A.T
    if C.shape != (X.shape[0], X.shape[0]):
        raise ValueError('C shape must be (n,n) for op(A) shape (n,k)')
    if not C.flags.writeable:
        raise ValueError('C must be writable')
    if np.shares_memory(A, C):
        raise ValueError('A and C may not alias in this implementation')
    return X, uplo.upper() == 'U'


def syrk(A, C, alpha=1.0, beta=0.0, trans='N', uplo='U', method='13_two_level_OWN'):
    """BLAS-style real SYRK, updates C in place; returns C."""
    X, u = prepare(A, C, trans, uplo)
    try:
        fn = dict(FUNCTIONS)[method]
    except KeyError:
        raise ValueError('unknown method: ' + str(method)) from None
    fn(X, C, float(alpha), float(beta), u)
    return C


def reference(X, C0, alpha, beta, u):
    out = C0.copy()
    v14_blas_dsyrk(X, out, alpha, beta, u)
    return out


def test_all(verbose=True):
    set_num_threads(4)
    # Odd, non-tile-aligned and rectangular shapes; exact untouched triangle test.
    cases = [(1, 3), (5, 7), (17, 9), (33, 35), (65, 13)]
    for n, k in cases:
        rng = np.random.default_rng(200 + n)
        for u in (True, False):
            for trans in ('N', 'T'):
                X0 = rng.normal(size=(n, k))
                A = X0 if trans == 'N' else X0.T.copy()
                C0 = rng.normal(size=(n, n))
                X = X0
                expected = reference(X, C0, 1.25, -0.4, u)
                opposite = np.tril_indices(n, -1) if u else np.triu_indices(n, 1)
                for name, fn in FUNCTIONS:
                    if name == '00_python_naive' and n > 33: continue
                    if name == '06_parallel_l' and n > 1: continue  # excessive per-dot nested scheduling
                    got = C0.copy()
                    syrk(A, got, 1.25, -0.4, trans, 'U' if u else 'L', name)
                    if not np.array_equal(got[opposite], C0[opposite]):
                        raise AssertionError(f'{name} wrote the inactive triangle at {(n, k, u, trans)}')
                    if not np.allclose(got, expected, rtol=2e-9, atol=2e-9):
                        err = np.max(np.abs(got - expected))
                        raise AssertionError(f'{name} incorrect at {(n, k, u, trans)}; max err {err}')
        if verbose: print(f'PASS n={n}, k={k}, both UPLO and TRANS')
    if verbose: print('ALL TESTS PASS')


# ---------------------------------------------------------
# Execution & Benchmarking Routine (professor's original structure)
# ---------------------------------------------------------
def run_benchmark(matrix_size=(512, 256), alpha=1.25, beta=-0.4,
                  trans='N', uplo='U', threads=4, reps=9):
    global N, K
    N, K = matrix_size
    set_num_threads(threads)

    np.random.seed(42)
    X = np.random.randn(N, K).astype(np.float64)
    A = X if trans == 'N' else X.T.copy()
    C = np.random.randn(N, N).astype(np.float64)
    C_initial = C.copy()

    # Reference respects the input triangle, alpha/beta and transpose.
    C_expected = reference(X, C_initial, alpha, beta, uplo == 'U')
    total_flops = float(N * (N + 1) * K)

    def measure(fn, warmup=True, repetitions=reps):
        C[:, :] = C_initial
        if warmup:
            fn(A, C, alpha, beta, trans, uplo)

        # Correctness check against reference matrix C_expected
        C[:, :] = C_initial
        fn(A, C, alpha, beta, trans, uplo)
        is_correct = np.allclose(C, C_expected, rtol=1e-5, atol=1e-5)

        start = time.perf_counter()
        for _ in range(repetitions):
            C[:, :] = C_initial
            fn(A, C, alpha, beta, trans, uplo)
        elapsed = (time.perf_counter() - start) / repetitions
        gflops = (total_flops / elapsed) / 1e9
        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python, truncated like professor's example to avoid lockups.
    N_py, K_py = min(N, 128), min(K, 128)
    X_py = X[:N_py, :K_py].copy()
    A_py = X_py if trans == 'N' else X_py.T.copy()
    C_py = C_initial[:N_py, :N_py].copy()
    C_py_expected = reference(X_py, C_py, alpha, beta, uplo == 'U')

    t0 = time.perf_counter()
    syrk(A_py, C_py, alpha, beta, trans, uplo, '00_python_naive')
    py_elapsed = time.perf_counter() - t0
    py_gflops = (N_py * (N_py + 1) * K_py / py_elapsed) / 1e9
    py_correct = np.allclose(C_py, C_py_expected, rtol=1e-5, atol=1e-5)
    results.append(('00_python_naive', py_gflops, py_elapsed, py_correct))

    # Baseline functions list: professor techniques plus SYRK-specific methods,
    # deliberately interleaved to make each development step visible.
    functions = [(name, name) for name, _ in FUNCTIONS if name != '00_python_naive']
    for name, method in functions:
        def invoke(A_, C_, alpha_, beta_, trans_, uplo_, method_=method):
            return syrk(A_, C_, alpha_, beta_, trans_, uplo_, method_)

        gflops, elapsed, is_correct = measure(invoke)
        results.append((name, gflops, elapsed, is_correct))

    return results


# ---------------------------------------------------------
# Formatting and Output Execution (professor's original structure)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Original positional dimension input, extended to accept SYRK N and K.
    # Additional optional flags do not change the professor's chart/layout.
    parser = argparse.ArgumentParser()
    parser.add_argument('N', nargs='?', type=int, default=700)
    parser.add_argument('K', nargs='?', type=int, default=400)
    parser.add_argument('--alpha', type=float, default=1.25)
    parser.add_argument('--beta', type=float, default=-0.4)
    parser.add_argument('--trans', choices=['N', 'T'], default='N')
    parser.add_argument('--uplo', choices=['U', 'L'], default='U')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reps', type=int, default=9)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    if args.test:
        test_all()
        sys.exit(0)
    NK_input = (args.N, args.K)
    print(f"\nRunning SYRK Benchmarks for Matrix Size {NK_input} "
          f"(UPLO={args.uplo}, TRANS={args.trans}, alpha={args.alpha}, beta={args.beta})...\n")
    benchmark_data = run_benchmark(NK_input, args.alpha, args.beta, args.trans,
                                   args.uplo, args.threads, args.reps)
    py_elapsed = benchmark_data[0][2]

    # Table Header Formatting - professor original
    header = f"| {'Baseline Implementation':<33} | {'GFLOP/s':<10} | {'Abs Speedup':<12} | {'Rel Speedup':<12} | {'Correct':<8} |"
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)
    prev_elapsed = None
    for name, gflops, elapsed, is_correct in benchmark_data:
        abs_speedup = py_elapsed / elapsed
        rel_speedup = (prev_elapsed / elapsed) if prev_elapsed is not None else 1.0
        status = "PASS" if is_correct else "FAIL"
        print(f"| {name:<33} | {gflops:10.3f} | {abs_speedup:12.2f}x | {rel_speedup:12.2f}x | {status:<8} |")
        prev_elapsed = elapsed
    print(divider)
    print('Note: Pure Python uses capped dimensions (N,K <= 128); Abs Speedup')
    print('is not a same-size comparison if either input dimension exceeds 128.')

    # Plotting Output - professor original, intentionally retained.
    names = [row[0] for row in benchmark_data]
    gflops_vals = [row[1] for row in benchmark_data]
    plt.figure(figsize=(16, 6))
    bars = plt.bar(names, gflops_vals, color="skyblue", edgecolor="navy")
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (0.02 * max(gflops_vals)),
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.ylabel("GFLOP/s (Higher is better)")
    plt.title(f"Numba SYRK Benchmark Performance (N={args.N}, K={args.K})")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    if not args.no_plot:
        plt.show()

