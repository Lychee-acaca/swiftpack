import os
import sys
import time
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np

# Set single-threaded execution for external BLAS libraries to ensure reproducible benchmarks
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Default Matrix Dimensions
N = 512
K = 256
BLOCK_SIZE = 32
BLOCK_SIZE_L2 = 128
BLOCK_SIZE_L1 = 32

# ---------------------------------------------------------
# Baseline 0: Pure Python Naive SYRK
# ---------------------------------------------------------
def syrk_python_0(A, C):
    n, k = A.shape
    C.fill(0.0)
    # A * A', selecting row i and row j
    for i in range(n):
        for j in range(i, n):
            for l in range(k):
                C[i, j] += A[i, l] * A[j, l]
            C[j, i] = C[i, j]

# ---------------------------------------------------------
# Baseline 1: Naive SYRK in Numba
# ---------------------------------------------------------
@njit
def syrk_numba_1(A, C):
    n, k = A.shape
    C.fill(0.0)
    for i in range(n):
        for j in range(i, n):
            for l in range(k):
                C[i, j] += A[i, l] * A[j, l]
            C[j, i] = C[i, j]

# ---------------------------------------------------------
# Baseline 2: Loop Order Permutations (6 explicit functions)
# ---------------------------------------------------------
@njit
def syrk_2_ijl(A, C):
    n, k = A.shape
    C.fill(0.0)
    for i in range(n):
        for j in range(i, n):
            for l in range(k):
                C[i, j] += A[i, l] * A[j, l]
    for i in range(n):
        for j in range(i + 1, n):
            C[j, i] = C[i, j]

@njit
def syrk_2_ilj(A, C):
    n, k = A.shape
    C.fill(0.0)
    for i in range(n):
        for l in range(k):
            for j in range(i, n):
                C[i, j] += A[i, l] * A[j, l]
    for i in range(n):
        for j in range(i + 1, n):
            C[j, i] = C[i, j]

@njit
def syrk_2_jil(A, C):
    n, k = A.shape
    C.fill(0.0)
    for j in range(n):
        for i in range(j + 1):
            for l in range(k):
                C[i, j] += A[i, l] * A[j, l]
    for i in range(n):
        for j in range(i + 1, n):
            C[j, i] = C[i, j]

@njit
def syrk_2_jli(A, C):
    n, k = A.shape
    C.fill(0.0)
    for j in range(n):
        for l in range(k):
            for i in range(j + 1):
                C[i, j] += A[i, l] * A[j, l]
    for i in range(n):
        for j in range(i + 1, n):
            C[j, i] = C[i, j]

@njit
def syrk_2_lij(A, C):
    n, k = A.shape
    C.fill(0.0)
    for l in range(k):
        for i in range(n):
            for j in range(i, n):
                C[i, j] += A[i, l] * A[j, l]
    for i in range(n):
        for j in range(i + 1, n):
            C[j, i] = C[i, j]

@njit
def syrk_2_lji(A, C):
    n, k = A.shape
    C.fill(0.0)
    for l in range(k):
        for j in range(n):
            for i in range(j + 1):
                C[i, j] += A[i, l] * A[j, l]
    for i in range(n):
        for j in range(i + 1, n):
            C[j, i] = C[i, j]

# ---------------------------------------------------------
# Baseline 3: Optimization Flags for Backend
# ---------------------------------------------------------
@njit(fastmath=True)
def syrk_opt_flags_3(A, C):
    n, k = A.shape
    C.fill(0.0)
    for i in range(n):
        for j in range(i, n):
            for l in range(k):
                C[i, j] += A[i, l] * A[j, l]
            C[j, i] = C[i, j]

# ---------------------------------------------------------
# Baseline 4: Parallel Loop Versions
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def syrk_parallel_i_4(A, C):
    n, k = A.shape
    C.fill(0.0)
    for i in prange(n):
        for j in range(i, n):
            for l in range(k):
                C[i, j] += A[i, l] * A[j, l]
            C[j, i] = C[i, j]

@njit(parallel=True, fastmath=True)
def syrk_parallel_j_4(A, C):
    n, k = A.shape
    C.fill(0.0)
    for i in range(n):
        for j in prange(i, n):
            for l in range(k):
                C[i, j] += A[i, l] * A[j, l]
            C[j, i] = C[i, j]

@njit(parallel=True, fastmath=True)
def syrk_parallel_l_4(A, C):
    n, k = A.shape
    C.fill(0.0)
    for i in range(n):
        for j in range(i, n):
            # race condition here if using shared C[i, j]
            s = 0.0
            for l in prange(k):
                s += A[i, l] * A[j, l]
            C[i, j] = s
            C[j, i] = C[i, j]

# ---------------------------------------------------------
# Baseline 5: Blocked (Tiled) Parallel I Code
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def syrk_blocked_parallel_5(A, C):
    n, k = A.shape
    bs = BLOCK_SIZE
    C.fill(0.0)

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)

        for j_block in range(i_block, n, bs):
            j_end = min(j_block + bs, n)
            for l_block in range(0, k, bs):
                l_end = min(l_block + bs, k)

                for i in range(i_block, i_end):
                    for j in range(max(i, j_block), j_end):
                        for l in range(l_block, l_end):
                            C[i, j] += A[i, l] * A[j, l]

            for i in range(i_block, i_end):
                for j in range(max(i, j_block), j_end):
                    C[j, i] = C[i, j]

# ---------------------------------------------------------
# Baseline 6: Blocked Parallel I using np.dot for Sub-blocks
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def syrk_blocked_np_dot_6(A, C):
    n, k = A.shape
    bs = BLOCK_SIZE
    C.fill(0.0)

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)

        for j_block in range(i_block, n, bs):
            j_end = min(j_block + bs, n)

            for l_block in range(0, k, bs):
                l_end = min(l_block + bs, k)

                C[i_block:i_end, j_block:j_end] += np.dot(
                    A[i_block:i_end, l_block:l_end],
                    A[j_block:j_end, l_block:l_end].T
                )

            # Copy upper triangle to lower triangle
            for i in range(i_block, i_end):
                for j in range(max(i, j_block), j_end):
                    C[j, i] = C[i, j]


# ---------------------------------------------------------
# Baseline 7: Blocked Temp Copy-In / Copy-Out
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def syrk_blocked_temp_copy_7(A, C):
    n, k = A.shape
    bs = BLOCK_SIZE
    C.fill(0.0)

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        for j_block in range(i_block, n, bs):
            j_end = min(j_block + bs, n)

            temp = C[i_block:i_end, j_block:j_end].copy()
            for l_block in range(0, k, bs):
                l_end = min(l_block + bs, k)

                temp += np.dot(
                    A[i_block:i_end, l_block:l_end],
                    A[j_block:j_end, l_block:l_end].T
                )
            C[i_block:i_end, j_block:j_end] = temp

            for i in range(i_block, i_end):
                for j in range(max(i, j_block), j_end):
                    C[j, i] = C[i, j]


# ---------------------------------------------------------
# Baseline 8: Two-Level Blocked Parallel I with Temp Copy & np.dot
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def syrk_two_level_blocked_temp_np_dot_8(A, C):
    n, k = A.shape
    l2 = BLOCK_SIZE_L2
    l1 = BLOCK_SIZE_L1
    C.fill(0.0)

    num_l2_i_blocks = (n + l2 - 1) // l2

    for b2 in prange(num_l2_i_blocks):
        i2_start = b2 * l2
        i2_end = min(i2_start + l2, n)

        for j2_start in range(i2_start, n, l2):
            j2_end = min(j2_start + l2, n)

            temp_l2 = C[i2_start:i2_end,j2_start:j2_end].copy()
            for k2_start in range(0, k, l2):
                k2_end = min(k2_start + l2, k)
                for i1_start in range(i2_start, i2_end, l1):
                    i1_end = min(i1_start + l1, i2_end)

                    i1_rel_start = i1_start - i2_start
                    i1_rel_end = i1_end - i2_start
                    j1_start = max(i1_start, j2_start)

                    for j1_start in range(j1_start, j2_end, l1):
                        j1_end = min(j1_start + l1, j2_end)
                        if j1_end <= i1_start:
                            continue
                        j1_rel_start = j1_start - j2_start
                        j1_rel_end = j1_end - j2_start
                        temp_l1 = temp_l2[
                            i1_rel_start:i1_rel_end,
                            j1_rel_start:j1_rel_end
                        ].copy()

                        for k1_start in range(k2_start, k2_end, l1):
                            k1_end = min(k1_start + l1, k2_end)
                            temp_l1 += np.dot(
                                A[i1_start:i1_end,k1_start:k1_end],
                                A[j1_start:j1_end,k1_start:k1_end].T
                            )
                        temp_l2[i1_rel_start:i1_rel_end,j1_rel_start:j1_rel_end] = temp_l1

            C[i2_start:i2_end,j2_start:j2_end] = temp_l2

            for i in range(i2_start, i2_end):
                for j in range(max(i, j2_start), j2_end):
                    C[j, i] = C[i, j]

# ---------------------------------------------------------
# Baseline 10: Reference NumPy dot
# ---------------------------------------------------------
def syrk_np_dot(A, C):
    # not fair
    np.copyto(C, np.dot(A, A.T))
    # n, k = A.shape
    # C.fill(0.0)
    # for i in range(n):
    #     for j in range(i, n):
    #         C[i, j] = np.dot(A[i, :], A[j, :])
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         C[j, i] = C[i, j]

# ---------------------------------------------------------
# Execution & Benchmarking Routine
# ---------------------------------------------------------
def run_benchmark(matrix_size=(512, 256)):
    global N, K
    N, K = matrix_size

    np.random.seed(42)
    A = np.random.randn(N, K).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)

    # Compute ground truth reference solution for correctness verification
    C_expected = np.dot(A, A.T)
    total_flops = 2.0 * (N * (N + 1) / 2 * K)

    def measure(fn, warmup=True, reps=9):
        C.fill(0.0)
        if warmup:
            fn(A, C)

        # Correctness check against reference matrix C_expected
        C.fill(0.0)
        fn(A, C)
        is_correct = np.allclose(C, C_expected, rtol=1e-5, atol=1e-5)

        start = time.perf_counter()
        for _ in range(reps):
            C.fill(0.0)
            fn(A, C)
        elapsed = (time.perf_counter() - start) / reps
        gflops = (total_flops / elapsed) / 1e9

        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python (Run on a smaller dimension if N is large to prevent lockups)
    N_py = min(N, 128)
    K_py = min(K, 128)
    A_py, C_py = A[:N_py, :K_py].copy(), np.zeros((N_py, N_py), dtype=np.float64)
    C_py_expected = np.dot(A_py, A_py.T)

    t0 = time.perf_counter()
    syrk_python_0(A_py, C_py)
    py_elapsed = time.perf_counter() - t0
    py_gflops = (2.0 * (N_py * (N_py + 1) / 2 * K_py) / py_elapsed) / 1e9
    py_correct = np.allclose(C_py, C_py_expected, rtol=1e-5, atol=1e-5)

    results.append(("0_python_naive", py_gflops, py_elapsed, py_correct))

    # Baseline functions list
    functions = [
        ("1_numba_naive", syrk_numba_1),
        ("2_order_ijl", syrk_2_ijl),
        ("2_order_ilj", syrk_2_ilj),
        ("2_order_jil", syrk_2_jil),
        ("2_order_jli", syrk_2_jli),
        ("2_order_lij", syrk_2_lij),
        ("2_order_lji", syrk_2_lji),
        ("3_fastmath_ijl", syrk_opt_flags_3),
        ("4_parallel_i", syrk_parallel_i_4),
        ("4_parallel_j", syrk_parallel_j_4),
        ("4_parallel_l", syrk_parallel_l_4),
        ("5_blocked_parallel_i", syrk_blocked_parallel_5),
        ("6_blocked_np_dot", syrk_blocked_np_dot_6),
        ("7_blocked_temp_copy", syrk_blocked_temp_copy_7),
        ("8_two_level_blocked_temp_np_dot", syrk_two_level_blocked_temp_np_dot_8),
        ("10_np_dot", syrk_np_dot),
    ]

    for name, fn in functions:
        gflops, elapsed, is_correct = measure(fn)
        results.append((name, gflops, elapsed, is_correct))

    return results

# ---------------------------------------------------------
# Formatting and Output Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Get user input for matrix dimension N
    if len(sys.argv) > 2:
        try:
            NK_input = (int(sys.argv[1]), int(sys.argv[2]))
        except ValueError:
            NK_input = (512, 256)
    else:
        NK_input = (512, 256)

    print(f"\nRunning SYRK Benchmarks for Matrix Size {NK_input}...\n")
    benchmark_data = run_benchmark(NK_input)

    py_elapsed = benchmark_data[0][2]  # Reference execution time for pure Python naive

    # Table Header Formatting
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

    # Plotting Output
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
    plt.title(f"Numba SYRK Benchmark Performance (N, K={NK_input})")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()