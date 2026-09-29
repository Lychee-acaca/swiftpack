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

# ---------------------------------------------------------
# Baseline 0: Pure Python Naive SYRK
# ---------------------------------------------------------
def syrk_python_0(A, C):
    n, k = A.shape
    for i in range(n):
        for j in range(i, n):
            C[i, j] = 0.0
            for l in range(k):
                C[i, j] += A[i, l] * A[j, l]
            C[j, i] = C[i, j]

# ---------------------------------------------------------
# Baseline 1: Naive SYRK in Numba
# ---------------------------------------------------------
@njit
def syrk_numba_1(A, C):
    n, k = A.shape
    for i in range(n):
        for j in range(i, n):
            C[i, j] = 0.0
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
# Baseline 10: Reference NumPy dot
# ---------------------------------------------------------
def syrk_np_dot(A, C):
    np.copyto(C, np.dot(A, A.T))

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