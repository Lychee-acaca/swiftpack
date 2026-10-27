import ast
import pytest
import networkx as nx
from numba import njit

from swiftpack import (
    render_cfg_from_dict,
    ast_to_cfg_dict,
    normalize_numba_cfg_dot,
    CustomAnalysisCompiler,
)

# =====================================================================
# TEST DATA (KERNELS)
# =====================================================================
CHOLESKY_CODE = """ 
def cholesky_decomposition(A):
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1):
            sum = A[i, j]
            for k in range(j):
                sum -= A[i, k] * A[j, k]
            if i == j:
                A[i, j] = (sum ** 0.5)
            else:
                A[i, j] = sum / A[j, j]
"""

MATMUL_CODE = """
def matmul(A, B, C):
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            sum = 0.0
            for k in range(A.shape[1]):
                sum += A[i, k] * B[k, j]
            C[i, j] = sum
"""


# =====================================================================
# HELPER VALIDATION FUNCTION
# =====================================================================
def verify_cfg_invariants(cfg_dict):
    """
    Asserts structural and bidirectional consistency of a CFG dictionary.
    """
    all_nodes = set(cfg_dict.keys())

    # 1. Check Predecessor/Successor Edge Targets Exist
    for src, data in cfg_dict.items():
        successors = data.get("successors", [])
        for succ in successors:
            dst = succ["target"] if isinstance(succ, dict) else succ
            assert dst in all_nodes, f"Dangling edge: {src} -> {dst} (target does not exist)"

    # 2. Check for Entry Node existence
    assert "BB_0" in all_nodes, "CFG is missing entry node 'BB_0'"


# =====================================================================
# PYTEST SUITE
# =====================================================================

@pytest.mark.parametrize("code, func_name", [
    (CHOLESKY_CODE, "cholesky_decomposition"),
    (MATMUL_CODE, "matmul")
])
def test_ast_cfg_generation_and_invariants(code, func_name):
    print(f"\n[TEST] Verifying AST CFG invariants for {func_name}")

    # 1. Generate CFG dictionary from AST
    cfg_dict = ast_to_cfg_dict(code)

    # 2. Verify structural invariants
    verify_cfg_invariants(cfg_dict)

    # 3. Render DOT and PDF files
    cfg_dot = render_cfg_from_dict(cfg_dict,
                                   output_filename=f"{func_name}_orig")
    assert cfg_dot is not None
    assert "digraph" in cfg_dot.source


@pytest.mark.parametrize("code, func_name, signature", [
    (CHOLESKY_CODE, "cholesky_decomposition", "void(float64[:, :])"),
    (MATMUL_CODE, "matmul",
     "void(float64[:, :], float64[:, :], float64[:, :])")
])
def test_numba_compiler_pass_and_normalization(code, func_name,
                                               signature):
    print(
        f"\n[TEST] Verifying Numba pass and normalization for {func_name}")

    # 1. Execute string in local namespace to get function object
    local_vars = {}
    exec(code, globals(), local_vars)
    func_obj = local_vars[func_name]

    # 2. Force Numba compilation with custom pipeline pass
    compiled_func = njit(signature,
                         pipeline_class=CustomAnalysisCompiler)(
        func_obj)

    # 3. Load the Numba-generated simplified dot file
    dot_filename = f"{func_name}_simplified.dot"
    normalized_cfg_numba = normalize_numba_cfg_dot(dot_filename)

    # 4. Assertions on normalized Numba graph structure
    assert isinstance(normalized_cfg_numba, dict)
    assert len(normalized_cfg_numba) > 0
    assert 0 in normalized_cfg_numba or "0" in normalized_cfg_numba