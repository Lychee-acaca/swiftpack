import ast
from swiftpack.cfg import render_cfg_from_dict, ast_to_cfg_dict
from swiftpack.lvn import LocalAnalysis

if __name__ == "__main__":
    snippets = {
        "Constant Folding & Copy Propagation": """
def compute_metrics(x):
    a = 10
    b = 20
    c = a + b
    d = c + 0
    e = d * 1
    return e
""",
        "Algebraic Simplification": """
def scale_vector(val):
    base = val * 0
    offset = 5
    result = base + offset
    return result
""",
        "Conditional Clamping": """
def clamp_value(x):
    limit = 100
    if x > limit:
        y = 100
    else:
        y = x
    return y
""",
        "Multi-Way Sign Computation": """
def compute_sign(a, b):
    sum_val = a + b
    if sum_val > 0:
        res = 1
    elif sum_val < 0:
        res = -1
    else:
        res = 0
    return res
"""
    }

    for title, code_str in snippets.items():
        print("=" * 70)
        print(f"PIPELINE TEST: {title}")
        print("=" * 70)

        print("\n--- Original Source Code ---")
        print(code_str.strip())

        tree = ast.parse(code_str)

        # Generate Original CFG
        orig_cfg = ast_to_cfg_dict(tree)
        orig_pdf = render_cfg_from_dict(orig_cfg,
                                        output_filename=f"{title.lower().replace(' ', '_')}_original")
        print(f" -> Original CFG PDF: {orig_pdf}")

        # Apply LVN + DAE Optimization
        optimizer = LocalAnalysis()
        optimized_tree = optimizer.apply(tree)

        # Generate Optimized CFG
        opt_cfg = ast_to_cfg_dict(optimized_tree)
        opt_pdf = render_cfg_from_dict(opt_cfg,
                                       output_filename=f"{title.lower().replace(' ', '_')}_optimized")
        print(f" -> Optimized CFG PDF: {opt_pdf}")

        # Print Final Optimized Code
        print("\n--- Optimized Code ---")
        print(ast.unparse(optimized_tree))
        print("\n")