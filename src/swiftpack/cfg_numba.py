from numba.core import ir_utils
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.untyped_passes import IRProcessing
from numba import njit

import subprocess
import re


def normalize_numba_cfg_dot(dot_input):
    """
    Takes a Numba CFG DOT representation (file path, DOT string, or graphviz Digraph object),
    normalizes the node identifiers to start from 0 (0, 1, 2, ...), and returns
    the graph as a structured dictionary.
    """
    # 1. Obtain the DOT string content
    if hasattr(dot_input, 'source'):  # graphviz.Digraph object
        dot_str = dot_input.source
    elif isinstance(dot_input, str):
        try:
            with open(dot_input, 'r', encoding='utf-8') as f:
                dot_str = f.read()
        except OSError:
            dot_str = dot_input  # Treat as raw DOT string
    else:
        raise TypeError(
            "Input must be a DOT string, file path, or graphviz Digraph object.")

    # 2. Extract node labels and attributes
    node_labels = {}
    node_pattern = re.compile(r'^\s*([a-zA-Z0-9_"\.-]+)\s*\[(.*?)\]',
                              re.MULTILINE)
    for match in node_pattern.finditer(dot_str):
        node_id = match.group(1).strip('"')
        attrs = match.group(2)
        # Extract label if present
        label_match = re.search(
            r'label\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,\s]+)',
            attrs)
        if label_match:
            lbl = label_match.group(1).strip('"\'')
            lbl = lbl.replace('\\n', '\n').replace('\\l', '\n').replace(
                '\\r', '\n')
            node_labels[node_id] = lbl
        else:
            node_labels[node_id] = ""

    # 3. Extract edges
    edge_pattern = re.compile(
        r'^\s*([a-zA-Z0-9_"\.-]+)\s*->\s*([a-zA-Z0-9_"\.-]+)',
        re.MULTILINE)
    raw_edges = []
    nodes_set = set(node_labels.keys())

    for match in edge_pattern.finditer(dot_str):
        src = match.group(1).strip('"')
        dst = match.group(2).strip('"')
        raw_edges.append((src, dst))
        nodes_set.add(src)
        nodes_set.add(dst)

    # 4. Sort and create mapping to normalize node IDs starting from 0
    def sort_key(n):
        nums = re.findall(r'\d+', n)
        if nums:
            return (0, int(nums[-1]), n)
        return (1, 0, n)

    sorted_nodes = sorted(list(nodes_set), key=sort_key)
    old_to_new = {old_id: idx for idx, old_id in
                  enumerate(sorted_nodes)}
    new_to_old = {idx: old_id for old_id, idx in old_to_new.items()}

    # 5. Build the normalized graph dictionary
    graph_dict = {
        idx: {
            "original_id": new_to_old[idx],
            "label": node_labels.get(new_to_old[idx], ""),
            "successors": [],
            "predecessors": []
        }
        for idx in range(len(sorted_nodes))
    }

    # Populate edges
    for src, dst in raw_edges:
        new_src = old_to_new[src]
        new_dst = old_to_new[dst]
        if new_dst not in graph_dict[new_src]["successors"]:
            graph_dict[new_src]["successors"].append(new_dst)
        if new_src not in graph_dict[new_dst]["predecessors"]:
            graph_dict[new_dst]["predecessors"].append(new_src)

    # Sort neighbor lists for determinism
    for idx in graph_dict:
        graph_dict[idx]["successors"].sort()
        graph_dict[idx]["predecessors"].sort()

    return graph_dict


@register_pass(mutates_CFG=False, analysis_only=True)
class VisualizeCFGPass(FunctionPass):
    _name = "visualize_cfg_pass"

    def __init__(self):
        super().__init__(self)

    def run_pass(self, state):
        # state contains the FunctionIR to be mutated,
        # along with environment and typing information
        mutate = False
        func_ir = state.func_ir
        func_name = state.func_id.func_qualname

        # Create CFG from the IR blocks
        cfg = ir_utils.compute_cfg_from_blocks(func_ir.blocks)

        # Simplify the CFG blocks
        cfg_simplified_blks = ir_utils.simplify_CFG(func_ir.blocks)
        cfg_simplified = ir_utils.compute_cfg_from_blocks(
            cfg_simplified_blks)

        print("--- Function IR Blocks ---")
        print(func_ir.blocks)
        print("=====================================")

        # Visualize and save the CFG into DOT format files
        try:
            cfg.render_dot().save(f'{func_name}.dot')
            cfg_simplified.render_dot().save(f'{func_name}_simplified.dot')
            # make pdf from dot file
            subprocess.run(['dot', '-Tpdf', f'{func_name}.dot', '-o', f'{func_name}.pdf'])
            subprocess.run(['dot', '-Tpdf', f'{func_name}_simplified.dot', '-o', f'{func_name}_simplified.pdf'])
            print(f"CFG successfully rendered and saved to '{func_name}.dot' and '{func_name}_simplified.dot'.")
            normalized_cfg = normalize_numba_cfg_dot(cfg_simplified.render_dot())
            print("Normalized Graph Dictionary:")
            print(normalized_cfg)
        except Exception as e:
            print(
                f"Note: Graphviz rendering skipped or failed due to missing system dependency: {e}")

        return mutate




# 1. Define a Custom Compiler using CompilerBase
class CustomAnalysisCompiler(CompilerBase):
    def define_pipelines(self):
        # Initialize the default nopython mode pass manager
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)

        # 2. Insert your custom pass to run immediately after IR processing
        pm.add_pass_after(VisualizeCFGPass, IRProcessing)

        pm.finalize()
        return [pm]





if __name__ == "__main__":
    # 3. Pass your custom compiler class to the @njit decorator
    @njit(pipeline_class=CustomAnalysisCompiler)
    def example_func(x):
        if x > 0:
            y = x * 2
        else:
            y = x - 2
        return y

    # Trigger compilation by calling the function
    res = example_func(5)
    print("Result:", res)