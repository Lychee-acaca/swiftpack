import ast
import json
import graphviz


class BasicBlock:
    """Represents a basic block: a sequence of statements with single entry and exit."""

    def __init__(self, block_id):
        self.block_id = block_id
        self.statements = []
        self.successors = []

    def add_statement(self, stmt):
        self.statements.append(
            ast.unparse(stmt) if hasattr(ast, "unparse") else str(stmt)
        )

    def __repr__(self):
        return f"Block(BB_{self.block_id}, stmts={len(self.statements)}, succs={self.successors})"


class ASTToCFGBuilder:
    """Traverses Python AST statements to partition them into Basic Blocks and edges."""

    def __init__(self):
        self.blocks = {}
        self.block_counter = 0
        self.current_block = self._create_block()

    def _create_block(self):
        bid = self.block_counter
        self.block_counter += 1
        block = BasicBlock(bid)
        self.blocks[bid] = block
        return block

    def _add_edge(self, from_id, to_id, label=None):
        edge = (to_id, label) if label else to_id
        if edge not in self.blocks[from_id].successors:
            self.blocks[from_id].successors.append(edge)

    def build(self, ast_node):
        if isinstance(ast_node, ast.Module):
            for item in ast_node.body:
                if isinstance(item, ast.FunctionDef):
                    self._process_statements(item.body)
                else:
                    self._process_statements([item])
        elif isinstance(ast_node, ast.FunctionDef):
            self._process_statements(ast_node.body)
        elif isinstance(ast_node, list):
            self._process_statements(ast_node)
        else:
            self._process_statements([ast_node])

        active_block_ids = {
            bid
            for bid, block in self.blocks.items()
            if block.statements or block.successors
        }

        cfg_dict = {}
        for bid, block in self.blocks.items():
            if bid not in active_block_ids:
                continue

            valid_successors = []
            for s in block.successors:
                target_id = s[0] if isinstance(s, tuple) else s
                if target_id in active_block_ids:
                    if isinstance(s, tuple):
                        valid_successors.append(
                            {"target": f"BB_{s[0]}", "label": s[1]})
                    else:
                        valid_successors.append({"target": f"BB_{s}"})

            cfg_dict[f"BB_{bid}"] = {
                "statements": block.statements,
                "successors": valid_successors,
            }
        return cfg_dict

    def _process_statements(self, stmts):
        for stmt in stmts:
            if isinstance(stmt, ast.FunctionDef):
                self._process_statements(stmt.body)

            elif isinstance(stmt, ast.If):
                test_str = f"if {ast.unparse(stmt.test)}:"
                self.current_block.statements.append(test_str)

                header_id = self.current_block.block_id

                then_block = self._create_block()
                self._add_edge(header_id, then_block.block_id,
                               label="True")
                self.current_block = then_block
                self._process_statements(stmt.body)
                then_exit_id = self.current_block.block_id

                else_exit_id = None
                if stmt.orelse:
                    else_block = self._create_block()
                    self._add_edge(header_id, else_block.block_id,
                                   label="False")
                    self.current_block = else_block
                    self._process_statements(stmt.orelse)
                    else_exit_id = self.current_block.block_id

                join_block = self._create_block()
                self._add_edge(then_exit_id, join_block.block_id)
                if else_exit_id is not None:
                    self._add_edge(else_exit_id, join_block.block_id)
                else:
                    self._add_edge(header_id, join_block.block_id,
                                   label="False")

                self.current_block = join_block

            elif isinstance(stmt, (ast.For, ast.While)):
                header_block = self._create_block()
                self._add_edge(self.current_block.block_id,
                               header_block.block_id)

                loop_cond_str = (
                    f"while {ast.unparse(stmt.test)}:"
                    if isinstance(stmt, ast.While)
                    else f"for {ast.unparse(stmt.target)} in {ast.unparse(stmt.iter)}:"
                )
                header_block.statements.append(loop_cond_str)

                body_block = self._create_block()
                self._add_edge(
                    header_block.block_id, body_block.block_id,
                    label="True"
                )
                self.current_block = body_block
                self._process_statements(stmt.body)

                self._add_edge(
                    self.current_block.block_id,
                    header_block.block_id,
                    label="Back-edge",
                )

                exit_block = self._create_block()
                self._add_edge(
                    header_block.block_id, exit_block.block_id,
                    label="False"
                )
                self.current_block = exit_block

            elif isinstance(stmt, ast.Return):
                self.current_block.add_statement(stmt)
                terminal_block = self._create_block()
                self.current_block = terminal_block

            else:
                self.current_block.add_statement(stmt)


def ast_to_cfg_dict(source_code_or_ast):
    if isinstance(source_code_or_ast, str):
        tree = ast.parse(source_code_or_ast)
    else:
        tree = source_code_or_ast

    builder = ASTToCFGBuilder()
    return builder.build(tree)


def dict_to_normalized_dot(graph_dict, graph_name="NormalizedCFG"):
    dot = graphviz.Digraph(graph_name,
                           comment="Normalized Control Flow Graph")
    dot.attr(compound='true')

    sorted_nodes = sorted(graph_dict.keys(), key=lambda x: int(
        ''.join(filter(str.isdigit, str(x)))) if any(
        c.isdigit() for c in str(x)) else str(x))

    for node in sorted_nodes:
        node_data = graph_dict[node]
        statements = node_data.get("statements", [])

        if isinstance(statements, list):
            clean_lines = [str(stmt) for stmt in statements]
            dot_label = "".join([f"{line}\\l" for line in clean_lines])
        else:
            dot_label = f"{statements}\\l"

        dot.node(str(node), label=dot_label, shape='box')

    added_edges = set()
    for src_id, node_data in graph_dict.items():
        successors = node_data.get("successors", [])
        for succ in successors:
            dst_id = succ.get("target") if isinstance(succ,
                                                      dict) else succ
            edge_label = succ.get("label") if isinstance(succ,
                                                         dict) else None

            if dst_id:
                edge = (str(src_id), str(dst_id))
                if edge not in added_edges:
                    if edge_label:
                        dot.edge(edge[0], edge[1],
                                 label=str(edge_label))
                    else:
                        dot.edge(edge[0], edge[1])
                    added_edges.add(edge)

    return dot


def render_cfg_from_dict(cfg_dict: dict,
                         output_filename: str = "cfg_output") -> graphviz.Digraph:
    """
    Takes the CFG dictionary produced by ASTToCFGBuilder, generates a Graphviz
    Digraph, and saves both the .dot file and the rendered .pdf file.
    """
    dot = graphviz.Digraph(comment="AST Control Flow Graph",
                           format="pdf")
    dot.attr(compound='true', rankdir='TB')

    # 1. Add nodes with formatted block statements
    for node_id, node_data in cfg_dict.items():
        statements = node_data.get("statements", [])

        if isinstance(statements, list):
            clean_lines = [str(stmt) for stmt in statements]
            label = "".join([f"{line}\\l" for line in clean_lines])
        else:
            label = f"{statements}\\l"

        dot.node(str(node_id), label=label, shape='box',
                 fontname='Courier', fontsize='10')

    # 2. Add edges from successors
    added_edges = set()
    for src_id, node_data in cfg_dict.items():
        successors = node_data.get("successors", [])
        for succ in successors:
            if isinstance(succ, dict):
                dst_id = succ.get("target")
                edge_label = succ.get("label")
            else:
                dst_id = succ
                edge_label = None

            if dst_id:
                edge = (str(src_id), str(dst_id))
                if edge not in added_edges:
                    if edge_label:
                        dot.edge(str(src_id), str(dst_id),
                                 label=str(edge_label))
                    else:
                        dot.edge(str(src_id), str(dst_id))
                    added_edges.add(edge)

    # 3. Save the source DOT file
    dot_path = f"{output_filename}.dot"
    dot.save(dot_path)

    # 4. Render to PDF
    pdf_path = dot.render(filename=output_filename, cleanup=True)

    print(f"CFG successfully rendered and saved to:")
    print(f" - DOT file: {dot_path}")
    print(f" - PDF file: {pdf_path}")

    return dot


# =====================================================================
# MAIN ENTRY POINT FOR TESTING
# =====================================================================
if __name__ == "__main__":
    test_code = """ 
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

    print("=" * 70)
    print("TESTING AST TO CFG DICTIONARY GENERATION (CHOLESKY KERNEL)")
    print("=" * 70)

    # 1. Generate CFG dictionary
    cfg_dict = ast_to_cfg_dict(test_code)

    print("\nGenerated CFG Dictionary:")
    print(json.dumps(cfg_dict, indent=2))

    # 2. Export DOT and PDF
    render_cfg_from_dict(cfg_dict,
                         output_filename="cholesky_decomposition")