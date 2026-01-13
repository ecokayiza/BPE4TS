import networkx as nx
import matplotlib.pyplot as plt
from src.tokenizer import TimeSeriesBPE

def plot_token_hierarchy(bpe_model: TimeSeriesBPE, token_id: int, output_path: str):
    """
    Visualizes the hierarchical structure of a specific BPE token as a tree.
    """
    graph = nx.DiGraph()
    labels = {}
    
    def build_tree(current_id):
        # Label node
        labels[current_id] = str(current_id)
        
        # Base case: initial tokens (leaves)
        if current_id < bpe_model.initial_vocab_size:
            graph.add_node(current_id, type='leaf')
            return

        # Recursive step: get children
        if current_id in bpe_model.rules:
            left, right = bpe_model.rules[current_id]
            
            graph.add_edge(current_id, left)
            graph.add_edge(current_id, right)
            
            build_tree(left)
            build_tree(right)
        else:
            # Fallback if rule missing (shouldn't happen)
            graph.add_node(current_id, type='leaf')

    build_tree(token_id)
    
    # Plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)
    # Ideally use a tree layout like graphviz's dot, but spring is easier dep-wise for now
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(graph, prog='dot')
    except ImportError:
        # Fallback to a custom hierarchical layout if possible, or just spring
        pass

    nx.draw(graph, pos, with_labels=True, labels=labels, 
            node_color='lightblue', node_size=500, font_size=10, arrows=True)
    plt.title(f"Hierarchy of Token {token_id}")
    plt.savefig(output_path)
    plt.close()
    print(f"Token hierarchy saved to {output_path}")

def plot_token_hierarchy_tree(bpe_model: TimeSeriesBPE, token_id: int, output_path: str):
    """
    Manual simple tree layout logic to avoid graphviz dependency issues.
    """
    graph = nx.DiGraph()
    labels = {}
    
    def build_tree(current_id):
        labels[current_id] = str(current_id)
        if current_id < bpe_model.initial_vocab_size:
            return 0  # Depth
        if current_id in bpe_model.rules:
            left, right = bpe_model.rules[current_id]
            graph.add_edge(current_id, left)
            graph.add_edge(current_id, right)
            d_l = build_tree(left)
            d_r = build_tree(right)
            return max(d_l, d_r) + 1
        return 0

    max_depth = build_tree(token_id)
    
    # Custom hierarchy layout
    pos = {}
    def assign_pos(node, x, y, width):
        pos[node] = (x, y)
        if node in bpe_model.rules and node >= bpe_model.initial_vocab_size:
            left, right = bpe_model.rules[node]
            assign_pos(left, x - width/2, y - 1, width/2)
            assign_pos(right, x + width/2, y - 1, width/2)
    
    assign_pos(token_id, 0, 0, 4)

    plt.figure(figsize=(12, 6))
    nx.draw(graph, pos, with_labels=True, labels=labels, 
            node_color=['lightgreen' if n < bpe_model.initial_vocab_size else 'skyblue' for n in graph.nodes()],
            node_size=600, font_size=8, arrows=True)
            
    plt.title(f"Compositional Hierarchy of Token {token_id}")
    plt.savefig(output_path)
    plt.close()
    print(f"Hierarchy tree saved to {output_path}")
