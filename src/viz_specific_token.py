import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from src.data_loader import load_data, get_series
from src.discretizer import TimeSeriesDiscretizer
from src.tokenizer import TimeSeriesBPE
import os

RESULT_DIR = "result"

def viz_token_example():
    # 1. Setup Data & Config (Best Config: Bins=55, Uniform, MinFreq=2)
    print("Loading Data & Retraining Model...")
    df = load_data()
    raw_series = get_series(df, 'OT').values
    
    # Preprocessing
    mean = np.mean(raw_series)
    std = np.std(raw_series)
    series_norm = (raw_series - mean) / std
    series_final = np.clip(series_norm, -5.0, 5.0)
    
    # 2. Discretization
    n_bins = 55
    discretizer = TimeSeriesDiscretizer(n_bins=n_bins, strategy='uniform_fixed', range_min=-5.0, range_max=5.0)
    discretizer.fit(series_final)
    train_discrete = discretizer.transform(series_final)
    
    # 3. Train BPE
    # Use higher min_freq to match robust configuration
    min_freq = 50 
    bpe = TimeSeriesBPE(vocab_size=5000, initial_vocab_size=n_bins)
    compressed = bpe.train(train_discrete, min_freq=min_freq)
    
    # 4. Select a random complex token
    used_tokens = list(set(compressed))
    # Filter for non-leaf tokens to make it interesting
    complex_tokens = [t for t in used_tokens if t >= bpe.initial_vocab_size]
    
    if complex_tokens:
        target_token = random.choice(complex_tokens)
        print(f"Randomly selected complex Token ID: {target_token}")
    else:
        print("No complex tokens found with current settings.")
        target_token = used_tokens[0] if used_tokens else 0
        
    print(f"Visualizing Token ID: {target_token}")

    # 5. Build Graph
    G = nx.DiGraph()
    labels = {}
    node_colors = []
    
    def build_graph(tid):
        labels[tid] = str(tid)
        if tid < bpe.initial_vocab_size:
            G.add_node(tid, type='leaf')
            return
            
        if tid in bpe.rules:
            l, r = bpe.rules[tid]
            G.add_edge(tid, l)
            G.add_edge(tid, r)
            build_graph(l)
            build_graph(r)

    build_graph(target_token)
    
    # Layout using custom hierarchy logic
    pos = {}
    def assign_pos(node, x, y, width):
        pos[node] = (x, y)
        if node in bpe.rules and node >= bpe.initial_vocab_size:
            l, r = bpe.rules[node]
            assign_pos(l, x - width/2, y - 1, width/2)
            assign_pos(r, x + width/2, y - 1, width/2)
    
    assign_pos(target_token, 0, 0, 4)

    # 6. Prepare Plots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: Hierarchy Tree (Left & Top)
    ax_tree = fig.add_subplot(gs[0, :])
    
    color_map = ['lightgreen' if n < bpe.initial_vocab_size else 'skyblue' for n in G.nodes()]
    
    nx.draw(G, pos, ax=ax_tree, with_labels=True, labels=labels, 
            node_color=color_map, node_size=700, font_size=9, arrows=True, edge_color='gray')
    ax_tree.set_title(f"Compositional Hierarchy of Token {target_token}", fontsize=12, fontweight='bold')
    
    # Plot 2: Reconstructed Shape (Bottom Left - Discrete)
    ax_shape = fig.add_subplot(gs[1, 0])
    
    decoded_ids = bpe.decode_token(target_token)
    decoded_vals = discretizer.inverse_transform(np.array(decoded_ids))
    
    # De-normalize for real world values
    real_vals = decoded_vals * std + mean
    
    ax_shape.plot(real_vals, marker='o', linestyle='-', color='crimson', linewidth=2)
    ax_shape.set_title(f"Decoded Time Series Shape (Real Values)", fontsize=11)
    ax_shape.set_xlabel("Time Steps")
    ax_shape.set_ylabel("Oil Temperature")
    ax_shape.grid(True, alpha=0.3)
    
    # Plot 3: Composition Blocks (Bottom Right)
    ax_comp = fig.add_subplot(gs[1, 1])
    
    # Recursive breakdown for blocks
    current_x = 0
    colors = plt.cm.tab20.colors
    
    def plot_blocks(tid, x_start, depth=0):
        length = len(bpe.decode_token(tid))
        # Draw box
        rect = plt.Rectangle((x_start, depth), length, 0.8, 
                           facecolor=colors[tid % 20], alpha=0.8, edgecolor='white')
        ax_comp.add_patch(rect)
        ax_comp.text(x_start + length/2, depth + 0.4, str(tid), 
                   ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        if tid >= bpe.initial_vocab_size:
            l, r = bpe.rules[tid]
            l_len = len(bpe.decode_token(l))
            plot_blocks(l, x_start, depth - 1)
            plot_blocks(r, x_start + l_len, depth - 1)
            
    plot_blocks(target_token, 0, 0)
    ax_comp.set_xlim(0, len(decoded_ids))
    ax_comp.set_ylim(-5, 2) # Assuming max depth approx 5
    ax_comp.set_title("Compositional Blocks (Timeline View)", fontsize=11)
    ax_comp.set_xlabel("Time Steps")
    ax_comp.set_yticks([])
    ax_comp.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(RESULT_DIR, "token_example_detailed.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    viz_token_example()
