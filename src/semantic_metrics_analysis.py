import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from scipy.stats import entropy
from src.data_loader import load_data, get_series
from src.discretizer import TimeSeriesDiscretizer
from src.tokenizer import TimeSeriesBPE
import os

RESULT_DIR = "result"

def get_token_length(bpe, token_id):
    # Recursively calculate the length (number of leaves) of a token
    if token_id < bpe.initial_vocab_size:
        return 1
    if token_id not in bpe.rules:
        return 1
    l, r = bpe.rules[token_id]
    return get_token_length(bpe, l) + get_token_length(bpe, r)

def analyze_semantics():
    print("Loading Data...")
    df = load_data()
    raw_series = get_series(df, 'OT').values
    
    # Preprocessing
    mean = np.mean(raw_series)
    std = np.std(raw_series)
    series_norm = (raw_series - mean) / std
    series_final = np.clip(series_norm, -5.0, 5.0)
    
    # Semantic Search Setting
    bins_test = [20, 30, 40, 50, 60, 80, 100, 150]
    min_freq = 50 
    
    metrics = []
    
    print(f"\nAnalyzing Semantic Metrics (MinFreq={min_freq})...")
    print("-" * 100)
    print(f"{'Bins':<5} | {'AvgLen':<8} | {'MaxLen':<8} | {'VocabSize':<10} | {'UniqueUsed':<10} | {'Entropy':<8} | {'MSE':<8}")
    print("-" * 100)
    
    for bins in bins_test:
        # 1. Pipeline
        discretizer = TimeSeriesDiscretizer(n_bins=bins, strategy='uniform_fixed', range_min=-5.0, range_max=5.0)
        discretizer.fit(series_final)
        discrete_seq = discretizer.transform(series_final)
        
        bpe = TimeSeriesBPE(vocab_size=5000, initial_vocab_size=bins)
        bpe.train(discrete_seq, min_freq=min_freq)
        
        # 2. Encode
        encoded_seq = bpe.encode(discrete_seq)
        
        # 3. Calculate Metrics
        
        # A. Token Length Statistics (How "high-level" are the concepts?)
        # Compression Ratio is technically Avg Length, but let's be explicit
        avg_len = len(series_final) / len(encoded_seq)
        
        # Max Length in the vocabulary (Structural Depth)
        # We check all tokens in the learned vocabulary
        learned_tokens = list(bpe.rules.keys())
        if learned_tokens:
            max_vocab_len = max([get_token_length(bpe, t) for t in learned_tokens])
        else:
            max_vocab_len = 1
            
        # B. Vocabulary Statistics (How rich is the language?)
        total_vocab_size = bpe.current_vocab_count
        
        # C. Usage Statistics (Are we actually using the rich words?)
        token_counts = Counter(encoded_seq)
        unique_tokens_used = len(token_counts)
        used_probs = np.array(list(token_counts.values())) / len(encoded_seq)
        seq_entropy = entropy(used_probs) # Higher entropy = more information content per token choice
        
        # D. MSE (Cost)
        decoded = bpe.decode(encoded_seq)
        reconstructed = discretizer.inverse_transform(np.array(decoded))
        mse = np.mean((series_final - reconstructed) ** 2)
        
        print(f"{bins:<5} | {avg_len:<8.2f} | {max_vocab_len:<8} | {total_vocab_size:<10} | {unique_tokens_used:<10} | {seq_entropy:<8.3f} | {mse:<8.4f}")
        
        metrics.append({
            'bins': bins,
            'avg_len': avg_len,
            'max_len': max_vocab_len,
            'vocab_size': total_vocab_size,
            'unique_used': unique_tokens_used,
            'entropy': seq_entropy,
            'mse': mse
        })

    # 4. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Average Token Length (Abstraction Level)
    ax1 = axes[0, 0]
    ax1.plot([m['bins'] for m in metrics], [m['avg_len'] for m in metrics], marker='o', color='tab:blue')
    ax1.set_title("Average Token Length (Abstraction Level)")
    ax1.set_xlabel("Bins")
    ax1.set_ylabel("Avg Time Steps per Token")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max Token Length (Pattern Complexity)
    ax2 = axes[0, 1]
    ax2.plot([m['bins'] for m in metrics], [m['max_len'] for m in metrics], marker='s', color='tab:orange')
    ax2.set_title("Max Token Length (Deepest Pattern)")
    ax2.set_xlabel("Bins")
    ax2.set_ylabel("Max Time Steps")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Vocabulary Usage (Richness)
    ax3 = axes[1, 0]
    ax3.plot([m['bins'] for m in metrics], [m['unique_used'] for m in metrics], marker='^', color='tab:green', label='Unique Used')
    ax3.plot([m['bins'] for m in metrics], [m['vocab_size'] for m in metrics], linestyle='--', color='gray', label='Total Learned')
    ax3.set_title("Vocabulary Utilization")
    ax3.set_xlabel("Bins")
    ax3.set_ylabel("Count")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Entropy (Information Balance)
    ax4 = axes[1, 1]
    ax4.plot([m['bins'] for m in metrics], [m['entropy'] for m in metrics], marker='D', color='tab:purple')
    ax4.set_title("Token Distribution Entropy (Information Richness)")
    ax4.set_xlabel("Bins")
    ax4.set_ylabel("Shannon Entropy (nats)")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(RESULT_DIR, "semantic_metrics.png")
    plt.savefig(output_path)
    print(f"\nAnalysis plot saved to {output_path}")

if __name__ == "__main__":
    analyze_semantics()
