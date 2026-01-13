import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_data, get_series
from src.discretizer import TimeSeriesDiscretizer
from src.tokenizer import TimeSeriesBPE
import os

RESULT_DIR = "result"

def find_semantic_sweetspot():
    print("Loading Data...")
    df = load_data()
    raw_series = get_series(df, 'OT').values
    
    # Preprocessing
    mean = np.mean(raw_series)
    std = np.std(raw_series)
    series_norm = (raw_series - mean) / std
    series_final = np.clip(series_norm, -5.0, 5.0)
    
    # Range of bins to test for "Semantic Sweet Spot"
    # We look for the peak of compression/pattern finding with robust min_freq
    bins_test = [20, 30, 40, 50, 60, 80, 100]
    min_freq = 50 
    
    results = []
    
    print(f"\nSearching for Semantic Sweet Spot (MinFreq={min_freq})...")
    print("-" * 70)
    print(f"{'Bins':<6} | {'MSE':<10} | {'Compression':<12} | {'Vocab Learned':<15} | {'Richness'}")
    print("-" * 70)
    
    for bins in bins_test:
        discretizer = TimeSeriesDiscretizer(n_bins=bins, strategy='uniform_fixed', range_min=-5.0, range_max=5.0)
        discretizer.fit(series_final)
        discrete_seq = discretizer.transform(series_final)
        
        # BPE
        bpe = TimeSeriesBPE(vocab_size=3000, initial_vocab_size=bins)
        bpe.train(discrete_seq, min_freq=min_freq)
        
        # Evaluate
        compressed = bpe.encode(discrete_seq)
        decoded = bpe.decode(compressed)
        reconstructed = discretizer.inverse_transform(np.array(decoded))
        
        mse = np.mean((series_final - reconstructed) ** 2)
        comp = len(series_final) / len(compressed)
        learned_vocab = bpe.current_vocab_count - bins
        
        # Heuristic for "Semantic Richness": efficiency of the learned vocabulary
        # How much does each new token contribute to compression?
        richness = (len(series_final) - len(compressed)) / (learned_vocab + 1e-9)
        
        print(f"{bins:<6} | {mse:<10.6f} | {comp:<12.2f} | {learned_vocab:<15} | {richness:.2f}")
        results.append({'bins': bins, 'mse': mse, 'comp': comp, 'vocab': learned_vocab, 'richness': richness})

    # Plot
    bins_x = [r['bins'] for r in results]
    mse_y = [r['mse'] for r in results]
    comp_y = [r['comp'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Number of Bins')
    ax1.set_ylabel('MSE (Lower is Better)', color=color)
    ax1.plot(bins_x, mse_y, color=color, marker='o', label='MSE (Precision)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Compression Ratio (Higher is Better)', color=color)
    ax2.plot(bins_x, comp_y, color=color, marker='s', label='Compression (Semantics)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"Semantic Sweet Spot Analysis (Min Freq={min_freq})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULT_DIR, "semantic_sweetspot.png"))
    print(f"\nPlot saved to {os.path.join(RESULT_DIR, 'semantic_sweetspot.png')}")

if __name__ == "__main__":
    find_semantic_sweetspot()
