import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_data, get_series
from src.discretizer import TimeSeriesDiscretizer
from src.tokenizer import TimeSeriesBPE

def test_high_bins():
    print("Loading Data...")
    df = load_data()
    raw_series = get_series(df, 'OT').values
    
    # Preprocessing
    mean = np.mean(raw_series)
    std = np.std(raw_series)
    series_norm = (raw_series - mean) / std
    series_final = np.clip(series_norm, -5.0, 5.0)
    
    # We will test 3 configs
    configs = [50, 200, 500]
    results = []
    
    print(f"\nComparing Bin Counts: {configs} (Fixed Strategy: Uniform, MinFreq: 30)")
    print("-" * 60)
    print(f"{'Bins':<10} | {'MSE':<15} | {'Compression':<15} | {'Vocab Used'}")
    print("-" * 60)
    
    for bins in configs:
        # Discretize
        discretizer = TimeSeriesDiscretizer(n_bins=bins, strategy='uniform_fixed', range_min=-5.0, range_max=5.0)
        discretizer.fit(series_final)
        discrete_seq = discretizer.transform(series_final)
        
        # Train BPE
        # We need a larger vocab size allowance for high bins
        bpe = TimeSeriesBPE(vocab_size=10000, initial_vocab_size=bins)
        
        # Use a reasonable min_freq to see if patterns are found
        compressed = bpe.train(discrete_seq, min_freq=30)
        
        # Reconstruct
        decoded_tokens = bpe.decode(compressed)
        reconstructed = discretizer.inverse_transform(np.array(decoded_tokens))
        
        mse = np.mean((series_final - reconstructed) ** 2)
        comp_ratio = len(series_final) / len(compressed)
        vocab_used = bpe.current_vocab_count - bins # Number of merged tokens created
        
        print(f"{bins:<10} | {mse:<15.6f} | {comp_ratio:<15.2f} | {vocab_used}")
        results.append((bins, mse, comp_ratio))

    print("-" * 60)
    
if __name__ == "__main__":
    test_high_bins()
