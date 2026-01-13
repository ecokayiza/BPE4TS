import pandas as pd
import os

csv_path = "result/experiment_grid_results.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    # Filter for robust frequencies
    robust_df = df[df['min_freq'] >= 50].copy()
    
    if robust_df.empty:
        print("No robust configurations found (min_freq >= 50).")
    else:
        # Sort by compression ratio (descending) to see high compression robust models
        print("Top Robust Configurations (by Compression):")
        print(robust_df.sort_values('compression_ratio', ascending=False)[['n_bins', 'strategy', 'min_freq', 'mse', 'compression_ratio']].head(5))
        
        # Sort by MSE (ascending)
        print("\nTop Robust Configurations (by MSE):")
        print(robust_df.sort_values('mse', ascending=True)[['n_bins', 'strategy', 'min_freq', 'mse', 'compression_ratio']].head(5))
else:
    print("No results file found.")
