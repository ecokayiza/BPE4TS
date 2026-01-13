import numpy as np
import pandas as pd
from src.data_loader import load_data, get_series
from src.discretizer import TimeSeriesDiscretizer
from src.tokenizer import TimeSeriesBPE
from src.visualization import plot_pareto_frontier, plot_detailed_tokenization, plot_tokenization_gallery
import os

RESULT_DIR = "result"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    
GRID_RESULTS_PATH = os.path.join(RESULT_DIR, "experiment_grid_results.csv")

def run_experiment(
    series, 
    n_bins, 
    strategy, 
    min_freq=2, 
    train_ratio=0.8,
    return_models=False
):
    series_len = len(series)
    train_len = int(series_len * train_ratio)
    train_data = series[:train_len]
    test_data = series[train_len:]

    # 1. Discretization
    if strategy == 'uniform_fixed':
        discretizer = TimeSeriesDiscretizer(n_bins=n_bins, strategy='uniform_fixed', range_min=-5.0, range_max=5.0)
    elif strategy == 'gaussian':
        discretizer = TimeSeriesDiscretizer(n_bins=n_bins, strategy='gaussian', range_min=-5.0, range_max=5.0)
    else:
        discretizer = TimeSeriesDiscretizer(n_bins=n_bins, strategy='quantile')
    
    discretizer.fit(train_data)
    
    train_discrete = discretizer.transform(train_data)
    test_discrete = discretizer.transform(test_data)
    
    # 2. Train BPE
    # We set a high vocab limit so that min_freq is the main stopping criterion
    bpe = TimeSeriesBPE(vocab_size=5000, initial_vocab_size=n_bins)
    train_compressed = bpe.train(train_discrete, min_freq=min_freq)
    
    # 3. Evaluate on Test
    test_compressed = bpe.encode(test_discrete)
    test_decoded_tokens = bpe.decode(test_compressed)
    
    # Reconstruct
    test_reconstructed = discretizer.inverse_transform(np.array(test_decoded_tokens))
    
    # Metrics
    mse = np.mean((test_data - test_reconstructed) ** 2)
    compression_ratio = len(test_data) / len(test_compressed)
    
    actual_vocab_size = bpe.current_vocab_count
    
    result_dict = {
        'n_bins': n_bins,
        'strategy': strategy,
        'min_freq': min_freq,
        'mse': mse,
        'compression_ratio': compression_ratio
    }
    
    if return_models:
        return result_dict, bpe, discretizer, test_data, test_compressed
    return result_dict

def run_parameter_sweep(series, n_bins_list, strategies, min_freq_list, csv_path=GRID_RESULTS_PATH):
    """
    Generic function to run a grid search and save results incrementally.
    """
    done_configs = set()
    results = []

    # Load existing
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}...")
        existing_df = pd.read_csv(csv_path)
        if 'min_freq' in existing_df.columns:
            results = existing_df.to_dict('records')
            for r in results:
                try: 
                    done_configs.add((int(r['n_bins']), r['strategy'], int(r['min_freq'])))
                except:
                    pass
    
    total_exp = len(n_bins_list) * len(strategies) * len(min_freq_list)
    curr = 1
    
    print(f"\nScanning {total_exp} configurations...")

    for n_bins in n_bins_list:
        for strategy in strategies:
            for min_freq in min_freq_list:
                if (n_bins, strategy, min_freq) in done_configs:
                    curr += 1
                    continue

                print(f"Run {curr}/{total_exp}: Bins={n_bins}, Strat={strategy}, MinFreq={min_freq}")
                try:
                    res = run_experiment(
                        series, n_bins, strategy, min_freq=min_freq
                    )
                    results.append(res)
                    # Incremental save
                    curr_df = pd.DataFrame([res])
                    curr_df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
                except Exception as e:
                    print(f"Error: {e}")
                
                curr += 1
                
    # Filter results to return ONLY what was requested (or relevant to this sweep)
    # This prevents Phase 1 from selecting a "best" candidate from a previous Phase 2 run that isn't in the coarse grid.
    all_df = pd.DataFrame(results)
    if all_df.empty:
        return all_df
        
    mask = (
        all_df['n_bins'].isin(n_bins_list) & 
        all_df['strategy'].isin(strategies) & 
        all_df['min_freq'].isin(min_freq_list)
    )
    return all_df[mask]

def analyze_best_candidates(df, top_n=3):
    """
    Select best candidates based on a heuristic score.
    Heuristic: Minimize MSE while maximizing Compression.
    Distance from Ideal Point (0 MSE, Max Compression) method.
    """
    df = df.copy()
    
    # Normalize 0..1
    mse_norm = (df['mse'] - df['mse'].min()) / (df['mse'].max() - df['mse'].min() + 1e-9)
    # For compression, higher is better. We invert it to 'distance from best' or use 1-norm.
    comp_norm = (df['compression_ratio'] - df['compression_ratio'].min()) / (df['compression_ratio'].max() - df['compression_ratio'].min() + 1e-9)
    
    # Distance to (MSE=0, Comp=1) in normalized space
    df['dist_to_ideal'] = np.sqrt(mse_norm**2 + (1 - comp_norm)**2)
    
    best_balanced = df.sort_values('dist_to_ideal').iloc[0]
    
    print("\n--- Best Balanced Candidate ---")
    print(best_balanced[['n_bins', 'strategy', 'min_freq', 'mse', 'compression_ratio']])
    
    return best_balanced

def main():
    print("Loading Data...")
    df = load_data()
    raw_series = get_series(df, 'OT').values
    
    # Preprocessing
    print("Normalizing Data...")
    mean = np.mean(raw_series)
    std = np.std(raw_series)
    series_norm = (raw_series - mean) / std
    
    print("Truncating Data (-5 to 5)...")
    series_final = np.clip(series_norm, -5.0, 5.0)
    
    # --- Phase 1: Coarse Grid Search ---
    print("\n=== Phase 1: Coarse Grid Search ===")
    coarse_n_bins = [20, 50, 100]
    coarse_strategies = ['quantile', 'gaussian', 'uniform_fixed']
    coarse_min_freq = [5, 50, 200]
    
    df_coarse = run_parameter_sweep(
        series_final, coarse_n_bins, coarse_strategies, coarse_min_freq
    )
    
    # Analyze best candidate to refine
    best_candidate = analyze_best_candidates(df_coarse)
    
    best_bins = int(best_candidate['n_bins'])
    best_strategy = best_candidate['strategy']
    best_freq = int(best_candidate['min_freq'])
    
    # --- Phase 2: Fine Grid Search ---
    print(f"\n=== Phase 2: Fine Grid Search around {best_bins} bins, {best_strategy}, min_freq={best_freq} ===")
    
    # Define fine grid around best parameters
    # Bins: +/- 10 and +/- 5
    fine_n_bins = sorted(list(set([
        max(10, best_bins - 10), 
        max(10, best_bins - 5), 
        best_bins, 
        best_bins + 5, 
        best_bins + 10
    ])))
    
    # Freq: +/- relative range
    fine_min_freq = sorted(list(set([
        max(2, int(best_freq * 0.5)),
        max(2, int(best_freq * 0.75)),
        best_freq,
        int(best_freq * 1.25),
        int(best_freq * 1.5)
    ])))
    
    fine_strategies = [best_strategy] 
    
    df_all = run_parameter_sweep(
        series_final, fine_n_bins, fine_strategies, fine_min_freq
    )
    
    # --- Phase 3: Final Selection & Visualization ---
    print("\n=== Phase 3: Final Selection ===")
    final_best = analyze_best_candidates(df_all)
    
    final_bins = int(final_best['n_bins'])
    final_strategy = final_best['strategy']
    final_freq = int(final_best['min_freq'])
    
    print("Generating Pareto Plot for ALL results...")
    plot_pareto_frontier(df_all, os.path.join(RESULT_DIR, "pareto_frontier.png"))
    
    print(f"\nGenerating Detailed Visualization for Final Best: Bins={final_bins}, {final_strategy}, Freq={final_freq}")
    _, best_bpe, best_disc, test_data, test_tokens = run_experiment(
        series_final, 
        n_bins=final_bins, 
        strategy=final_strategy, 
        min_freq=final_freq, 
        return_models=True
    )
    
    plot_detailed_tokenization(
        test_data, 
        test_tokens, 
        best_bpe, 
        best_disc, 
        os.path.join(RESULT_DIR, "detailed_tokenization.png"),
        info_dict=final_best.to_dict()
    )
    
    print("Generating Tokenization Gallery...")
    plot_tokenization_gallery(
        test_data,
        test_tokens,
        best_bpe,
        os.path.join(RESULT_DIR, "tokenization_gallery.png"),
        n_examples=8,
        info_dict=final_best.to_dict()
    )

if __name__ == "__main__":
    main()
