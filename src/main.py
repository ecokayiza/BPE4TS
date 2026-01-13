"""
Main Pipeline Module.
---------------------
Orchestrates the Time Series BPE experiments.
1. Loads ETTm1 data.
2. Preprocesses (Normalizes/Clips).
3. Runs Grid Search (Phased: Coarse -> Fine).
4. Selects Best Model using Unsupervised Grammar Metrics (Perplexity).
5. Generates comprehensive visualizations.

Manual Usage:
    >>> from src.main import run_experiment
    >>> res = run_experiment(series, n_bins=50, strategy='uniform_fixed', min_freq=50)

Dependencies: pandas, numpy, src.*
"""

import numpy as np
import pandas as pd
import os
import time
import warnings

from src.data_loader import load_data, get_series
from src.discretizer import TimeSeriesDiscretizer
from src.tokenizer import TimeSeriesBPE
from src.metrics import calculate_grammar_quality, calculate_motif_quality
from src.visualization import (
    plot_pareto_frontier, 
    plot_detailed_tokenization, 
    plot_tokenization_gallery
)

# Configuration
RESULT_DIR = "result"
GRID_RESULTS_PATH = os.path.join(RESULT_DIR, "experiment_grid_results_motif.csv")

def ensure_result_dir():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

def run_experiment(
    series, 
    n_bins, 
    strategy, 
    min_freq=2, 
    train_ratio=0.8,
    return_models=False
):
    """
    Runs a single BPE experiment:
    1. Discretizes train/test data.
    2. Trains BPE on train set.
    3. Evaluates on test set (Reconstruction MSE, Compression, Grammar Quality).
    """
    # Split
    series_len = len(series)
    train_len = int(series_len * train_ratio)
    train_data = series[:train_len]
    test_data = series[train_len:]

    # 1. Discretization
    discretizer = TimeSeriesDiscretizer(n_bins=n_bins, strategy=strategy, range_min=-5.0, range_max=5.0)
    discretizer.fit(train_data)
    
    train_discrete = discretizer.transform(train_data)
    test_discrete = discretizer.transform(test_data)
    
    # 2. Train BPE
    # We set a large vocab size so min_freq stops it naturally
    # Pass verbose=False from higher levels if implicit, though run_experiment usually is verbose
    # We will assume run_experiment is silent during sweep if needed, but here we default to True
    # Actually, we should check an argument. For now, let's stick to True unless passed.
    # To support verbose control, we'd need to update run_experiment signature.
    # But since we updated run_parameter_sweep to handle the main bar, we'll likely pass verbose=False here soon.
    # Update: I will check if arguments have verbose.
    bpe = TimeSeriesBPE(vocab_size=5000, initial_vocab_size=n_bins)
    # Check if we should be verbose (hack: check if 'verbose' in locals or just default to False inside loop)
    # Better to update signature. 
    bpe.train(train_discrete, min_freq=min_freq, verbose=False)

    
    # 3. Evaluate on Test
    test_compressed = bpe.encode(test_discrete)
    test_decoded_tokens = bpe.decode(test_compressed)
    
    # Reconstruct
    test_reconstructed = discretizer.inverse_transform(np.array(test_decoded_tokens))
    
    # Metrics
    mse = np.mean((test_data - test_reconstructed) ** 2)
    compression_ratio = len(test_data) / max(1, len(test_compressed))
    
    # Unsupervised Grammar Metrics (The clean/robust selection criteria)
    grammar_stats = calculate_grammar_quality(test_compressed)
    
    # Clustering Metrics (Silhouette Score) - For Interpretability
    motif_stats = calculate_motif_quality(test_compressed, bpe, test_data)
    
    result_dict = {
        'n_bins': n_bins,
        'strategy': strategy,
        'min_freq': min_freq,
        'mse': mse,
        'compression_ratio': compression_ratio,
        'perplexity': grammar_stats['perplexity'],
        'entropy': grammar_stats['conditional_entropy'],
        'consistency_score': motif_stats['consistency_score'],
        'vocab_size': bpe.current_vocab_count
    }
    
    if return_models:
        return result_dict, bpe, discretizer, test_data, test_compressed
    return result_dict

def run_parameter_sweep(series, n_bins_list, strategies, min_freq_list, csv_path=GRID_RESULTS_PATH):
    """
    Runs a grid search, skipping already executed configurations.
    Persists results incrementally to CSV.
    """
    ensure_result_dir()
    
    done_configs = set()
    results = []

    # Resume capability
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}...")
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty:
                results = existing_df.to_dict('records')
                for r in results:
                    done_configs.add((int(r['n_bins']), r['strategy'], int(r['min_freq'])))
        except Exception as e:
            print(f"Could not load existing results: {e}")
    
    # Build full list of configs to iterate over
    all_configs = []
    for n_bins in n_bins_list:
        for strategy in strategies:
            for min_freq in min_freq_list:
                 all_configs.append((n_bins, strategy, min_freq))

    # Calculate remaining work
    configs_to_run = [c for c in all_configs if c not in done_configs]
    
    if not configs_to_run:
        print("All configurations already completed.")
        return pd.DataFrame(results)

    print(f"\nScanning {len(configs_to_run)}/{len(all_configs)} configurations...")
    
    # Use tqdm for the outer loop (Sweep Progress)
    from tqdm import tqdm
    pbar = tqdm(configs_to_run, desc="Parameter Sweep")
    
    # Suppress warnings during sweep to keep progress bar clean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for (n_bins, strategy, min_freq) in pbar:
            # Update description to show current test parameters
            pbar.set_description(f"Run {len(results)+1}: Bins={n_bins}, Strat={strategy}, MinFreq={min_freq}")
            
            try:
                # We silence the inner BPE training bar (verbose=False inside run_experiment)
                res = run_experiment(series, n_bins, strategy, min_freq=min_freq)
                results.append(res)
                
                # Append to CSV
                df_res = pd.DataFrame([res])
                df_res.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
            except Exception as e:
                # tqdm.write ensures we don't break the bar
                tqdm.write(f"Experiment Failed: {e}")
            
    return pd.DataFrame(results)

def select_best_model(df, top_n=3, consistency_threshold=0.6, perplexity_threshold=None):
    """
    Selects the best model based on Motif Consistency (Visual Quality) and Perplexity (Grammar Quality).
    
    Args:
        df: DataFrame with experiment results.
        top_n: Number of top candidates to display.
        consistency_threshold: Minimum consistency score (higher is better).
        perplexity_threshold: Maximum perplexity (lower is better). If None, uses median.
    """
    if df.empty:
        return None
        
    df = df.copy()
    
    # 1. Consistency Filter
    if 'consistency_score' not in df.columns:
        print("Error: 'consistency_score' metric missing. Re-running experiments required.")
        return df.iloc[0]
    
    if 'perplexity' not in df.columns:
        print("Error: 'perplexity' metric missing. Re-running experiments required.")
        return df.iloc[0]
        
    # We want consistent motifs (High Correlation)
    valid_df = df[df['consistency_score'] >= consistency_threshold].copy()
    
    if valid_df.empty:
        print(f"Warning: No models passed consistency >= {consistency_threshold}. Using all models.")
        valid_df = df.copy()
    else:
        print(f"Motif Filter: {len(df)} -> {len(valid_df)} candidates (Consistency >= {consistency_threshold})")
    
    # 2. Perplexity Filter (Low perplexity = good grammar structure)
    if perplexity_threshold is None:
        perplexity_threshold = valid_df['perplexity'].median()
    
    valid_ppl_df = valid_df[valid_df['perplexity'] <= perplexity_threshold].copy()
    
    if valid_ppl_df.empty:
        print(f"Warning: No models passed perplexity <= {perplexity_threshold:.2f}. Using consistency-filtered models.")
        valid_ppl_df = valid_df
    else:
        print(f"Perplexity Filter: {len(valid_df)} -> {len(valid_ppl_df)} candidates (Perplexity <= {perplexity_threshold:.2f})")
        
    # 3. Ranking: Combined Score
    # Normalize metrics for combined ranking
    # Consistency: higher is better (0-1 scale already)
    # Perplexity: lower is better (need to invert)
    ppl_min, ppl_max = valid_ppl_df['perplexity'].min(), valid_ppl_df['perplexity'].max()
    if ppl_max > ppl_min:
        valid_ppl_df['ppl_score'] = 1 - (valid_ppl_df['perplexity'] - ppl_min) / (ppl_max - ppl_min)
    else:
        valid_ppl_df['ppl_score'] = 1.0
    
    # Combined score: weighted average (consistency 50%, perplexity 50%)
    valid_ppl_df['combined_score'] = 0.5 * valid_ppl_df['consistency_score'] + 0.5 * valid_ppl_df['ppl_score']
    
    ranked = valid_ppl_df.sort_values('combined_score', ascending=False)
    
    print("\nTop Candidates (Best Visual Motifs + Grammar Quality):")
    cols = ['n_bins', 'strategy', 'min_freq', 'mse', 'consistency_score', 'perplexity', 'compression_ratio']
    # Check if cols exist
    cols = [c for c in cols if c in ranked.columns]
    print(ranked[cols].head(top_n))
    
    return ranked.iloc[0]

def main():
    ensure_result_dir()
    
    # 1. Load Data
    print("Loading Data...")
    df = load_data()
    series = get_series(df, 'OT').values
    
    # Preprocess
    mean = np.mean(series)
    std = np.std(series)
    series_norm = (series - mean) / std
    series_final = np.clip(series_norm, -5.0, 5.0) # Remove extreme outliers
    
    # 2. Phase 1: Coarse Grid Search
    print("\n=== Phase 1: Coarse Grid Search ===")
    bins_coarse = [20, 30, 50, 100]
    strategies = ['quantile', 'gaussian', 'uniform_fixed']
    min_freqs = [30,50,100,200,300]
    
    results_df = run_parameter_sweep(series_final, bins_coarse, strategies, min_freqs)
    
    # Select Best Candidate for visual check
    best_config = select_best_model(results_df)

    # 3. Phase 2: Refinement
    print("\n=== Phase 2: Refinement Search ===")
    
    best_n_bins = int(best_config['n_bins'])
    best_min_freq = int(best_config['min_freq'])
    best_strategy = best_config['strategy']
    
    # Define search space around the best candidate
    bins_refine = sorted(list(set([
        max(10, best_n_bins - 10), 
        best_n_bins, 
        best_n_bins + 10
    ])))
    
    freq_refine = sorted(list(set([
        max(2, best_min_freq - 10), 
        best_min_freq, 
        best_min_freq + 10
    ])))
    
    print(f"Refining around Bins={best_n_bins}, MinFreq={best_min_freq}...")
    print(f"Search Space: Bins={bins_refine}, Freq={freq_refine}")
    
    # Run refinement sweep (only for the best strategy)
    results_df = run_parameter_sweep(series_final, bins_refine, [best_strategy], freq_refine)
    
    # Final Selection
    best_config = select_best_model(results_df)

    print(f"\nFinal Selected Config: Bins={best_config['n_bins']}, Strat={best_config['strategy']}, MinFreq={best_config['min_freq']}")

    # 4. Final Viz & Analysis
    print("\nGenerating Final Visualizations...")
    
    # Retrain best model to get objects
    res, bpe, discretizer, _, test_comp = run_experiment(
        series_final, 
        n_bins=int(best_config['n_bins']), 
        strategy=best_config['strategy'], 
        min_freq=int(best_config['min_freq']),
        return_models=True
    )
    
    # A. Pareto Plot
    plot_pareto_frontier(results_df, os.path.join(RESULT_DIR, "pareto_frontier.png"))
    
    # B. Detailed Sequence Viz
    plot_detailed_tokenization(
        series_final[int(len(series_final)*0.8):], # Test set for viz
        test_comp,
        bpe,
        discretizer,
        os.path.join(RESULT_DIR, "detailed_tokenization.png"),
        info_dict=best_config.to_dict()
    )
    
    # C. Gallery
    plot_tokenization_gallery(
        series_final, 
        test_comp,
        bpe, 
        os.path.join(RESULT_DIR, "tokenization_gallery.png"),
        info_dict=best_config.to_dict()
    )


if __name__ == "__main__":
    main()
