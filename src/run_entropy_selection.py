import numpy as np
import pandas as pd
from src.data_loader import load_data, get_series
from src.discretizer import TimeSeriesDiscretizer
from src.tokenizer import TimeSeriesBPE
from src.metrics_entropy import calculate_grammar_quality
import os

def get_token_depth(bpe, token_id):
    """Recursively calculate the depth (leaf count) of a token."""
    if token_id < bpe.initial_vocab_size:
        return 1
    if token_id not in bpe.rules:
        return 1
    l, r = bpe.rules[token_id]
    return get_token_depth(bpe, l) + get_token_depth(bpe, r)

def run_grammar_analysis():
    # 1. Load Data
    full_df = load_data()
    series = get_series(full_df).values
    
    # 2. Setup Experiment Grid
    bins_list = [10, 20, 30, 50, 100, 200, 500]
    strategies = ['uniform'] 
    
    results = []
    
    print(f"{'Bins':<5} | {'Strategy':<10} | {'MaxDepth':<8} | {'Perplex':<8} | {'Entropy':<8} | {'Compr':<6} | {'Vocab':<6}")
    print("-" * 75)
    
    for strategy in strategies:
        for n_bins in bins_list:
            # A. Discretize
            discretizer = TimeSeriesDiscretizer(n_bins=n_bins, strategy=strategy)
            discretizer.fit(series)
            tokens = discretizer.transform(series)
            
            # B. Train BPE
            # We set a high vocab limit so that min_freq is the main stopping criterion
            bpe = TimeSeriesBPE(vocab_size=2000, initial_vocab_size=n_bins) 
            encoded_tokens = bpe.train(tokens, min_freq=50)
            
            # C. Calculate Grammar Metrics
            grammar_stats = calculate_grammar_quality(encoded_tokens)
            
            # D. Calculate Structural Depth
            max_depth = 0
            if bpe.rules:
                # Check depth of the top 50 most frequent tokens to save time, or all
                # Checking all valid tokens in the encoded sequence is better
                unique_used = set(encoded_tokens)
                max_depth = max([get_token_depth(bpe, t) for t in unique_used]) if unique_used else 0
                
            # E. Compression
            compression = len(tokens) / len(encoded_tokens)
            
            # Log
            res = {
                'bins': n_bins,
                'strategy': strategy,
                'max_depth': max_depth,
                'perplexity': grammar_stats['perplexity'],
                'entropy': grammar_stats['conditional_entropy'],
                'compression': compression,
                'vocab_size': bpe.current_vocab_count
            }
            results.append(res)
            
            print(f"{n_bins:<5} | {strategy:<10} | {max_depth:<8.0f} | {res['perplexity']:<8.2f} | {res['entropy']:<8.2f} | {compression:<6.2f} | {res['vocab_size']:<6}")

    # 3. Save
    df = pd.DataFrame(results)
    df.to_csv("result/grammar_analysis.csv", index=False)
    print("\nanalysis saved to result/grammar_analysis.csv")

if __name__ == "__main__":
    run_grammar_analysis()
