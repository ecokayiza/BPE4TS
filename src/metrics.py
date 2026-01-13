"""
Metric Calculation Module.
--------------------------
This module provides statistical and information-theoretic metrics for evaluating
the quality of tokenization, specifically focusing on "Grammar Discovery".

Public Interface:
- calculate_grammar_quality(tokens): 
    Calculates Perplexity and Entropy of the token sequence.
    Useful for unsupervised model selection (low perplexity = good structure).

Dependencies: numpy, math, collections
"""

import numpy as np
import math
from collections import Counter
from sklearn.metrics import silhouette_score
from scipy.signal import resample

def calculate_motif_quality(tokens, bpe_model, original_series, max_samples=5000):
    """
    Evaluates the "Visual/Semantic Consistency" of the generated tokens.
    High consistency means that every time Token X appears, the raw data shap
    looks effectively the same (High Correlation / Low Variance).
    
    Metric: Weighted Average Pearson Correlation of instances to their cluster centroid.
    
    Args:
        tokens: List of token IDs.
        bpe_model: Trained BPE model.
        original_series: Continuous data.
        
    Returns:
        dict: {'consistency_score': float, 'explained_variance': float}
    """
    # 1. Group indices by token
    token_instances = {}
    
    current_idx = 0
    series_len = len(original_series)
    
    # We sampled for speed if needed, but iterating tokens is fast
    # Gathering segments is the cost.
    
    # To limit memory/time, we can just process the first N tokens or sample
    
    for t in tokens:
        decoded = bpe_model.decode_token(t)
        l = len(decoded)
        
        start = current_idx
        end = current_idx + l
        
        if end > series_len:
            break
            
        # Only consider "motifs" with length > 1 for correlation analysis
        # Length 1 tokens are "points", correlation is undefined/meaningless (or 1.0)
        if l > 1:
            if t not in token_instances:
                token_instances[t] = []
            
            # Extract and Resample to fixed length for comparison
            # Using 16 points is enough for shape matching
            segment = original_series[start:end]
            if len(segment) > 1:
                # Normalize amplitude for shape comparison?
                # Pearson correlation is invariant to scale and shift.
                # So we just need to resample time.
                resampled = resample(segment, 16)
                token_instances[t].append(resampled)
        
        current_idx += l
        
    if not token_instances:
        return {"consistency_score": 0.0}

    # 2. Calculate Consistency per Token
    weighted_corr_sum = 0
    total_weight = 0
    
    # Limit number of clusters processed for speed
    # Sort by frequency
    sorted_tokens = sorted(token_instances.keys(), key=lambda k: len(token_instances[k]), reverse=True)
    
    for t in sorted_tokens[:100]: # Top 100 most frequent tokens
        instances = np.array(token_instances[t])
        count = len(instances)
        
        if count < 2:
            continue
            
        # Centroid
        centroid = np.mean(instances, axis=0)
        centroid_std = np.std(centroid)
        
        # If centroid is flat, correlation is undefined. 
        # But if instances are also flat, they are consistent.
        # If instances are noisy, they are inconsistent.
        # We check variance of instances.
        
        if centroid_std < 1e-6:
             # Flat line motif. Check if instances are close to constant.
             # Calculate MSE instead?
             # Let's assign 1.0 if instance variance is low, else 0.
             inst_stds = np.std(instances, axis=1)
             avg_inst_std = np.mean(inst_stds)
             corr = 1.0 if avg_inst_std < 1e-2 else 0.0
        else:
            # Correlation of each instance with centroid
            # (centered_inst * centered_centroid) / (std_inst * std_centroid)
            
            # Vectorized Correlation
            X = instances - np.mean(instances, axis=1, keepdims=True)
            Y = centroid - np.mean(centroid)
            
            X_norm = np.linalg.norm(X, axis=1)
            Y_norm = np.linalg.norm(Y)
            
            # Avoid divide by zero
            valid_mask = X_norm > 1e-9
            if Y_norm > 1e-9 and np.any(valid_mask):
                dot_prods = np.dot(X[valid_mask], Y)
                corrs = dot_prods / (X_norm[valid_mask] * Y_norm)
                corr = np.mean(corrs)
            else:
                corr = 0.0
                
        weighted_corr_sum += corr * count
        total_weight += count
        
    score = weighted_corr_sum / total_weight if total_weight > 0 else 0.0
    
    return {
        "consistency_score": score
    }

def calculate_grammar_quality(tokens):
    """
    Calculates statistical quality of the token sequence using Information Theory.
    
    We want a tokenizer that makes the sequence 'predictable' (Low Perplexity),
    meaning it has found real grammatical structures (motifs), 
    rather than just random noise (High Perplexity).

    Returns:
       dict containing:
       - conditional_entropy: H(Next|Current) in bits
       - perplexity: 2^Entropy (Average branching factor)
       - h_x: Marginal Entropy
       - unique_tokens: Count of unique tokens used
    """
    if len(tokens) < 2:
        return {"conditional_entropy": 0.0, "perplexity": 0.0}
        
    # 1. Count Frequencies
    tokens = list(tokens)
    
    bigrams = Counter(zip(tokens[:-1], tokens[1:]))
    unigrams = Counter(tokens)
    
    total_bigrams = sum(bigrams.values())
    total_unigrams = sum(unigrams.values())
    
    # 2. Calculate H(X) - Marginal Entropy of current token
    h_x = 0
    for cnt in unigrams.values():
        p = cnt / total_unigrams
        if p > 0:
            h_x -= p * math.log2(p)
        
    # 3. Calculate H(X, Y) - Joint Entropy of (Current, Next)
    h_xy = 0
    for cnt in bigrams.values():
        p = cnt / total_bigrams
        if p > 0:
            h_xy -= p * math.log2(p)
    
    # 4. Calculate Conditional Entropy H(Y | X) = H(X, Y) - H(X)
    conditional_entropy = h_xy - h_x
    
    # Perplexity = 2^Entropy
    perplexity = 2 ** conditional_entropy
    
    return {
        "h_x": h_x,
        "h_xy": h_xy,
        "conditional_entropy": conditional_entropy,
        "perplexity": perplexity,
        "unique_tokens": len(unigrams)
    }
