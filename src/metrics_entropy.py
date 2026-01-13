import numpy as np
import math
from collections import Counter

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
    """
    if len(tokens) < 2:
        return {"conditional_entropy": 0.0, "perplexity": 0.0}
        
    # 1. Count Frequencies
    # Casting to list ensures we iterate over a tangible sequence, 
    # useful if tokens is a generator or pd.Series
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
    # How much uncertainty remains about the Next token, given the Current token?
    conditional_entropy = h_xy - h_x
    
    # Perplexity = 2^Entropy
    # This represents the "average number of choices" for the next token.
    # Lower is better (more structured).
    perplexity = 2 ** conditional_entropy
    
    return {
        "h_x": h_x,
        "h_xy": h_xy,
        "conditional_entropy": conditional_entropy,
        "perplexity": perplexity,
        "unique_tokens": len(unigrams)
    }
