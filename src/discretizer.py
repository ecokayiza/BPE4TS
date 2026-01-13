"""
Time Series Discretization Module.
----------------------------------
This module handles the conversion of continuous time series data into discrete
symbolic sequences (tokens), supporting multiple strategies like SAX (Gaussian),
Uniform, and Quantile binning.

Public Classes:
- TimeSeriesDiscretizer: 
    Scikit-learn style transformer (fit/transform/inverse_transform).
    Strategies: 'quantile', 'uniform', 'uniform_fixed', 'gaussian'.

Dependencies: numpy, sklearn, scipy
"""

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import norm

class TimeSeriesDiscretizer:
    def __init__(self, n_bins: int = 26, strategy: str = 'quantile', range_min: float = -5.0, range_max: float = 5.0):
        self.n_bins = n_bins
        self.strategy = strategy
        self.range_min = range_min
        self.range_max = range_max
        self.bin_edges_ = None
        self.est = None

    def fit(self, X: np.ndarray):
        if self.strategy == 'uniform_fixed':
            self.bin_edges_ = np.linspace(self.range_min, self.range_max, self.n_bins + 1)
        
        elif self.strategy == 'quantile':
            # This corresponds to "Precise Data Distribution P(D)" - Empirical Quantiles
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')
            self.est.fit(X)
            self.bin_edges_ = self.est.bin_edges_[0]
            
        elif self.strategy == 'gaussian':
            # Theoretical Normal Distribution Clamped to range
            # We assume X is already normalized (N(0,1)).
            probs = np.linspace(0, 1, self.n_bins + 1)
            # Clip probabilities slightly to avoid inf at 0 and 1
            epsilon = 1e-6
            probs = np.clip(probs, epsilon, 1-epsilon)
            self.bin_edges_ = norm.ppf(probs)
            
        elif self.strategy == 'uniform':
             if X.ndim == 1:
                X = X.reshape(-1, 1)
             self.est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
             self.est.fit(X)
             self.bin_edges_ = self.est.bin_edges_[0]
            
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Pre-clipping for fixed/gaussian ensures stability
        # Important: Reshape is handled inside methods or here if generic
        
        X_clipped = np.clip(X, self.range_min, self.range_max)
        
        if self.strategy in ['uniform_fixed', 'gaussian']:
            # digitize returns 1..N
            indices = np.digitize(X_clipped, self.bin_edges_, right=False)
            indices = indices - 1
            return np.clip(indices, 0, self.n_bins - 1)
        else:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return self.est.transform(X).flatten().astype(int)

    def inverse_transform(self, X_discrete: np.ndarray) -> np.ndarray:
        if self.strategy in ['uniform_fixed', 'gaussian']:
            # Decode using bin centers
            centers = (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2
            return centers[X_discrete]
        else:
            if X_discrete.ndim == 1:
                X_discrete = X_discrete.reshape(-1, 1)
            return self.est.inverse_transform(X_discrete).flatten()
            
    def get_bin_edges(self):
        return self.bin_edges_
