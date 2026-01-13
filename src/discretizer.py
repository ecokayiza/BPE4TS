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
            # We want equal probability mass under the Gaussian curve.
            # linspace(0, 1, n_bins+1) gives cumulative probabilities.
            # norm.ppf gives the value (quantile) for that probability.
            probs = np.linspace(0, 1, self.n_bins + 1)
            # Clip probabilities slightly to avoid inf at 0 and 1
            epsilon = 1e-6
            probs = np.clip(probs, epsilon, 1-epsilon)
            self.bin_edges_ = norm.ppf(probs)
            # Ensure edges cover the truncated range or data range if needed
            # Usually SAX open-ends the first and last bin (-inf, +inf)
            # but for reconstruction we need concrete centers.
            
        elif self.strategy == 'uniform':
             if X.ndim == 1:
                X = X.reshape(-1, 1)
             self.est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
             self.est.fit(X)
             self.bin_edges_ = self.est.bin_edges_[0]
            
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Pre-clipping for fixed/gaussian ensures stability
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
            # For Gaussian, the probabilistic center is strictly NOT the geometric center of edges
            # mean value of truncated normal in that interval.
            # But geometric center is a fine approximation for visualization.
            # Let's use geometric center of edges for simplicity, 
            # except for outer bins which might be large.
            centers = (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2
            return centers[X_discrete]
        else:
            if X_discrete.ndim == 1:
                X_discrete = X_discrete.reshape(-1, 1)
            return self.est.inverse_transform(X_discrete).flatten()
            
    def get_bin_edges(self):
        return self.bin_edges_
