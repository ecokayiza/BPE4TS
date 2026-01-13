# Time Series BPE Tokenizer Experiment (时间序列 BPE 分词器实验)

This project implements and evaluates a **Byte Pair Encoding (BPE)** tokenizer adapted for Time Series data. By discretizing continuous time series values into symbols and then applying BPE, we can compress the series into meaningful "motifs" (tokens) of variable lengths.

Methodology based on: *Time Series Tokenization via BPE (Conceptual)*.

## 📂 Project Structure (项目结构)

```text
d:\Projects\BPE
├── data/                   # Dataset folder (ETTm1.csv)
├── result/                 # Experiment results & visualizations
│   ├── detailed_tokenization.png  # Viz of reconstruction & token segments
│   ├── pareto_frontier.png        # Compression vs MSE trade-off plot
│   ├── tokenization_gallery.png   # Gallery of tokenization examples
│   └── experiment_grid_results.csv # All metrics
├── src/                    # Source code
│   ├── main.py             # Entry point (2-stage search pipeline)
│   ├── data_loader.py      # Data downloading & loading
│   ├── discretizer.py      # SAX-like discretization (Uniform/Quantile/Gaussian)
│   ├── tokenizer.py        # BPE implementation (Train/Encode/Decode)
│   └── visualization.py    # Plotting utilities
├── uv.lock                 # Dependency lock file
└── pyproject.toml          # Project configuration
```

## 🚀 Quick Start (快速开始)

Make sure you have [uv](https://github.com/astral-sh/uv) installed.

1. **Run the Experiment (运行实验)**:
   This will download data, preprocess, run a 2-stage (Coarse-to-Fine) parameter search, and generate visualizations.
   ```bash
   uv run -m src.main
   ```

2. **Check Results (查看结果)**:
   Go to the `result/` folder to see the generated plots and CSV.

## 🔬 Methodology (方法论)

1.  **Preprocessing (预处理)**:
    - Z-Score Normalization (Zero Mean, Unit Var).
    - Truncation to range $[-5, 5]$.

2.  **Discretization (离散化)**:
    - Mapping continuous values to discrete symbols using:
        - `quantile`: Empirical distribution (SAX-like).
        - `gaussian`: Theoretical Normal distribution.
        - `uniform_fixed`: Fixed intervals in $[-5, 5]$.

3.  **BPE Tokenization (BPE 分词)**:
    - Iteratively merges the most frequent adjacent symbol pairs into new tokens.
    - Result: A vocabulary of variable-length motifs representing shapes like "rise", "fall", "peak".

4.  **Optimization (优化策略)**:
    - **Metric**: "Distance to Ideal" heuristic ($\sqrt{MSE_{norm}^2 + (1-Compression_{norm})^2}$).
    - **Strategy**: 2-Stage Search (Coarse Grid -> Fine Local Search).

## 📊 Experiment Results (实验结果)

**Dataset**: ETTm1 (Univariate 'OT' column - Oil Temperature).

### Best Configuration (最佳模型配置)
After an automated Coarse-to-Fine search:
- **Bins**: **55**
- **Strategy**: **Uniform Fixed**
- **Min Frequency**: **2**

### Performance Metrics (性能指标)
- **Reconstruction MSE**: **0.002784** (Very high fidelity)
- **Compression Ratio**: **~8.9x** (Significant reduction in sequence length)

### Visualizations (可视化解读)

1.  **`pareto_frontier.png`**:
    - Shows the trade-off between **MSE** (x-axis) and **Compression Ratio** (y-axis).
    - **Insight**: `uniform_fixed` strategy generally yields higher compression but slightly higher error than `quantile`.

2.  **`detailed_tokenization.png`**:
    - **Top**: Red dashed line (Reconstructed) closely follows Black line (Original).
    - **Middle**: Shows actual **Tokens**. Long colored segments indicate the model learned long-term patterns.
    - **Bottom**: Histogram of token durations.

3.  **`tokenization_gallery.png`**:
    - A gallery of 8 distinct time windows, showing how the tokenizer adapts to different data shapes (trends, noise, seasonality) across the dataset.

## 📝 Key Findings (主要发现)

1.  **Uniform Strategy Wins**: on this normalized dataset, simple uniform binning proved more robust for balancing compression and error compared to quantile binning (which is often too granular).
2.  **Adaptive Length**: The BPE successfully identified motifs of varying lengths. Stable regions are compressed into single "long tokens", while noisy regions use "short tokens".
3.  **Fine-Tuning Matters**: The 2-stage search successfully refined the bin count from a coarse 50 to a precise 55, improving the trade-off metrics.
