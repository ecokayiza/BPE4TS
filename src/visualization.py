import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pareto_frontier(results_df, output_path):
    """
    Plots the Pareto Frontier of Compression Ratio vs MSE for different strategies.
    """
    plt.figure(figsize=(10, 6))
    
    strategies = results_df['strategy'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, strategy in enumerate(strategies):
        subset = results_df[results_df['strategy'] == strategy]
        plt.scatter(
            subset['compression_ratio'], 
            subset['mse'], 
            label=strategy, 
            s=subset['n_bins']*3, # Size by bin count
            alpha=0.6,
            marker=markers[i % len(markers)],
            edgecolors='k',
            linewidth=0.5
        )
        
    plt.xlabel("Compression Ratio")
    plt.ylabel("MSE (Reconstruction Error)")
    plt.title("Pareto Frontier: Compression vs Quality\n(Size = n_bins)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Pareto plot saved to {output_path}")

def plot_detailed_tokenization(original_series, compressed_tokens, bpe_model, discretizer, output_path, viz_len=200, info_dict=None):
    """
    Generates a detailed 3-panel visualization of the tokenization process.
    """
    # 1. Prepare data for the window
    viz_original = original_series[:viz_len]
    
    # Slice tokens that cover the window
    current_len = 0
    viz_tokens_subset = []
    token_lengths = []
    
    for t in compressed_tokens:
        decoded_pattern = bpe_model.decode_token(t)
        l = len(decoded_pattern)
        
        # If the FIRST token is huge, we must include it, but clip the view.
        if current_len == 0 and l > viz_len:
            viz_tokens_subset.append(t)
            token_lengths.append(l)
            current_len += l
            break
            
        if current_len + l > viz_len:
            break
        viz_tokens_subset.append(t)
        token_lengths.append(l)
        current_len += l
    
    # Adjust viz_len to matches full tokens
    if current_len > 0:
        viz_original = original_series[:current_len]
    else:
        # Fallback if something went wrong (e.g. no tokens)
        viz_original = original_series[:viz_len]
        print("Warning: No tokens selected for visualization.")

    # Reconstruct full sequence for this window
    reconstructed_full = []
    for t in viz_tokens_subset:
        decoded_pattern = bpe_model.decode_token(t)
        # map back to values
        reconstructed_full.extend(discretizer.inverse_transform(np.array(decoded_pattern)))

    mse = np.mean((viz_original - reconstructed_full) ** 2)

    # 2. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    if info_dict:
        # Create a title string from the best parameters
        title_str = (f"Best Configuration: Bins={info_dict.get('n_bins')}, "
                     f"Strategy={info_dict.get('strategy')}, "
                     f"MinFreq={info_dict.get('min_freq')}\n"
                     f"MSE={info_dict.get('mse'):.5f}, "
                     f"Comp.Ratio={info_dict.get('compression_ratio'):.2f}x")
        fig.suptitle(title_str, fontsize=14, fontweight='bold')
        # Adjust layout to make room for suptitle
        plt.subplots_adjust(top=0.92)

    # Panel 1: Reconstruction
    axes[0].plot(viz_original, label='Original', color='black', alpha=0.7, linewidth=1.5)
    axes[0].plot(reconstructed_full, label='Reconstructed', color='red', alpha=0.7, linestyle='--', linewidth=1.5)
    axes[0].set_title(f"Reconstruction Quality (MSE: {mse:.4f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Token Segments
    axes[1].plot(viz_original, color='gray', alpha=0.2, linewidth=1) 
    
    start_idx = 0
    colors = plt.cm.tab10.colors 
    
    for i, (t, length) in enumerate(zip(viz_tokens_subset, token_lengths)):
        end_idx = start_idx + length
        segment = viz_original[start_idx:end_idx]
        x_range = range(start_idx, end_idx)
        
        color = colors[i % len(colors)]
        axes[1].plot(x_range, segment, color=color, linewidth=2.5)
        
        # Boundary line
        if start_idx > 0:
            axes[1].axvline(x=start_idx - 0.5, color='gray', linestyle=':', alpha=0.5)
            
        # Annotation
        if length >= 1: 
            mid_x = start_idx + length / 2 - 0.5
            mid_y = np.max(segment) + (np.max(viz_original) - np.min(viz_original))*0.05
            axes[1].text(mid_x, mid_y, str(t), ha='center', fontsize=8, color=color, fontweight='bold')

        start_idx = end_idx
        
    axes[1].set_title("Adaptive Tokenization: Segments covered by single BPE tokens")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Duration
    axes[2].bar(range(len(token_lengths)), token_lengths, color='skyblue', edgecolor='black', alpha=0.7)
    axes[2].set_title("Duration of each Token (Adaptive Resolution)")
    axes[2].set_xlabel("Token Index")
    axes[2].set_ylabel("Time Steps")
    axes[2].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Detailed visualization saved to {output_path}")

def plot_tokenization_gallery(original_series, compressed_tokens, bpe_model, output_path, n_examples=8, window_len=200, info_dict=None):
    """
    Generates a multi-row gallery of tokenization examples.
    """
    # 1. Map all tokens to timeline
    token_boundaries = []
    current_idx = 0
    for t in compressed_tokens:
        # We only need length here
        
        # Optimize: bpe_model.tokens[t] should give the tuple directly if accessible, 
        # but decode_token handles recursive expansion.
        pattern_len = len(bpe_model.decode_token(t))
        token_boundaries.append((current_idx, current_idx + pattern_len, t))
        current_idx += pattern_len
    
    total_len = current_idx
    series_len = len(original_series)
    if total_len > series_len:
         total_len = series_len 

    # 2. Select windows
    max_start = max(0, total_len - window_len)
    start_indices = np.linspace(0, max_start, n_examples, dtype=int)
    
    # 3. Plot
    fig, axes = plt.subplots(n_examples, 1, figsize=(15, 2.5 * n_examples))
    if n_examples == 1: axes = [axes]
    
    if info_dict:
         params = f"Bins={info_dict.get('n_bins')}, Strat={info_dict.get('strategy')}, MinFreq={info_dict.get('min_freq')}"
         fig.suptitle(f"Tokenization Gallery (Selected Windows) - {params}", fontsize=14, fontweight='bold', y=1.0) # Move title up
         plt.subplots_adjust(top=0.95, hspace=0.6)
    else:
        plt.subplots_adjust(hspace=0.6)

    colors = plt.cm.tab10.colors 
    
    for i, start_idx in enumerate(start_indices):
        ax = axes[i]
        end_idx = min(start_idx + window_len, total_len)
        vis_len_actual = end_idx - start_idx
        
        # Background: Original Data
        segment_original = original_series[start_idx:end_idx]
        ax.plot(range(start_idx, end_idx), segment_original, color='gray', alpha=0.3, linewidth=1)
        
        # Calculate dynamic y-offset for labels
        y_min, y_max = np.min(segment_original), np.max(segment_original)
        y_range = y_max - y_min if (y_max - y_min) > 1e-9 else 1.0
        label_offset = y_range * 0.05
        
        # Ensure plot has room for labels
        ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.2)
        
        # Find relevant tokens
        row_tokens = [
            (s, e, t) for (s, e, t) in token_boundaries 
            if s < end_idx and e > start_idx
        ]
        
        color_cycle_idx = 0
        for (s, e, t_id) in row_tokens:
            c = colors[color_cycle_idx % len(colors)]
            color_cycle_idx += 1
            
            # Intersection with window
            draw_s = max(s, start_idx)
            draw_e = min(e, end_idx)
            
            if draw_s >= draw_e: continue
            
            # Slice original data
            data_chunk = original_series[draw_s:draw_e]
            ax.plot(range(draw_s, draw_e), data_chunk, color=c, linewidth=2.0)
            
            # Boundary
            if s > start_idx:
                ax.axvline(x=s - 0.5, color='gray', linestyle=':', alpha=0.5)
                
            # Label
            mid = s + (e - s) / 2 - 0.5
            if start_idx <= mid < end_idx:
                local_max = np.max(data_chunk)
                ax.text(mid, local_max + label_offset, str(t_id), ha='center', fontsize=8, color=c, fontweight='bold')
                
        ax.set_title(f"Window: {start_idx} - {end_idx}", fontsize=9, loc='left')
        ax.set_ylabel("Val")
        ax.grid(True, alpha=0.2)
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Gallery visualization saved to {output_path}")
