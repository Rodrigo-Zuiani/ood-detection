"""
Visualization Functions
Functions to create plots for NC1, NC2, and NC3 metrics
"""

import matplotlib.pyplot as plt


def plot_nc1_nc2_metrics(metrics, save_path):
    """
    Create a 2x2 subplot showing NC1 and NC2 metrics over training.
    
    Args:
        metrics: Dictionary containing epochs and metric values
        save_path: Path to save the figure
    """
    epochs = metrics['epochs']
    
    # Extract metric lists in epoch order
    wt_var = [metrics['nc1']['within_class_var'][e] for e in epochs]
    stds = [metrics['nc2']['std_dist'][e] for e in epochs]
    mean_dists = [metrics['nc2']['mean_dist'][e] for e in epochs]
    min_dists = [metrics['nc2']['min_dist'][e] for e in epochs]
    max_dists = [metrics['nc2']['max_dist'][e] for e in epochs]
    cv = [metrics['nc2']['cv'][e] for e in epochs]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # NC1: Within-class variance
    axes[0, 0].plot(epochs, wt_var, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_title("NC1: Within-Class Variance", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Within-Class Variance")
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # NC2: Standard deviation of distances
    axes[0, 1].plot(epochs, stds, 'r-o', linewidth=2, markersize=6)
    axes[0, 1].set_title("NC2: Std of Mean Distances", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Standard Deviation")
    axes[0, 1].grid(True, alpha=0.3)
    
    # NC2: Mean pairwise distance with range
    axes[1, 0].plot(epochs, mean_dists, 'g-o', linewidth=2, markersize=6, label='Mean')
    axes[1, 0].fill_between(
                            epochs,
                            [m - s for m, s in zip(mean_dists, stds)],
                            [m + s for m, s in zip(mean_dists, stds)],
                            alpha=0.3,
                            label='±1 std'
                        )
    axes[1, 0].set_title("NC2: Mean Pairwise Distance", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Distance")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # NC2: Coefficient of variation
    axes[1, 1].plot(epochs, cv, 'm-o', linewidth=2, markersize=6)
    axes[1, 1].set_title("NC2: Coefficient of Variation", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("CV (Std/Mean)")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved NC1/NC2 plot to {save_path}")


def plot_nc3_metrics(metrics, save_path):
    """
    Create a plot showing NC3 (self-duality) metrics over training.
    
    Args:
        metrics: Dictionary containing epochs and metric values
        save_path: Path to save the figure
    """
    epochs = metrics['epochs']
    
    # Extract metric lists in epoch order
    nc3_means = [metrics['nc3']['mean_cos'][e] for e in epochs]
    nc3_stds = [metrics['nc3']['std_cos'][e] for e in epochs]
    
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, nc3_means, 'k-o', linewidth=2, markersize=6, label="Mean Cosine Similarity")
    plt.fill_between(
        epochs,
        [m - s for m, s in zip(nc3_means, nc3_stds)],
        [m + s for m, s in zip(nc3_means, nc3_stds)],
        alpha=0.3,
        color='gray',
        label="±1 Std Dev"
    )
    plt.ylim(0, 1.05)
    plt.title("NC3: Self-Duality (Weight || Mean)", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved NC3 plot to {save_path}")
