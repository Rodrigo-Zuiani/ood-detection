"""
Visualization Functions
Functions to create plots for NC1, NC2, and NC3 metrics
Includes comparison plots for ID vs OOD data
"""

import matplotlib.pyplot as plt


def plot_nc1_nc2_metrics(metrics, save_path, title_prefix=""):
    """
    Create a 2x2 subplot showing NC1 and NC2 metrics over training.
    
    Args:
        metrics: Dictionary containing epochs and metric values
        save_path: Path to save the figure
        title_prefix: Optional prefix for plot titles (e.g., "ID" or "OOD")
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
    
    prefix = f"{title_prefix}: " if title_prefix else ""
    
    # NC1: Within-class variance
    axes[0, 0].plot(epochs, wt_var, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_title(f"{prefix}NC1: Within-Class Variance", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Within-Class Variance")
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # NC2: Standard deviation of distances
    axes[0, 1].plot(epochs, stds, 'r-o', linewidth=2, markersize=6)
    axes[0, 1].set_title(f"{prefix}NC2: Std of Mean Distances", fontsize=12, fontweight='bold')
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
    axes[1, 1].set_title(f"{prefix}NC2: Coefficient of Variation", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("CV (Std/Mean)")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved NC1/NC2 plot to {save_path}")


def plot_nc3_metrics(metrics, save_path, title_prefix=""):
    """
    Create a plot showing NC3 (self-duality) metrics over training.
    
    Args:
        metrics: Dictionary containing epochs and metric values
        save_path: Path to save the figure
        title_prefix: Optional prefix for plot title (e.g., "ID" or "OOD")
    """
    epochs = metrics['epochs']
    
    # Extract metric lists in epoch order
    nc3_means = [metrics['nc3']['mean_cos'][e] for e in epochs]
    nc3_stds = [metrics['nc3']['std_cos'][e] for e in epochs]
    
    prefix = f"{title_prefix}: " if title_prefix else ""
    
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, nc3_means, 'k-o', linewidth=2, markersize=6, label="Mean Cosine Similarity")
    plt.fill_between(
        epochs,
        [m - s for m, s in zip(nc3_means, nc3_stds)],
        [m + s for m, s in zip(nc3_means, nc3_stds)],
        alpha=0.3,
        label="±1 Std Dev"
    )
    plt.ylim(0, 1.05)
    plt.title(f"{prefix}NC3: Self-Duality (Weight || Mean)", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved NC3 plot to {save_path}")


def plot_nc_comparison(metrics_id, metrics_ood, save_path):
    """
    Create comparison plots showing ID vs OOD metrics side by side.
    
    Args:
        metrics_id: Dictionary containing in-distribution metrics
        metrics_ood: Dictionary containing out-of-distribution metrics
        save_path: Path to save the figure
    """
    epochs_id = metrics_id['epochs']
    epochs_ood = metrics_ood['epochs']
    
    # Extract ID metrics
    wt_var_id = [metrics_id['nc1']['within_class_var'][e] for e in epochs_id]
    mean_dist_id = [metrics_id['nc2']['mean_dist'][e] for e in epochs_id]
    cv_id = [metrics_id['nc2']['cv'][e] for e in epochs_id]
    nc3_id = [metrics_id['nc3']['mean_cos'][e] for e in epochs_id]
    
    # Extract OOD metrics
    wt_var_ood = [metrics_ood['nc1']['within_class_var'][e] for e in epochs_ood]
    mean_dist_ood = [metrics_ood['nc2']['mean_dist'][e] for e in epochs_ood]
    cv_ood = [metrics_ood['nc2']['cv'][e] for e in epochs_ood]
    nc3_ood = [metrics_ood['nc3']['mean_cos'][e] for e in epochs_ood]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # NC1: Within-class variance
    axes[0, 0].plot(epochs_id, wt_var_id, 'b-o', linewidth=2, markersize=6, label='ID (CIFAR-100)')
    axes[0, 0].plot(epochs_ood, wt_var_ood, 'r--s', linewidth=2, markersize=6, label='OOD (SVHN)')
    axes[0, 0].set_title("NC1: Within-Class Variance", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Within-Class Variance")
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # NC2: Mean pairwise distance
    axes[0, 1].plot(epochs_id, mean_dist_id, 'b-o', linewidth=2, markersize=6, label='ID (CIFAR-100)')
    axes[0, 1].plot(epochs_ood, mean_dist_ood, 'r--s', linewidth=2, markersize=6, label='OOD (SVHN)')
    axes[0, 1].set_title("NC2: Mean Pairwise Distance", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Distance")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # NC2: Coefficient of variation
    axes[1, 0].plot(epochs_id, cv_id, 'b-o', linewidth=2, markersize=6, label='ID (CIFAR-100)')
    axes[1, 0].plot(epochs_ood, cv_ood, 'r--s', linewidth=2, markersize=6, label='OOD (SVHN)')
    axes[1, 0].set_title("NC2: Coefficient of Variation", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("CV (Std/Mean)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # NC3: Self-duality
    # axes[1, 1].plot(epochs_id, nc3_id, 'b-o', linewidth=2, markersize=6, label='ID (CIFAR-100)')
    # axes[1, 1].plot(epochs_ood, nc3_ood, 'r--s', linewidth=2, markersize=6, label='OOD (SVHN)')
    # axes[1, 1].set_title("NC3: Mean Cosine Similarity", fontsize=12, fontweight='bold')
    # axes[1, 1].set_xlabel("Epoch")
    # axes[1, 1].set_ylabel("Cosine Similarity")
    # axes[1, 1].set_ylim(0, 1.05)
    # axes[1, 1].grid(True, alpha=0.3)
    # axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("NEURAL COLLAPSE COMPARISON: ID vs OOD")
    print("="*80)
    
    final_epoch_id = epochs_id[-1]
    final_epoch_ood = epochs_ood[-1]
    
    print(f"\nFinal Epoch Comparison (Epoch {final_epoch_id}):")
    print("-"*80)
    print(f"{'Metric':<30} {'ID (CIFAR-100)':<20} {'OOD (SVHN)':<20}")
    print("-"*80)
    print(f"{'NC1: Within-class var':<30} {wt_var_id[-1]:<20.6f} {wt_var_ood[-1]:<20.6f}")
    print(f"{'NC2: Mean distance':<30} {mean_dist_id[-1]:<20.4f} {mean_dist_ood[-1]:<20.4f}")
    print(f"{'NC2: CV':<30} {cv_id[-1]:<20.4f} {cv_ood[-1]:<20.4f}")
    print(f"{'NC3: Mean cosine sim':<30} {nc3_id[-1]:<20.4f} {nc3_ood[-1]:<20.4f}")
    print("-"*80)
    
    print("\nInterpretation:")
    print("- Higher NC1 (within-class var) for OOD → OOD features less collapsed")
    print("- Different NC2 values → Different class separation patterns")
    print("- Lower NC3 for OOD → Classifier weights less aligned with OOD means")
    print("="*80 + "\n")