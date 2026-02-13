"""
Neural Collapse Analysis - Main Script
Analyzes NC1, NC2, and NC3 metrics across training checkpoints
Includes both In-Distribution (CIFAR-100) and Out-of-Distribution (SVHN) analysis
"""

import os
import torch
import copy
from pathlib import Path
from .utils.models_utils import setup_reproducibility, load_model_checkpoint, extract_features
from .utils.NC_utils import compute_nc1_metrics, compute_nc2_metrics, compute_nc3_metrics
from .utils.plot_utils import plot_nc1_nc2_metrics, plot_nc3_metrics, plot_nc_comparison


def main():
    # Config and Setup
    config = {
        'checkpoint_folder': "src/checkpoints",
        'plots_folder': "src/plots",
        'batch_size': 128,
        'num_workers': 2,
        'seed': 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    Path(config['plots_folder']).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(config['device'])
    setup_reproducibility(config['seed'])
    
    # Import here to avoid circular dependencies
    from .models.resnet import BasicBlock, ResNet18
    from .dataset.cifar100 import get_cifar100_loaders
    from .dataset.svhn import get_svhn_ood_loader
    
    # Initialize model
    model = ResNet18(
        block=BasicBlock, 
        num_blocks=[2, 2, 2, 2], 
        num_classes=100
    ).to(device)
    
    # Get data loaders
    g = torch.Generator()
    g.manual_seed(config['seed'])
    
    trainloader, testloader, classes = get_cifar100_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        generator=g,
        worker_init_fn=lambda worker_id: setup_reproducibility(
            config['seed'] + worker_id
        )
    )

    testLoader_OOD, classes_OOD = get_svhn_ood_loader(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'] # Use CIFAR normalization for fair comparison
    )
    
    # Checkpoints
    checkpoint_files = sorted(os.listdir(config['checkpoint_folder']))
    print(f"Found {len(checkpoint_files)} checkpoint files")
    print("=" * 80)
    
    # Storage for metrics across epochs
    metrics = {
        'epochs': [],
        'nc1': {'within_class_var': {}, 'within_total_ratio': {}},
        'nc2': {
            'mean_dist': {},
            'std_dist': {},
            'min_dist': {},
            'max_dist': {},
            'cv': {}
        },
        'nc3': {
            'mean_cos': {},
            'std_cos': {},
            'min_cos': {}
        }
    }
    
    # Deep copy for OOD metrics
    metrics_OOD = copy.deepcopy(metrics)
    
    # Process all checkpoints
    for cp_file in checkpoint_files:
        checkpoint_path = os.path.join(config['checkpoint_folder'], cp_file)
        epoch = load_model_checkpoint(model, checkpoint_path, device)
        metrics['epochs'].append(epoch)
        metrics_OOD['epochs'].append(epoch)
        
        print(f"\nEpoch {epoch}:")
        
        # Extract features from training data (ID) and OOD data
        features, labels = extract_features(model, trainloader, device)
        features_OOD, labels_OOD = extract_features(model, testLoader_OOD, device)
        print(f"  ID Features shape: {features.shape}, Labels shape: {labels.shape}")
        print(f"  OOD Features shape: {features_OOD.shape}, Labels shape: {labels_OOD.shape}")
        
        # Compute class means
        num_classes = labels.max().item() + 1
        class_means = compute_class_means(features, labels, num_classes)

        num_classes_OOD = labels_OOD.max().item() + 1
        class_means_OOD = compute_class_means(features_OOD, labels_OOD, num_classes_OOD)
        
        # ========================================================
        # NC1: Within-class variance
        # ========================================================
        nc1_results = compute_nc1_metrics(features, labels, class_means)
        nc1_results_OOD = compute_nc1_metrics(features_OOD, labels_OOD, class_means_OOD)
        
        # print("\n  NC1 Metrics (ID / OOD):")
        # print(f"    Within-class variance: {nc1_results['within_class_var']:.6f} / {nc1_results_OOD['within_class_var']:.6f}")
        # print(f"    Within/Total ratio: {nc1_results['within_total_ratio']:.6f} / {nc1_results_OOD['within_total_ratio']:.6f}")
        
        # Store ID metrics
        metrics['nc1']['within_class_var'][epoch] = nc1_results['within_class_var']
        metrics['nc1']['within_total_ratio'][epoch] = nc1_results['within_total_ratio']
        
        # Store OOD metrics
        metrics_OOD['nc1']['within_class_var'][epoch] = nc1_results_OOD['within_class_var']
        metrics_OOD['nc1']['within_total_ratio'][epoch] = nc1_results_OOD['within_total_ratio']
        
        # ========================================================
        # NC2: Class mean distances
        # ========================================================
        nc2_results = compute_nc2_metrics(class_means)
        nc2_results_OOD = compute_nc2_metrics(class_means_OOD)
        
        # print("\n  NC2 Metrics (ID / OOD):")
        # print(f"    Mean distance: {nc2_results['mean_dist']:.4f} / {nc2_results_OOD['mean_dist']:.4f}")
        # print(f"    Std distance: {nc2_results['std_dist']:.4f} / {nc2_results_OOD['std_dist']:.4f}")
        # print(f"    Coefficient of variation: {nc2_results['cv']:.4f} / {nc2_results_OOD['cv']:.4f}")
        
        # Store ID metrics
        for key in ['mean_dist', 'std_dist', 'min_dist', 'max_dist', 'cv']:
            metrics['nc2'][key][epoch] = nc2_results[key]
        
        # Store OOD metrics
        for key in ['mean_dist', 'std_dist', 'min_dist', 'max_dist', 'cv']:
            metrics_OOD['nc2'][key][epoch] = nc2_results_OOD[key]
        
        # ========================================================
        # NC3: Self-Duality
        # ========================================================
        nc3_results = compute_nc3_metrics(model, class_means)
        # nc3_results_OOD = compute_nc3_metrics(model, class_means_OOD)
        
        # print("\n  NC3 Metrics (ID / OOD):")
        # print(f"    Mean cosine similarity: {nc3_results['mean_cos']:.4f} / {nc3_results_OOD['mean_cos']:.4f}")
        # print(f"    Std cosine similarity: {nc3_results['std_cos']:.4f} / {nc3_results_OOD['std_cos']:.4f}")
        # print(f"    Min cosine similarity: {nc3_results['min_cos']:.4f} / {nc3_results_OOD['min_cos']:.4f}")
        
        # Store ID metrics
        for key in ['mean_cos', 'std_cos', 'min_cos']:
            metrics['nc3'][key][epoch] = nc3_results[key]
        
        # Store OOD metrics
        # for key in ['mean_cos', 'std_cos', 'min_cos']:
        #     metrics_OOD['nc3'][key][epoch] = nc3_results_OOD[key]
        
        print("=" * 80)
    
    # ========================================================
    # Generate Plots
    # ========================================================
    metrics['epochs'].sort()
    metrics_OOD['epochs'].sort()
    
    # Plot ID metrics
    plot_nc1_nc2_metrics(
        metrics, 
        save_path=os.path.join(config['plots_folder'], 'nc_metrics_ID.png'),
        title_prefix="ID (CIFAR-100)"
    )
    
    plot_nc3_metrics(
        metrics,
        save_path=os.path.join(config['plots_folder'], 'nc3_self_duality_ID.png'),
        title_prefix="ID (CIFAR-100)"
    )
    
    # Plot OOD metrics
    plot_nc1_nc2_metrics(
        metrics_OOD, 
        save_path=os.path.join(config['plots_folder'], 'nc_metrics_OOD.png'),
        title_prefix="OOD (SVHN)"
    )
    
    # plot_nc3_metrics(
    #     metrics_OOD,
    #     save_path=os.path.join(config['plots_folder'], 'nc3_self_duality_OOD.png'),
    #     title_prefix="OOD (SVHN)"
    # )
    
    # Plot comparison (ID vs OOD)
    plot_nc_comparison(
        metrics, 
        metrics_OOD,
        save_path=os.path.join(config['plots_folder'], 'nc_comparison_ID_vs_OOD.png')
    )
    
    print("\nAnalysis complete. Plots saved.")
    print(f"  ID plots: nc_metrics_ID.png, nc3_self_duality_ID.png")
    print(f"  OOD plots: nc_metrics_OOD.png, nc3_self_duality_OOD.png")
    print(f"  Comparison plot: nc_comparison_ID_vs_OOD.png")


def compute_class_means(features, labels, num_classes):
    """
    Compute mean feature vector for each class.
    
    Args:
        features: Tensor of shape (N, d) with feature vectors
        labels: Tensor of shape (N,) with class labels
        num_classes: Number of classes
        
    Returns:
        class_means_centered: Tensor of shape (num_classes, d) with centered class means
    """
    d = features.size(1)
    class_means = torch.zeros(num_classes, d)
    
    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        class_means[c] = features[idx].mean(dim=0)
    
    # Center class means by subtracting global mean
    global_class_mean = class_means.mean(dim=0, keepdim=True)
    class_means_centered = class_means - global_class_mean
    
    return class_means_centered


if __name__ == "__main__":
    main()