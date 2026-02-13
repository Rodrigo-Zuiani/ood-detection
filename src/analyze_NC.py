"""
Neural Collapse Analysis - Main Script
Analyzes NC1, NC2, and NC3 metrics across training checkpoints
"""

import os
import torch
from pathlib import Path
from .utils.models_utils import setup_reproducibility, load_model_checkpoint, extract_features
from .utils.NC_utils import compute_nc1_metrics, compute_nc2_metrics, compute_nc3_metrics
from .utils.plot_utils import plot_nc1_nc2_metrics, plot_nc3_metrics


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
    
    # Process all checkpoints
    for cp_file in checkpoint_files:
        checkpoint_path = os.path.join(config['checkpoint_folder'], cp_file)
        epoch = load_model_checkpoint(model, checkpoint_path, device)
        metrics['epochs'].append(epoch)
        
        print(f"\nEpoch {epoch}:")
        
        # Extract features from training data
        features, labels = extract_features(model, trainloader, device)
        # print(f"  Features shape: {features.shape}")
        # print(f"  Labels shape: {labels.shape}")
        
        # Compute class means
        num_classes = labels.max().item() + 1
        class_means = compute_class_means(features, labels, num_classes)
        
    ## NC1: Within-class variance
        nc1_results = compute_nc1_metrics(features, labels, class_means)
        # print("\n  NC1 Metrics:")
        # print(f"    Within-class variance: {nc1_results['within_class_var']:.6f}")
        # print(f"    Within/Total ratio: {nc1_results['within_total_ratio']:.6f}")
        
        metrics['nc1']['within_class_var'][epoch] = nc1_results['within_class_var']
        metrics['nc1']['within_total_ratio'][epoch] = nc1_results['within_total_ratio']
        
    ## NC2: Class mean distances
        nc2_results = compute_nc2_metrics(class_means)
        # print("\n  NC2 Metrics:")
        # print(f"    Mean distance: {nc2_results['mean_dist']:.4f}")
        # print(f"    Std distance: {nc2_results['std_dist']:.4f}")
        # print(f"    Coefficient of variation: {nc2_results['cv']:.4f}")
        
        for key in ['mean_dist', 'std_dist', 'min_dist', 'max_dist', 'cv']:
            metrics['nc2'][key][epoch] = nc2_results[key]
        
    ## NC3: Self-Duality
        nc3_results = compute_nc3_metrics(model, class_means)
        # print("\n  NC3 Metrics:")
        # print(f"    Mean cosine similarity: {nc3_results['mean_cos']:.4f}")
        # print(f"    Std cosine similarity: {nc3_results['std_cos']:.4f}")
        # print(f"    Min cosine similarity: {nc3_results['min_cos']:.4f}")
        
        for key in ['mean_cos', 'std_cos', 'min_cos']:
            metrics['nc3'][key][epoch] = nc3_results[key]
        
        print("=" * 80)
    
    # Plots
    metrics['epochs'].sort()
    
    plot_nc1_nc2_metrics(
        metrics, 
        save_path=os.path.join(config['plots_folder'], 'nc_metrics.png')
    )
    
    plot_nc3_metrics(
        metrics,
        save_path=os.path.join(config['plots_folder'], 'nc3_self_duality.png')
    )
    
    print("\nAnalysis complete. Plots saved.")


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
