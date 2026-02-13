"""
Neural Collapse Analysis - Main Script
Analyzes NC1, NC2, and NC3 metrics across training checkpoints

Two modes:
    1) ID only
    2) ID + OOD
"""

import os
import torch
import copy
from pathlib import Path

from .utils.models_utils import (
    setup_reproducibility,
    load_model_checkpoint,
    extract_features,
    extract_layer_features
)
from .utils.NC_utils import (
    compute_nc1_metrics,
    compute_nc2_metrics,
    compute_nc3_metrics
)
from .utils.plot_utils import (
    plot_nc1_nc2_metrics,
    plot_nc3_metrics,
    plot_nc_comparison,
    plot_nc2_nc3_multilayer
)


# ============================================================
# Common Setup
# ============================================================

def setup_environment():
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

    return config, device


def initialize_model(device):
    from .models.resnet import BasicBlock, ResNet18

    model = ResNet18(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        num_classes=100
    ).to(device)

    return model


def initialize_id_loader(config):
    from .dataset.cifar100 import get_cifar100_loaders

    g = torch.Generator()
    g.manual_seed(config['seed'])

    trainloader, testloader, classes = get_cifar100_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        generator=g,
        worker_init_fn=lambda worker_id:
        setup_reproducibility(config['seed'] + worker_id)
    )

    return trainloader


def initialize_ood_loader(config):
    from .dataset.svhn import get_svhn_ood_loader

    testLoader_OOD, classes_OOD = get_svhn_ood_loader(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    return testLoader_OOD


def initialize_metrics():
    return {
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


# ============================================================
# Core Processing Logic (Reusable)
# ============================================================

def process_checkpoints(model, loader, device, config, return_layers = None):

    checkpoint_files = sorted(os.listdir(config['checkpoint_folder']))
    print(f"Found {len(checkpoint_files)} checkpoint files")
    print("=" * 80)

    metrics = initialize_metrics()

    for cp_file in checkpoint_files:
        checkpoint_path = os.path.join(config['checkpoint_folder'], cp_file)
        epoch = load_model_checkpoint(model, checkpoint_path, device)

        metrics['epochs'].append(epoch)

        print(f"\nEpoch {epoch}")

        if return_layers:
            features, labels = extract_layer_features(model, loader, device, return_layers)    
        else:
            features, labels = extract_features(model, loader, device)
        print(f"  Features shape: {features.shape}")

        num_classes = labels.max().item() + 1
        class_means = compute_class_means(features, labels, num_classes)

        # NC1
        nc1 = compute_nc1_metrics(features, labels, class_means)
        metrics['nc1']['within_class_var'][epoch] = nc1['within_class_var']
        metrics['nc1']['within_total_ratio'][epoch] = nc1['within_total_ratio']

        # NC2
        nc2 = compute_nc2_metrics(class_means)
        for key in ['mean_dist', 'std_dist', 'min_dist', 'max_dist', 'cv']:
            metrics['nc2'][key][epoch] = nc2[key]

        # NC3
        if not return_layers:
            nc3 = compute_nc3_metrics(model, class_means)
            for key in ['mean_cos', 'std_cos', 'min_cos']:
                metrics['nc3'][key][epoch] = nc3[key]

        print("=" * 80)

    metrics['epochs'].sort()
    return metrics


# ============================================================
# MODE 1: ID ONLY
# ============================================================

def run_id_only():
    print("\nRunning ID-only Neural Collapse Analysis\n")

    config, device = setup_environment()
    model = initialize_model(device)
    trainloader = initialize_id_loader(config)

    metrics = process_checkpoints(model, trainloader, device, config)

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

    print("\nID-only analysis complete.\n")


# ============================================================
# MODE 2: ID + OOD
# ============================================================

def run_id_ood():
    print("\nRunning ID + OOD Neural Collapse Analysis\n")

    config, device = setup_environment()
    model = initialize_model(device)

    trainloader = initialize_id_loader(config)
    ood_loader = initialize_ood_loader(config)

    # ID
    metrics_ID = process_checkpoints(model, trainloader, device, config)

    # OOD
    metrics_OOD = process_checkpoints(model, ood_loader, device, config)

    plot_nc1_nc2_metrics(
        metrics_ID,
        save_path=os.path.join(config['plots_folder'], 'nc_metrics_ID.png'),
        title_prefix="ID (CIFAR-100)"
    )

    plot_nc1_nc2_metrics(
        metrics_OOD,
        save_path=os.path.join(config['plots_folder'], 'nc_metrics_OOD.png'),
        title_prefix="OOD (SVHN)"
    )

    plot_nc3_metrics(
        metrics_ID,
        save_path=os.path.join(config['plots_folder'], 'nc3_self_duality_ID.png'),
        title_prefix="ID (CIFAR-100)"
    )

    plot_nc_comparison(
        metrics_ID,
        metrics_OOD,
        save_path=os.path.join(config['plots_folder'], 'nc_comparison_ID_vs_OOD.png')
    )

    print("\nID + OOD analysis complete.\n")


# ============================================================
# MODE 3: ID Multilayer
# ============================================================

def run_id_multilayer():
    print("\nRunning ID multilayer Neural Collapse Analysis\n")

    config, device = setup_environment()
    model = initialize_model(device)
    trainloader = initialize_id_loader(config)
    return_layers = [["conv1"], ["layer1"], ["layer2"], ["layer3"], ["layer4"] ]
    # ID
    metrics_ID_all = []
    for layer_list in return_layers:
        metrics_ID = process_checkpoints(model, trainloader, device, config, return_layers=layer_list)
        metrics_ID_all.append(metrics_ID)


    plot_nc1_nc2_metrics(
        metrics_ID,
        save_path=os.path.join(config['plots_folder'], 'nc_metrics_ID.png'),
        title_prefix="ID (CIFAR-100)"
    )


    plot_nc3_metrics(
        metrics_ID,
        save_path=os.path.join(config['plots_folder'], 'nc3_self_duality_ID.png'),
        title_prefix="ID (CIFAR-100)"
    )

    layer_names = ["conv1", "layer1", "layer2", "layer3", "layer4"]
    plot_nc2_nc3_multilayer(
        metrics_all=metrics_ID_all,
        layer_names =layer_names,
        save_path=os.path.join(config['plots_folder'], 'nc2_nc3_multilayer.png'),
        title_prefix="ID Cross Layer (CIFAR-100)"

    )
 
    print("\nID multilayer analysis complete.\n")



# ============================================================
# Class Means
# ============================================================

def compute_class_means(features, labels, num_classes):
    d = features.size(1)
    class_means = torch.zeros(num_classes, d)

    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        class_means[c] = features[idx].mean(dim=0)

    global_mean = class_means.mean(dim=0, keepdim=True)
    return class_means - global_mean


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    # Choose one:
    # run_id_only()
    # run_id_ood()
    run_id_multilayer()
