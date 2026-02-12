import torch
import matplotlib.pyplot as plt
from .models.resnet import BasicBlock, ResNet18
from .dataset.cifar100 import get_cifar100_loaders
import random
import numpy as np
import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=100).to(device)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

g = torch.Generator()
g.manual_seed(seed)

def worker_init_fn(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


trainloader, testloader, classes = get_cifar100_loaders(
    batch_size=128,
    num_workers=2,
    generator=g,
    worker_init_fn=worker_init_fn)


epochs = []
checkpoint_folder = "src/checkpoints"
cp_file_names = os.listdir(checkpoint_folder)

print(f"Found {len(cp_file_names)} checkpoint files")
print("="*80)

distance_means_dict = {}
std_means_dict = {}
wt_class_var = {}
mean_distances_dict = {}
min_distances_dict = {}
max_distances_dict = {}

for cp_file in cp_file_names:
    checkpoint_file = os.path.join(checkpoint_folder, cp_file)
    checkpoint = torch.load(checkpoint_file, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    epoch = checkpoint["epoch"]
    epochs.append(epoch)
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for x, y in trainloader:
            x = x.to(device)
            z = model.forward_features(x)  # penultimate features before FC
            features_list.append(z.cpu())
            labels_list.append(y)

    features = torch.cat(features_list)  # [N, d] [50000, 512] for cifar-100
    labels = torch.cat(labels_list)      # [N] ([50000])

    print(f"\nEpoch {epoch}:")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Class Means Distances (NC2) --> Simplex ETF Structure
    num_classes = labels.max().item() + 1
    d = features.size(1)

    class_means = torch.zeros(num_classes, d) # [100, 512]  

    for c in range(num_classes):
        idx = labels == c               # boolean mask : True where labels is class C, otherwise False
        if idx.sum() == 0:
            continue
        class_means[c] = features[idx].mean(dim=0)

    # NC2: Analyze class mean distances
    dist_matrix = torch.cdist(class_means, class_means, p=2) # Class mean distance for 1 epoch [C, C]
    pairwise_distances = dist_matrix[torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()] # Upper triangle
    
    mean_dist = pairwise_distances.mean()
    std_dist = pairwise_distances.std()
    min_dist = pairwise_distances.min()
    max_dist = pairwise_distances.max()
    cv_dist = std_dist / mean_dist  # Coefficient of variation
    
    print(f"\n  NC2 Metrics (Class Mean Distances):")
    print(f"    Mean pairwise distance: {mean_dist:.4f}")
    print(f"    Std of distances: {std_dist:.4f}")
    print(f"    Min distance: {min_dist:.4f}")
    print(f"    Max distance: {max_dist:.4f}")
    print(f"    Coefficient of variation: {cv_dist:.4f}")
    print(f"    Range (max-min): {(max_dist - min_dist):.4f}")
    
    # Within class variance (NC1) --> Variability Collapse
    # within_class_var = 0.0
    Sw = torch.zeros(d, d) 
    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        z_c = features[idx]                 # [N_c, d] all samples with d features
        mu_c = class_means[c]               # [d] the mean for every d feature
        
        # Compute sum of squared deviations for this class
        centered = z_c - mu_c
        Sw += (centered.T @ centered)
    
    # Normalize by total number of samples AND feature dimension
    # within_class_var /= (features.size(0) * d)
    Sw /= features.size(0) 
    within_class_var = torch.trace(Sw) / (features.size(0) * d)
    print(f"\n  NC1 Metrics (Within-Class Variance):")
    print(f"    Within-class variance (per dim per sample): {within_class_var:.6f}")
    
    # Additional diagnostic: compute global mean
    global_mean = features.mean(dim=0)
    total_var = ((features - global_mean)**2).sum() / (features.size(0) * d)
    between_class_var = total_var - within_class_var
    
    print(f"    Total variance: {total_var:.6f}")
    print(f"    Between-class variance: {between_class_var:.6f}")
    print(f"    Within/Total ratio: {(within_class_var / total_var):.6f}")
    
    print("="*80)
    
    wt_class_var[epoch] = within_class_var.item()
    distance_means_dict[epoch] = pairwise_distances
    std_means_dict[epoch] = std_dist.item()
    mean_distances_dict[epoch] = mean_dist.item()
    min_distances_dict[epoch] = min_dist.item()
    max_distances_dict[epoch] = max_dist.item()

# Sort everything by epoch
ordered_std = {k:v for k,v in sorted(std_means_dict.items(), key=lambda item:item[0])}
ordered_pairwise = {k:v for k,v in sorted(distance_means_dict.items(), key=lambda item:item[0])}
ordered_var = {k:v for k,v in sorted(wt_class_var.items(), key=lambda item:item[0])}
ordered_mean_dist = {k:v for k,v in sorted(mean_distances_dict.items(), key=lambda item:item[0])}
ordered_min_dist = {k:v for k,v in sorted(min_distances_dict.items(), key=lambda item:item[0])}
ordered_max_dist = {k:v for k,v in sorted(max_distances_dict.items(), key=lambda item:item[0])}

stds = list(ordered_std.values())
wt_var = list(ordered_var.values())
pairs = list(ordered_pairwise.values())
mean_dists = list(ordered_mean_dist.values())
min_dists = list(ordered_min_dist.values())
max_dists = list(ordered_max_dist.values())
epochs.sort()

# Plot 1: NC1 and NC2 metrics over time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# NC1: Within-class variance
axes[0, 0].plot(epochs, wt_var, 'b-', linewidth=2, marker='o')
axes[0, 0].set_xlabel("Epoch", fontsize=12)
axes[0, 0].set_ylabel("Within-class variance", fontsize=12)
axes[0, 0].set_title("NC1: Within-Class Variability Collapse", fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')  # Log scale often better for variance

# NC2: Std of distances (should go to 0)
axes[0, 1].plot(epochs, stds, 'r-', linewidth=2, marker='o')
axes[0, 1].set_xlabel("Epoch", fontsize=12)
axes[0, 1].set_ylabel("Std of pairwise distances", fontsize=12)
axes[0, 1].set_title("NC2: Distance Uniformity (should → 0)", fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# NC2: Mean distance (should stabilize)
axes[1, 0].plot(epochs, mean_dists, 'g-', linewidth=2, marker='o', label='Mean')
axes[1, 0].fill_between(epochs, min_dists, max_dists, alpha=0.3, label='Min-Max range')
axes[1, 0].set_xlabel("Epoch", fontsize=12)
axes[1, 0].set_ylabel("Distance", fontsize=12)
axes[1, 0].set_title("NC2: Mean Pairwise Distance", fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Coefficient of variation
cv = [s/m for s, m in zip(stds, mean_dists)]
axes[1, 1].plot(epochs, cv, 'm-', linewidth=2, marker='o')
axes[1, 1].set_xlabel("Epoch", fontsize=12)
axes[1, 1].set_ylabel("Coefficient of Variation", fontsize=12)
axes[1, 1].set_title("NC2: CV = Std/Mean (should → 0)", fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = "src/plots/nc_metrics.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved NC metrics plot to {plot_path}")

# Plot 2: Distribution of pairwise distances (boxplot)
data = [d.cpu().numpy() for d in pairs]

plt.figure(figsize=(12, 6))
plt.boxplot(data, positions=epochs, showfliers=False, widths=5)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Pairwise class mean distance", fontsize=12)
plt.title("Distribution of Class Mean Distances Over Training", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

plot_path = "src/plots/simplex_boxplot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved boxplot to {plot_path}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(f"Epochs analyzed: {min(epochs)} to {max(epochs)}")
print(f"\nNC1 (Within-class variance):")
print(f"  Initial (epoch {min(epochs)}): {wt_var[0]:.6f}")
print(f"  Final (epoch {max(epochs)}): {wt_var[-1]:.6f}")
print(f"  Reduction: {(1 - wt_var[-1]/wt_var[0])*100:.2f}%")
print(f"\nNC2 (Distance std):")
print(f"  Initial (epoch {min(epochs)}): {stds[0]:.4f}")
print(f"  Final (epoch {max(epochs)}): {stds[-1]:.4f}")
print(f"  Reduction: {(1 - stds[-1]/stds[0])*100:.2f}%")
print(f"\nNC2 (Coefficient of variation):")
print(f"  Initial (epoch {min(epochs)}): {cv[0]:.4f}")
print(f"  Final (epoch {max(epochs)}): {cv[-1]:.4f}")
print("="*80)