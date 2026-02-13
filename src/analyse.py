import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .models.resnet import BasicBlock, ResNet18
from .dataset.cifar100 import get_cifar100_loaders
import random
import numpy as np
import os 

# ============================================================
# Setup
# ============================================================

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

checkpoint_folder = "src/checkpoints"
cp_file_names = os.listdir(checkpoint_folder)

print(f"Found {len(cp_file_names)} checkpoint files")
print("="*80)

# ============================================================
# Storage dictionaries
# ============================================================

distance_means_dict = {}
std_means_dict = {}
wt_class_var = {}
mean_distances_dict = {}
min_distances_dict = {}
max_distances_dict = {}

# NC3
nc3_mean_dict = {}
nc3_std_dict = {}
nc3_min_dict = {}

epochs = []

# ============================================================
# Loop over checkpoints
# ============================================================

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
            z = model.forward_features(x)
            features_list.append(z.cpu())
            labels_list.append(y)

    features = torch.cat(features_list)
    labels = torch.cat(labels_list)

    print(f"\nEpoch {epoch}:")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")

    # ========================================================
    # Compute class means
    # ========================================================

    num_classes = labels.max().item() + 1
    d = features.size(1)

    class_means = torch.zeros(num_classes, d)

    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        class_means[c] = features[idx].mean(dim=0)

    global_class_mean = class_means.mean(dim=0, keepdim=True)
    class_means_centered = class_means - global_class_mean

    # ========================================================
    # NC2: Class mean distances
    # ========================================================

    dist_matrix = torch.cdist(class_means_centered, class_means_centered, p=2)
    pairwise_distances = dist_matrix[torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()]

    mean_dist = pairwise_distances.mean()
    std_dist = pairwise_distances.std()
    min_dist = pairwise_distances.min()
    max_dist = pairwise_distances.max()
    cv_dist = std_dist / mean_dist

    print("\n  NC2 Metrics:")
    print(f"    Mean distance: {mean_dist:.4f}")
    print(f"    Std distance: {std_dist:.4f}")
    print(f"    Coefficient of variation: {cv_dist:.4f}")

    # ========================================================
    # NC1: Within-class variance
    # ========================================================

    Sw = torch.zeros(d, d)

    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        z_c = features[idx]
        mu_c = class_means[c]
        centered = z_c - mu_c
        Sw += centered.T @ centered

    within_class_var = torch.trace(Sw) / (features.size(0) * d)

    global_mean = features.mean(dim=0)
    total_var = ((features - global_mean)**2).sum() / (features.size(0) * d)
    between_class_var = total_var - within_class_var

    print("\n  NC1 Metrics:")
    print(f"    Within-class variance: {within_class_var:.6f}")
    print(f"    Within/Total ratio: {(within_class_var / total_var):.6f}")

    # ========================================================
    # NC3: Self-Duality (Weight || Mean)
    # ========================================================

    W = model.linear.weight.detach().cpu()
    mu = class_means_centered

    W_norm = F.normalize(W, dim=1)
    mu_norm = F.normalize(mu, dim=1)

    cos_sim = torch.sum(W_norm * mu_norm, dim=1)

    mean_cos = cos_sim.mean()
    std_cos = cos_sim.std()
    min_cos = cos_sim.min()

    print("\n  NC3 Metrics:")
    print(f"    Mean cosine similarity: {mean_cos:.4f}")
    print(f"    Std cosine similarity: {std_cos:.4f}")
    print(f"    Min cosine similarity: {min_cos:.4f}")

    print("="*80)

    # Store
    wt_class_var[epoch] = within_class_var.item()
    std_means_dict[epoch] = std_dist.item()
    mean_distances_dict[epoch] = mean_dist.item()
    min_distances_dict[epoch] = min_dist.item()
    max_distances_dict[epoch] = max_dist.item()

    nc3_mean_dict[epoch] = mean_cos.item()
    nc3_std_dict[epoch] = std_cos.item()
    nc3_min_dict[epoch] = min_cos.item()

# ============================================================
# Sort by epoch
# ============================================================

epochs.sort()

wt_var = [wt_class_var[e] for e in epochs]
stds = [std_means_dict[e] for e in epochs]
mean_dists = [mean_distances_dict[e] for e in epochs]
min_dists = [min_distances_dict[e] for e in epochs]
max_dists = [max_distances_dict[e] for e in epochs]

nc3_means = [nc3_mean_dict[e] for e in epochs]
nc3_stds = [nc3_std_dict[e] for e in epochs]

# ============================================================
# Plot NC1 & NC2
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(epochs, wt_var, 'b-o')
axes[0, 0].set_title("NC1: Within-Class Variance")
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True)

axes[0, 1].plot(epochs, stds, 'r-o')
axes[0, 1].set_title("NC2: Std of Mean Distances")
axes[0, 1].grid(True)

axes[1, 0].plot(epochs, mean_dists, 'g-o')
axes[1, 0].fill_between(epochs, min_dists, max_dists, alpha=0.3)
axes[1, 0].set_title("NC2: Mean Pairwise Distance")
axes[1, 0].grid(True)

cv = [s/m for s, m in zip(stds, mean_dists)]
axes[1, 1].plot(epochs, cv, 'm-o')
axes[1, 1].set_title("NC2: Coefficient of Variation")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("src/plots/nc_metrics.png", dpi=300)
plt.close()

# ============================================================
# Plot NC3
# ============================================================

plt.figure(figsize=(8, 6))
plt.plot(epochs, nc3_means, 'k-o', label="Mean Cosine")
plt.fill_between(
    epochs,
    [m - s for m, s in zip(nc3_means, nc3_stds)],
    [m + s for m, s in zip(nc3_means, nc3_stds)],
    alpha=0.3
)
plt.ylim(0, 1.05)
plt.title("NC3: Self-Duality (Weight || Mean)")
plt.xlabel("Epoch")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("src/plots/nc3_self_duality.png", dpi=300)
plt.close()

print("\nAnalysis complete. Plots saved.")
