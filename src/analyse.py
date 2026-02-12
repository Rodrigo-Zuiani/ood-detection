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

print(cp_file_names)
distance_means_dict = {}
std_means_dict = {}
wt_class_var = {}

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
    labels = torch.cat(labels_list)      # [N] ([50000]


    # Class Means Distances (NC2) --> Simplex ETF Structure
    num_classes = labels.max().item() + 1
    d = features.size(1)

    class_means = torch.zeros(num_classes, d) # [100, 512]  

    for c in range(num_classes):
        idx = labels == c               # boolean mask : True where labels is class C, otherwise False
        if idx.sum() == 0:
            continue
        class_means[c] = features[idx].mean(dim=0)

    dist_matrix = torch.cdist(class_means, class_means, p=2) # Class mean distance for 1 epoch [C, C]
    print(f"dist_matrix: {dist_matrix}")
    pairwise_distances = dist_matrix[torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()] # Upper triangle
    std_dist  = pairwise_distances.std() # Since everyone tends to the same value: std-->0
    print(f"pairwise: {pairwise_distances}")
    print(f"std dist: {std_dist}")
    # Within class variance (NC1) --> Variability Collapse Sum(W) = 0
    within_class_var = 0.0

    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        z_c = features[idx]                 # [N_c, d]
        mu_c = class_means[c]               # [d]
        within_class_var += ((z_c - mu_c)**2).sum() 

    within_class_var /= num_classes
    print(within_class_var)
    wt_class_var[epoch] = within_class_var
    distance_means_dict[epoch] = pairwise_distances
    std_means_dict[epoch] = std_dist

ordered_std = {k:v for k,v in sorted(std_means_dict.items(), key=lambda item:item[0])}
ordered_pairwise = {k:v for k,v in sorted(distance_means_dict.items(), key=lambda item:item[0])}
ordered_var = {k:v for k,v in sorted(wt_class_var.items(), key=lambda item:item[0])}

stds = list(ordered_std.values())
wt_var = list(ordered_var.values())
pairs = list(ordered_pairwise.values())
epochs.sort()
import matplotlib.pyplot as plt

plt.figure()
plt.plot(epochs, wt_var, label="Variance distance")
plt.plot(epochs, stds, label="Std distance")
plt.xlabel("Epoch")
plt.ylabel("Distance")
plt.title("Class mean distances over training")
plt.legend()
plt.show()

plot_path = "src/plots/evolution_dist.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()


data = [d.cpu().numpy() for d in pairs]

plt.figure(figsize=(10,4))
plt.boxplot(data, positions=epochs, showfliers=False)
plt.xlabel("Epoch")
plt.ylabel("Class mean distance")
plt.title("Distribution of class mean distances")
plt.show()


plot_path = "src/plots/simplex.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()