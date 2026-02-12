import torch
import matplotlib.pyplot as plt
from .models.resnet import BasicBlock, ResNet18
from .dataset.cifar100 import get_cifar100_loaders
import random
import numpy as np

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

features = []
labels = []

model.load_state_dict(torch.load("checkpoints/resnet18_cifar100_epoch_020.pt", weights_only=True))
model.eval()

with torch.no_grad():
    for x, y in trainloader:
        x = x.to(device)
        z = model.forward_features(x)  # penultimate features
        features.append(z.cpu())
        labels.append(y)

features = torch.cat(features)  # [N, d]
print(features.size())
labels = torch.cat(labels)      # [N]
print(features.size())
