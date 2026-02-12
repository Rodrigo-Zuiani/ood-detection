import torch
from . import OODmethods as ood_methods
from models.resnet import BasicBlock, ResNet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=100).to(device)

checkpoint = "checkpoints/resnet18_cifar100.pth"
state_dict = torch.load(checkpoint)
model.load_state_dict(state_dict['net'], strict=False)
