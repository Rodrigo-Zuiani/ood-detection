import numpy as np
import torch
from numpy.linalg import pinv
from scipy.special import softmax

from models.resnet import BasicBlock, ResNet18
from dataset.cifar100 import get_cifar100_loaders
from dataset.svhn import get_svhn_ood_loader
import OODmethods as ood_methods


# Configuration
CHECKPOINT = "checkpoints/resnet18_cifar100_epoch_450.pt"
BATCH_SIZE = 128
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100


def extract_features(model, loader, device, num_samples=None):
    """Extract features and labels from a dataloader."""
    all_features = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            if num_samples is not None and len(all_features) * len(data) >= num_samples:
                break
                
            data = data.to(device)
            # Get features from avgpool layer (before classification head)
            features = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(
                torch.nn.functional.relu(model.bn1(model.conv1(data))))))))
            features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def main():
    print("Loading model...")
    model = ResNet18(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(DEVICE)
    
    print("Extracting features from ID dataset (CIFAR-100)...")
    cifar100_trainloader, cifar100_testloader, _ = get_cifar100_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    feature_id_train, train_labels = extract_features(model, cifar100_trainloader, DEVICE)
    feature_id_val, _ = extract_features(model, cifar100_testloader, DEVICE)
    
    print("Extracting features from OOD dataset (SVHN)...")
    ood_loader = get_svhn_ood_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    feature_ood, _ = extract_features(model, ood_loader, DEVICE)
    
    print(f'Feature shapes: train={feature_id_train.shape}, val={feature_id_val.shape}, ood={feature_ood.shape}')
    
    # Get weights and bias from the linear classification layer
    w = model.linear.weight.data.cpu().numpy()
    b = model.linear.bias.data.cpu().numpy()
    
    print('Computing logits and softmax...')
    logit_id_train = feature_id_train @ w.T + b
    logit_id_val = feature_id_val @ w.T + b
    logit_ood = feature_ood @ w.T + b
    
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    softmax_ood = softmax(logit_ood, axis=-1)
    
    u = -np.matmul(pinv(w), b)
    
    # Run OOD detection methods
    print('\n' + '='*80)
    print('Running OOD Detection Methods')
    print('='*80)
    
    print('\n[1/5] Max Softmax Probability (MSP)')
    ood_methods.maxSoftmaxProb(softmax_id_val, softmax_ood)
    
    print('\n[2/5] Max Logit Score')
    ood_methods.maxLogitScore(logit_id_val, logit_ood)
    
    print('\n[3/5] Energy Score')
    ood_methods.energyScore(logit_id_val, logit_ood)
    
    print('\n[4/5] ViM (Vim Scoring)')
    ood_methods.vim(feature_id_train, feature_id_val, feature_ood, logit_id_train,
                    logit_id_val, logit_ood, u)
    
    print('\n[5/5] Mahalanobis Distance')
    ood_methods.mahalanobis(feature_id_train, train_labels,
                            feature_id_val, feature_ood, NUM_CLASSES)
    
    print('\n' + '='*80)
    print('OOD Detection Complete')
    print('='*80)


if __name__ == '__main__':
    main()