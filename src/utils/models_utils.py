"""
Utility Functions
Helper functions for reproducibility, model loading, and feature extraction
"""

import torch
import random
import numpy as np


def setup_reproducibility(seed):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from checkpoint file.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        epoch: Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    epoch = checkpoint["epoch"]
    return epoch


def extract_features(model, dataloader, device):
    """
    Extract feature representations from a model for all data in the loader.
    
    Args:
        model: PyTorch model with forward_features() method
        dataloader: DataLoader containing the data
        device: Device to run inference on
        
    Returns:
        features: Tensor of shape (N, d) with feature vectors
        labels: Tensor of shape (N,) with labels
    """
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            z = model.forward_features(x)
            features_list.append(z.cpu())
            labels_list.append(y)
    
    features = torch.cat(features_list)
    labels = torch.cat(labels_list)
    
    return features, labels
