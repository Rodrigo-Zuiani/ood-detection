"""
Neural Collapse Metrics Computation
Functions to compute NC1, NC2, and NC3 metrics
"""

import torch
import torch.nn.functional as F


def compute_nc1_metrics(features, labels, class_means):
    """
    Compute NC1 (Within-class variability collapse) metrics.
    
    NC1 measures how tightly clustered samples are around their class means.
    Lower within-class variance indicates stronger neural collapse.
    
    Args:
        features: Tensor of shape (N, d) with feature vectors
        labels: Tensor of shape (N,) with class labels
        class_means: Tensor of shape (num_classes, d) with centered class means
        
    Returns:
        dict with:
            - within_class_var: Average within-class variance
            - total_var: Total variance
            - between_class_var: Between-class variance
            - within_total_ratio: Ratio of within-class to total variance
    """
    num_classes = class_means.size(0)
    d = features.size(1)
    N = features.size(0)
    
    # Compute within-class scatter matrix
    Sw = torch.zeros(d, d)
    
    # Get uncentered class means for distance computation
    class_means_uncentered = torch.zeros(num_classes, d)
    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        class_means_uncentered[c] = features[idx].mean(dim=0)
    
    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        z_c = features[idx]
        mu_c = class_means_uncentered[c]
        centered = z_c - mu_c
        Sw += centered.T @ centered
    
    # Normalized within-class variance
    within_class_var = torch.trace(Sw) / (N * d)
    
    # Total variance
    global_mean = features.mean(dim=0)
    total_var = ((features - global_mean) ** 2).sum() / (N * d)
    
    # Between-class variance
    between_class_var = total_var - within_class_var
    
    return {
        'within_class_var': within_class_var.item(),
        'total_var': total_var.item(),
        'between_class_var': between_class_var.item(),
        'within_total_ratio': (within_class_var / total_var).item()
    }


def compute_nc2_metrics(class_means_centered):
    """
    Compute NC2 (Maximal equidistance of class means) metrics.
    
    NC2 measures how uniformly distributed class means are. In perfect collapse,
    class means form an equiangular tight frame (ETF) with equal pairwise distances.
    Low coefficient of variation indicates more uniform spacing.
    
    Args:
        class_means_centered: Tensor of shape (num_classes, d) with centered class means
        
    Returns:
        dict with:
            - mean_dist: Mean pairwise distance between class means
            - std_dist: Standard deviation of pairwise distances
            - min_dist: Minimum pairwise distance
            - max_dist: Maximum pairwise distance
            - cv: Coefficient of variation (std/mean)
    """
    # Compute pairwise distances between class means
    dist_matrix = torch.cdist(class_means_centered, class_means_centered, p=2)
    
    # Extract upper triangular (unique pairs)
    pairwise_distances = dist_matrix[
        torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
    ]
    
    mean_dist = pairwise_distances.mean()
    std_dist = pairwise_distances.std()
    min_dist = pairwise_distances.min()
    max_dist = pairwise_distances.max()
    cv_dist = std_dist / mean_dist
    
    return {
        'mean_dist': mean_dist.item(),
        'std_dist': std_dist.item(),
        'min_dist': min_dist.item(),
        'max_dist': max_dist.item(),
        'cv': cv_dist.item()
    }


def compute_nc3_metrics(model, class_means_centered):
    """
    Compute NC3 (Self-duality) metrics.
    
    NC3 measures alignment between classifier weights and class means.
    In perfect collapse, each classifier weight vector points in the same
    direction as its corresponding class mean (cosine similarity = 1).
    
    Args:
        model: Neural network model with a 'linear' layer
        class_means_centered: Tensor of shape (num_classes, d) with centered class means
        
    Returns:
        dict with:
            - mean_cos: Mean cosine similarity between weights and means
            - std_cos: Standard deviation of cosine similarities
            - min_cos: Minimum cosine similarity
            - max_cos: Maximum cosine similarity
    """
    # Get classifier weights
    W = model.linear.weight.detach().cpu()
    mu = class_means_centered
    
    # Normalize to unit vectors
    W_norm = F.normalize(W, dim=1)
    mu_norm = F.normalize(mu, dim=1)
    
    # Compute cosine similarity for each class
    cos_sim = torch.sum(W_norm * mu_norm, dim=1)
    
    mean_cos = cos_sim.mean()
    std_cos = cos_sim.std()
    min_cos = cos_sim.min()
    max_cos = cos_sim.max()
    
    return {
        'mean_cos': mean_cos.item(),
        'std_cos': std_cos.item(),
        'min_cos': min_cos.item(),
        'max_cos': max_cos.item()
    }
