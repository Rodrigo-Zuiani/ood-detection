import torch
import torchvision
import torchvision.transforms as transforms

def get_svhn_loaders(batch_size=128, num_workers=2, generator=None, worker_init_fn=None):
    """
    Get SVHN dataset loaders.
    
    SVHN (Street View House Numbers) is a real-world image dataset for digit recognition.
    - 10 classes (digits 0-9)
    - 32x32 RGB images
    - Similar to CIFAR-100 in image size, making it useful for OOD detection
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        generator: Random generator for reproducibility
        worker_init_fn: Worker initialization function for reproducibility
        
    Returns:
        trainloader: DataLoader for training set
        testloader: DataLoader for test set
        classes: List of class names (digits 0-9)
    """
    # SVHN statistics (computed from training set)
    # Mean: (0.4377, 0.4438, 0.4728)
    # Std: (0.1980, 0.2010, 0.1970)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                             (0.1980, 0.2010, 0.1970))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                             (0.1980, 0.2010, 0.1970))
    ])

    # SVHN uses 'train' and 'test' splits
    trainset = torchvision.datasets.SVHN(
        root='./data',
        split='train',
        download=True,
        transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn
    )

    testset = torchvision.datasets.SVHN(
        root='./data',
        split='test',
        download=True,
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # SVHN classes are digits 0-9
    classes = [str(i) for i in range(10)]

    return trainloader, testloader, classes


def get_svhn_ood_loader(batch_size=128, num_workers=2, worker_init_fn = None):
    """
    Get SVHN test loader for OOD detection experiments.
    
    When using SVHN as OOD data for a CIFAR-100 trained model, you should
    normalize SVHN with CIFAR-100 statistics for fair comparison.
    
    Args:
        batch_size: Batch size for data loader
        num_workers: Number of worker processes
        use_cifar_normalization: If True, use CIFAR-100 normalization (recommended for OOD)
        
    Returns:
        ood_loader: DataLoader for SVHN test set
        classes: List of class names
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                                (0.1980, 0.2010, 0.1970))
    ])
    
    testset = torchvision.datasets.SVHN(
        root='./data',
        split='test',
        download=True,
        transform=transform
    )
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    ood_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )
    
    classes = [str(i) for i in range(10)]
    
    return ood_loader, classes