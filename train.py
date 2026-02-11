import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.resnet import BasicBlock, ResNet18
from dataset.cifar100 import get_cifar100_loaders
import os
import random
import numpy as np
import csv 
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##### Configurations #####
# Setting the previous run
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

model = ResNet18(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=100).to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.1
weight_decay = 0
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 350
train_losses, train_acc_list, test_acc_list = [], [], []

# Make sure folders exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
log_csv_path = "logs/train_log.csv"
logging.basicConfig(
    level=logging.INFO,  # INFO, DEBUG, WARNING, ERROR
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # prints to console
    ]
)
with open(log_csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "test_acc"])  # header

##### Training ######

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(trainloader.dataset)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_acc_list.append(train_acc)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100. * correct / total
    test_acc_list.append(test_acc)
    
    scheduler.step()
    ## Logger and savings ##
    logging.info(f'Epoch [{epoch+1}/{num_epochs}] '
             f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')
    with open(log_csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, train_acc, test_acc])

    if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs:
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "config": {
                "dataset": "CIFAR100",
                "model": "ResNet18",
                "lr": lr,
                "weight_decay": weight_decay,
                "seed": seed
            }
        }
        checkpoint_path = f"checkpoints/resnet18_cifar100_epoch_{epoch+1:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

model_path = "checkpoints/resnet18_cifar100.pth"
torch.save({
    "epoch": num_epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, model_path)

logging.info(f"Checkpoint saved: {model_path}")
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy')
plt.legend()

plot_path = "plots/resnet18_cifar100_training_NC.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Plot saved to {plot_path}")
