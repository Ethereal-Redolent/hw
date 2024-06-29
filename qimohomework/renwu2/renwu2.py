import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.models import resnet18
from vit_pytorch import ViT
import csv
import os

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data augmentation and loading
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
])

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# CutMix data augmentation
def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]

    targets = (targets, shuffled_targets, lam)
    return data, targets

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Define CNN model
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(SimpleResNet, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

cnn_model = SimpleResNet(num_classes=100).to(device)

# Define Transformer model
transformer_model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=100,
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer_cnn = optim.SGD(cnn_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler_cnn = optim.lr_scheduler.StepLR(optimizer_cnn, step_size=30, gamma=0.1)

optimizer_transformer = optim.SGD(transformer_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler_transformer = optim.lr_scheduler.StepLR(optimizer_transformer, step_size=30, gamma=0.1)

# Function to save model weights to CSV
def save_weights_to_csv(model, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for name, param in model.named_parameters():
            writer.writerow([name, param.data.cpu().numpy().tolist()])

# Training and validation functions
def train(model, trainloader, optimizer, criterion, epoch, writer, model_name):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = cutmix(inputs, targets)
        inputs, targets_a, targets_b, lam = inputs.to(device), targets[0].to(device), targets[1].to(device), targets[2]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets_a.size(0)
        correct += (lam * predicted.eq(targets_a).sum().item() + (1 - lam) * predicted.eq(targets_b).sum().item())

    avg_loss = train_loss / len(trainloader)
    accuracy = 100. * correct / total
    writer.add_scalar(f'{model_name}/Train Loss', avg_loss, epoch)
    writer.add_scalar(f'{model_name}/Train Accuracy', accuracy, epoch)
    print(f'{model_name} Epoch {epoch}, Train Loss: {avg_loss}, Train Accuracy: {accuracy}%')

def validate(model, testloader, criterion, epoch, writer, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / len(testloader)
    accuracy = 100. * correct / total
    writer.add_scalar(f'{model_name}/Validation Loss', avg_loss, epoch)
    writer.add_scalar(f'{model_name}/Validation Accuracy', accuracy, epoch)
    print(f'{model_name} Epoch {epoch}, Validation Loss: {avg_loss}, Validation Accuracy: {accuracy}%')

# Training loop
num_epochs = 30
writer = SummaryWriter('runs/cifar100_experiment')

if __name__ == '__main__':
    for epoch in range(num_epochs):
        train(cnn_model, trainloader, optimizer_cnn, criterion, epoch, writer, 'CNN')
        validate(cnn_model, testloader, criterion, epoch, writer, 'CNN')
        scheduler_cnn.step()
        # Save weights
        save_weights_to_csv(cnn_model, f'runs/cnn_weights_epoch_{epoch}.csv')

    for epoch in range(num_epochs):
        train(transformer_model, trainloader, optimizer_transformer, criterion, epoch, writer, 'Transformer')
        validate(transformer_model, testloader, criterion, epoch, writer, 'Transformer')
        scheduler_transformer.step()
        # Save weights
        save_weights_to_csv(transformer_model, f'runs/transformer_weights_epoch_{epoch}.csv')

    writer.close()
