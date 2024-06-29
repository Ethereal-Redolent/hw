'''import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


# SimCLR Model
class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = h.squeeze()
        z = self.projection(h)
        return h, z

    def get_backbone(self, x):
        h = self.backbone(x)
        h = h.squeeze()
        return h


# NT-Xent loss
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self._get_correlated_mask().type(torch.bool)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _get_correlated_mask(self):
        mask = torch.ones((2 * self.batch_size, 2 * self.batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positives = torch.cat((sim_i_j, sim_j_i), dim=0)
        negatives = sim[self.mask].view(N, -1)

        labels = torch.zeros(N).to(positives.device).long()
        logits = torch.cat((positives.unsqueeze(1), negatives), dim=1)

        loss = self.criterion(logits, labels)
        return loss / N


def train_simclr(model, loader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    for (x_i, x_j), _ in loader:
        x_i, x_j = x_i.to(device), x_j.to(device)
        h_i, z_i = model(x_i)
        h_j, z_j = model(x_j)
        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/SimCLR', avg_loss, epoch)
    return avg_loss


def get_simclr_dataloader(batch_size, data_dir):
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=SimCLRTransform(augmentation))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


class SimCLRTransform:
    def __init__(self, augmentation):
        self.augmentation = augmentation

    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)


def main_simclr():
    # Hyperparameters
    batch_size = 256
    learning_rate = 3e-4
    temperature = 0.5
    epochs = 30
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/simclr_experiment_1')

    # Model and optimizer
    model = ResNetSimCLR(models.resnet18(weights=None), out_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = NTXentLoss(batch_size, temperature)

    # DataLoader
    loader = get_simclr_dataloader(batch_size, data_dir)

    # Training
    for epoch in range(epochs):
        loss = train_simclr(model, loader, optimizer, criterion, device, writer, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "simclr_resnet18.pth")
    writer.close()


def get_cifar100_dataloader(batch_size, data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model.get_backbone(inputs)
            features.append(outputs.cpu())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def train_linear_classifier(model, features, labels, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    inputs, targets = features.to(device), labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss
    accuracy = correct / total
    writer.add_scalar('Loss/Train', avg_loss, epoch)
    writer.add_scalar('Accuracy/Train', accuracy, epoch)
    return avg_loss, accuracy


def test_linear_classifier(model, features, labels, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = features.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss
    accuracy = correct / total
    writer.add_scalar('Loss/Test', avg_loss, epoch)
    writer.add_scalar('Accuracy/Test', accuracy, epoch)
    return avg_loss, accuracy


def main_linear_eval():
    # Hyperparameters
    batch_size = 256
    learning_rate = 1e-3
    epochs = 30
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/linear_eval_experiment_1')

    # Load pretrained model and replace last layer
    simclr_model = ResNetSimCLR(models.resnet18(weights=None), out_dim=128).to(device)
    simclr_model.load_state_dict(torch.load("simclr_resnet18.pth"))
    linear_model = LinearClassifier(input_dim=512, num_classes=100).to(device)

    optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    train_loader, test_loader = get_cifar100_dataloader(batch_size, data_dir)
    train_features, train_labels = extract_features(simclr_model, train_loader, device)
    test_features, test_labels = extract_features(simclr_model, test_loader, device)

    # Training
    for epoch in range(epochs):
        train_loss, train_acc = train_linear_classifier(linear_model, train_features, train_labels, optimizer,
                                                        criterion, device, writer, epoch)
        test_loss, test_acc = test_linear_classifier(linear_model, test_features, test_labels, criterion, device,
                                                     writer, epoch)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    writer.close()


def main_supervised():
    # Hyperparameters
    batch_size = 256
    learning_rate = 1e-3
    epochs = 30
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/supervised_experiment_1')

    # Load pretrained supervised model
    supervised_model = models.resnet18(weights='IMAGENET1K_V1')
    supervised_model.fc = LinearClassifier(input_dim=512, num_classes=100).to(device)

    optimizer = optim.Adam(supervised_model.fc.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    train_loader, test_loader = get_cifar100_dataloader(batch_size, data_dir)

    # Training
    for epoch in range(epochs):
        train_loss, train_acc = train_linear_classifier(supervised_model.fc, train_loader, optimizer, criterion, device,
                                                        writer, epoch)
        test_loss, test_acc = test_linear_classifier(supervised_model.fc, test_loader, criterion, device, writer, epoch)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    writer.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python self_supervised_vs_supervised.py [simclr|linear_eval|supervised]")
        sys.exit(1)

    if sys.argv[1] == "simclr":
        main_simclr()
    elif sys.argv[1] == "linear_eval":
        main_linear_eval()
    elif sys.argv[1] == "supervised":
        main_supervised()
    else:
        print("Unknown argument:", sys.argv[1])
        sys.exit(1)'''
'''import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


# SimCLR Model
class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = h.squeeze()
        z = self.projection(h)
        return h, z

    def get_backbone(self, x):
        h = self.backbone(x)
        h = h.squeeze()
        return h


# NT-Xent loss
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self._get_correlated_mask().type(torch.bool)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _get_correlated_mask(self):
        mask = torch.ones((2 * self.batch_size, 2 * self.batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positives = torch.cat((sim_i_j, sim_j_i), dim=0)
        negatives = sim[self.mask].view(N, -1)

        labels = torch.zeros(N).to(positives.device).long()
        logits = torch.cat((positives.unsqueeze(1), negatives), dim=1)

        loss = self.criterion(logits, labels)
        return loss / N


def train_simclr(model, loader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    for (x_i, x_j), _ in loader:
        x_i, x_j = x_i.to(device), x_j.to(device)
        h_i, z_i = model(x_i)
        h_j, z_j = model(x_j)
        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/SimCLR', avg_loss, epoch)
    return avg_loss


def get_simclr_dataloader(batch_size, data_dir):
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=SimCLRTransform(augmentation))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


class SimCLRTransform:
    def __init__(self, augmentation):
        self.augmentation = augmentation

    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)


def main_simclr():
    # Hyperparameters
    batch_size = 256
    learning_rate = 3e-4
    temperature = 0.5
    epochs = 30
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/simclr_experiment_1')

    # Model and optimizer
    model = ResNetSimCLR(models.resnet18(weights=None), out_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = NTXentLoss(batch_size, temperature)

    # DataLoader
    loader = get_simclr_dataloader(batch_size, data_dir)

    # Training
    for epoch in range(epochs):
        loss = train_simclr(model, loader, optimizer, criterion, device, writer, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "simclr_resnet18.pth")
    writer.close()


def get_cifar100_dataloader(batch_size, data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model.get_backbone(inputs)
            features.append(outputs.cpu())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def train_linear_classifier(model, features, labels, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    inputs, targets = features.to(device), labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss
    accuracy = correct / total
    writer.add_scalar('Loss/Train', avg_loss, epoch)
    writer.add_scalar('Accuracy/Train', accuracy, epoch)
    return avg_loss, accuracy


def test_linear_classifier(model, features, labels, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = features.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss
    accuracy = correct / total
    writer.add_scalar('Loss/Test', avg_loss, epoch)
    writer.add_scalar('Accuracy/Test', accuracy, epoch)
    return avg_loss, accuracy


def main_linear_eval():
    # Hyperparameters
    batch_size = 256
    learning_rate = 1e-3
    epochs = 30
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/linear_eval_experiment_1')

    # Load pretrained model and replace last layer
    simclr_model = ResNetSimCLR(models.resnet18(weights=None), out_dim=128).to(device)
    simclr_model.load_state_dict(torch.load("simclr_resnet18.pth"))
    linear_model = LinearClassifier(input_dim=512, num_classes=100).to(device)

    optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    train_loader, test_loader = get_cifar100_dataloader(batch_size, data_dir)
    train_features, train_labels = extract_features(simclr_model, train_loader, device)
    test_features, test_labels = extract_features(simclr_model, test_loader, device)

    # Training
    for epoch in range(epochs):
        train_loss, train_acc = train_linear_classifier(linear_model, train_features, train_labels, optimizer, criterion, device, writer, epoch)
        test_loss, test_acc = test_linear_classifier(linear_model, test_features, test_labels, criterion, device, writer, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    writer.close()


def main_supervised():
    # Hyperparameters
    batch_size = 256
    learning_rate = 1e-3
    epochs = 30
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/supervised_experiment_1')

    # Load pretrained supervised model
    supervised_model = models.resnet18(weights='IMAGENET1K_V1')
    supervised_model.fc = LinearClassifier(input_dim=512, num_classes=100).to(device)

    optimizer = optim.Adam(supervised_model.fc.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    train_loader, test_loader = get_cifar100_dataloader(batch_size, data_dir)

    # Training
    for epoch in range(epochs):
        train_loss, train_acc = train_linear_classifier(supervised_model.fc, train_loader, optimizer, criterion, device, writer, epoch)
        test_loss, test_acc = test_linear_classifier(supervised_model.fc, test_loader, criterion, device, writer, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    writer.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python model3.py [simclr|linear_eval|supervised]")
        sys.exit(1)

    if sys.argv[1] == "simclr":
        main_simclr()
    elif sys.argv[1] == "linear_eval":
        main_linear_eval()
    elif sys.argv[1] == "supervised":
        main_supervised()
    else:
        print("Unknown argument:", sys.argv[1])
        sys.exit(1)'''



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


# SimCLR Model
class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = h.squeeze()
        z = self.projection(h)
        return h, z

    def get_backbone(self, x):
        h = self.backbone(x)
        h = h.squeeze()
        return h


# NT-Xent loss
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self._get_correlated_mask().type(torch.bool)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _get_correlated_mask(self):
        mask = torch.ones((2 * self.batch_size, 2 * self.batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positives = torch.cat((sim_i_j, sim_j_i), dim=0)
        negatives = sim[self.mask].view(N, -1)

        labels = torch.zeros(N).to(positives.device).long()
        logits = torch.cat((positives.unsqueeze(1), negatives), dim=1)

        loss = self.criterion(logits, labels)
        return loss / N


def train_simclr(model, loader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    for (x_i, x_j), _ in loader:
        x_i, x_j = x_i.to(device), x_j.to(device)
        h_i, z_i = model(x_i)
        h_j, z_j = model(x_j)
        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/SimCLR', avg_loss, epoch)
    return avg_loss


def get_simclr_dataloader(batch_size, data_dir):
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=SimCLRTransform(augmentation))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


class SimCLRTransform:
    def __init__(self, augmentation):
        self.augmentation = augmentation

    def __call__(self, x):
        return self.augmentation(x), self.augmentation(x)


def main_simclr():
    # Hyperparameters
    batch_size = 256
    learning_rate = 3e-4
    temperature = 0.5
    epochs = 30
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/simclr_experiment_1')

    # Model and optimizer
    model = ResNetSimCLR(models.resnet18(weights=None), out_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = NTXentLoss(batch_size, temperature)

    # DataLoader
    loader = get_simclr_dataloader(batch_size, data_dir)

    # Training
    for epoch in range(epochs):
        loss = train_simclr(model, loader, optimizer, criterion, device, writer, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "simclr_resnet18.pth")
    writer.close()


def get_cifar100_dataloader(batch_size, data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model.get_backbone(inputs)
            features.append(outputs.cpu())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def train_linear_classifier(model, loader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    writer.add_scalar('Loss/Train', avg_loss, epoch)
    writer.add_scalar('Accuracy/Train', accuracy, epoch)
    return avg_loss, accuracy


def test_linear_classifier(model, loader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    writer.add_scalar('Loss/Test', avg_loss, epoch)
    writer.add_scalar('Accuracy/Test', accuracy, epoch)
    return avg_loss, accuracy


def main_linear_eval():
    # Hyperparameters
    batch_size = 256
    learning_rate = 1e-3
    epochs = 30
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/linear_eval_experiment_1')

    # Load pretrained model and replace last layer
    simclr_model = ResNetSimCLR(models.resnet18(weights=None), out_dim=128).to(device)
    simclr_model.load_state_dict(torch.load("simclr_resnet18.pth"))
    linear_model = LinearClassifier(input_dim=512, num_classes=100).to(device)

    optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    train_loader, test_loader = get_cifar100_dataloader(batch_size, data_dir)

    # Extract features
    train_features, train_labels = extract_features(simclr_model, train_loader, device)
    test_features, test_labels = extract_features(simclr_model, test_loader, device)

    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training
    for epoch in range(epochs):
        train_loss, train_acc = train_linear_classifier(linear_model, train_loader, optimizer, criterion, device, writer, epoch)
        test_loss, test_acc = test_linear_classifier(linear_model, test_loader, criterion, device, writer, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Save model
    torch.save(linear_model.state_dict(), "linear_classifier.pth")
    writer.close()


def main_supervised():
    # Hyperparameters
    batch_size = 256
    learning_rate = 1e-3
    epochs = 30
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/supervised_experiment_1')

    # Load pretrained model and replace last layer
    supervised_model = models.resnet18(weights=None).to(device)
    supervised_model.fc = nn.Linear(supervised_model.fc.in_features, 100).to(device)

    optimizer = optim.Adam(supervised_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    train_loader, test_loader = get_cifar100_dataloader(batch_size, data_dir)

    # Training
    for epoch in range(epochs):
        train_loss, train_acc = train_linear_classifier(supervised_model, train_loader, optimizer, criterion, device, writer, epoch)
        test_loss, test_acc = test_linear_classifier(supervised_model, test_loader, criterion, device, writer, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Save model
    torch.save(supervised_model.state_dict(), "supervised_resnet18.pth")
    writer.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python model3.py [simclr|linear|supervised]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "simclr":
        main_simclr()
    elif mode == "linear":
        main_linear_eval()
    elif mode == "supervised":
        main_supervised()
    else:
        print("Invalid mode. Choose from [simclr|linear|supervised]")
        sys.exit(1)
