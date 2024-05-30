import os
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

def preprocess_dataset(data_path, train_dir, val_dir):
    # 设置文件路径
    images_path = os.path.join(data_path, 'images')
    train_test_split_file = os.path.join(data_path, 'train_test_split.txt')
    images_file = os.path.join(data_path, 'images.txt')
    labels_file = os.path.join(data_path, 'image_class_labels.txt')

    # 创建目标文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 读取文件
    with open(images_file, 'r') as f:
        image_lines = f.readlines()

    with open(train_test_split_file, 'r') as f:
        split_lines = f.readlines()

    with open(labels_file, 'r') as f:
        label_lines = f.readlines()

    # 创建类别文件夹
    classes = {}
    for line in label_lines:
        image_id, class_id = line.strip().split()
        class_id = int(class_id)
        if class_id not in classes:
            classes[class_id] = []
        classes[class_id].append(image_id)

    # 处理每张图片
    for i, line in enumerate(image_lines):
        image_id, image_name = line.strip().split()
        _, is_train = split_lines[i].strip().split()
        class_id = None
        for key, value in classes.items():
            if image_id in value:
                class_id = key
                break
        if class_id is None:
            continue
        class_folder = f"{class_id:03d}"
        src_path = os.path.join(images_path, image_name)
        if is_train == '1':
            dst_folder = os.path.join(train_dir, class_folder)
        else:
            dst_folder = os.path.join(val_dir, class_folder)
        os.makedirs(dst_folder, exist_ok=True)
        dst_path = os.path.join(dst_folder, os.path.basename(image_name))
        shutil.copy(src_path, dst_path)
    print("Dataset preprocessing complete.")

def save_model_weights_to_csv(model, save_path):
    # 获取模型的参数
    state_dict = model.state_dict()

    # 将参数转换为 DataFrame
    data = []
    for layer_name, layer_params in state_dict.items():
        if isinstance(layer_params, torch.Tensor):
            shape = list(layer_params.shape)
            values = layer_params.cpu().detach().numpy().flatten()
            for i, value in enumerate(values):
                param_name = f'{layer_name}[{i}]'
                data.append({'Layer Name': layer_name, 'Parameter Name': param_name, 'Shape': shape, 'Value': value})

    df = pd.DataFrame(data)

    # 拆分 DataFrame 并保存到多个 CSV 文件
    max_rows_per_file = 1048576
    num_files = (len(df) - 1) // max_rows_per_file + 1

    for i in range(num_files):
        start_row = i * max_rows_per_file
        end_row = min((i + 1) * max_rows_per_file, len(df))
        df_subset = df.iloc[start_row:end_row]
        subset_save_path = save_path.replace('.csv', f'_{i+1}.csv')
        df_subset.to_csv(subset_save_path, index=False)
        print(f"Saved {subset_save_path}")

def train_model(train_data_path, val_data_path, num_epochs=20):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载CUB-200-2011数据集
    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 加载预训练的模型（这里以ResNet-18为例）
    model = models.resnet18(pretrained=True)

    # 修改输出层
    model.fc = nn.Linear(512, 200)  # 200是CUB-200-2011数据集中的类别数量

    # 冻结预训练参数，只微调最后的全连接层
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    # 将模型移到GPU上
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)  # 使用较小的学习率微调输出层参数

    # 创建TensorBoard的SummaryWriter
    writer = SummaryWriter()

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # 每100个batch记录一次训练loss
            if i % 100 == 99:
                writer.add_scalar('Training Loss', running_loss / 100, epoch * len(train_loader) + i)
                running_loss = 0.0
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        writer.add_scalar('Training Accuracy', train_acc, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # 在验证集上评估模型性能
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")

    writer.close()

    # 保存模型的权重到 CSV 文件
    save_path = os.path.join(os.path.dirname(train_data_path), 'model_weights.csv')
    save_model_weights_to_csv(model, save_path)
    print(f"Model weights saved to {save_path}")

if __name__ == "__main__":
    data_path = r'C:\Users\machao\Desktop\CUB_200_2011\CUB_200_2011'  # 修改为实际的数据集路径
    train_dir = r'C:\Users\machao\Desktop\CUB_200_2011\CUB_200_2011\train'
    val_dir = r'C:\Users\machao\Desktop\CUB_200_2011\CUB_200_2011\val'

    preprocess_dataset(data_path, train_dir, val_dir)
    train_model(train_dir, val_dir)
