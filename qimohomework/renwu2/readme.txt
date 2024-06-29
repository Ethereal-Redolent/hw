# CIFAR-100 Classification with CNN and Transformer

## 简介

本项目实现了基于 ResNet 的卷积神经网络（CNN）和 Vision Transformer（ViT）在 CIFAR-100 数据集上的图像分类任务。训练过程中应用了 CutMix 数据增强技术，并提供了模型训练和验证的代码。

## 环境要求

- Python 3.8+
- PyTorch 2.3.1
- torchvision
- vit-pytorch
- numpy
- tensorboard

## 安装

1. 克隆此仓库到本地：

    ```bash
    git clone https://github.com/your_username/cifar100-classification.git
    cd cifar100-classification
    ```

2. 安装所需的 Python 包：

    ```bash
    pip install -r requirements.txt
    ```

## 数据集

该项目使用 CIFAR-100 数据集。数据集会在运行代码时自动下载并解压到 `./data` 目录下。

## 使用说明

### 训练和验证

1. 运行以下命令进行 CNN 模型训练和验证：

    ```bash
    python train_and_validate.py --model cnn
    ```

2. 运行以下命令进行 Transformer 模型训练和验证：

    ```bash
    python train_and_validate.py --model transformer
    ```

3. 训练和验证过程中，会在 `runs/cifar100_experiment` 文件夹中记录损失和准确率曲线，并保存每个 epoch 的模型权重到 CSV 文件。

### 使用 TensorBoard 可视化

1. 运行以下命令启动 TensorBoard：

    ```bash
    tensorboard --logdir=runs/cifar100_experiment
    ```

2. 打开浏览器并访问 [http://localhost:6006](http://localhost:6006)，查看训练和验证的损失和准确率曲线。
