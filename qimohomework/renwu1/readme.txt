#SimCLR with CIFAR-10 and CIFAR-100

## 简介

本项目实现了 SimCLR 的训练过程，并在 CIFAR-10 数据集上进行了无监督预训练，然后在 CIFAR-100 数据集上进行线性分类和监督学习的对比实验。整个过程包括数据预处理、模型定义、训练和评估。

## 环境要求

- Python 3.8+
- PyTorch 2.3.1
- torchvision
- torchsummary
- tensorboard

## 安装

1. 克隆此仓库到本地：

    ```bash
    git clone https://github.com/your_username/simclr-cifar.git
    cd simclr-cifar
    ```

2. 安装所需的 Python 包：

    ```bash
    pip install -r requirements.txt
    ```

## 数据集

该项目使用 CIFAR-10 和 CIFAR-100 数据集。数据集会在运行代码时自动下载并解压到 `./data` 目录下。

## 使用说明

### SimCLR 训练

1. 运行以下命令进行 SimCLR 模型训练：

    ```bash
    python model3.py simclr
    ```

2. 训练过程中的损失曲线会被记录在 `runs/simclr_experiment_1` 文件夹中，可以通过以下命令使用 TensorBoard 可视化：

    ```bash
    tensorboard --logdir=runs/simclr_experiment_1
    ```

3. 训练完成后，模型权重将保存为 `simclr_resnet18.pth`。

### 线性评估

1. 在 SimCLR 训练完成后，运行以下命令进行线性分类器训练：

    ```bash
    python model3.py linear
    ```

2. 训练过程中的损失和准确率曲线会被记录在 `runs/linear_eval_experiment_1` 文件夹中，可以通过以下命令使用 TensorBoard 可视化：

    ```bash
    tensorboard --logdir=runs/linear_eval_experiment_1
    ```

3. 训练完成后，模型权重将保存为 `linear_classifier.pth`。

### 监督学习

1. 运行以下命令进行监督学习模型训练：

    ```bash
    python model3.py supervised
    ```

2. 训练过程中的损失和准确率曲线会被记录在 `runs/supervised_experiment_1` 文件夹中，可以通过以下命令使用 TensorBoard 可视化：

    ```bash
    tensorboard --logdir=runs/supervised_experiment_1
    ```

3. 训练完成后，模型权重将保存为 `supervised_resnet18.pth`。