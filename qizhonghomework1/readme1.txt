以下是该实验的README文件示例，详细说明如何在CUB-200-2011数据集上微调预训练的ResNet-18模型进行鸟类识别，包括训练和测试的步骤。请根据具体项目需求和实际路径修改示例中的路径和参数。

---

# 使用微调的ResNet-18进行鸟类分类

本仓库包含在CUB-200-2011数据集上微调预训练的ResNet-18模型进行鸟类分类的实现。

## 目录
- [简介](#简介)
- [数据集](#数据集)
- [安装](#安装)
- [训练](#训练)
- [测试](#测试)
- [结果可视化](#结果可视化)
- [模型和报告](#模型和报告)

## 简介

该项目展示了如何在CUB-200-2011数据集上微调预训练的ResNet-18模型，以进行鸟类分类。主要步骤包括修改网络架构，在新数据集上微调，并与从头开始训练的模型进行性能对比。

## 数据集

CUB-200-2011数据集包含200种鸟类，共有11,788张图片。每张图片都带有属性和边框标注。

从[CUB-200-2011](https://data.caltech.edu/records/65de6-vp158)下载数据集，并解压到指定目录。

## 安装

1. 克隆此仓库：
   ```bash
   git clone https://github.com/yourusername/bird-classification.git
   cd bird-classification
   ```

2. 创建虚拟环境并激活：
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows系统使用 `venv\Scripts\activate`
   ```

3. 安装所需包：
   ```bash
   pip install -r requirements.txt
   ```

4. 确保已安装TensorFlow和TensorBoard：
   ```bash
   pip install tensorflow tensorboard
   ```

## 训练

运行以下命令进行模型训练：

```bash
python train.py --data_root path/to/CUB_200_2011 --log_dir logs/fit
```

- `data_root`：CUB-200-2011数据集的根目录路径。
- `log_dir`：保存TensorBoard日志的目录。

训练脚本将：
1. 加载预训练的ResNet-18模型。
2. 修改最后一层为200个输出类别。
3. 从头训练最后一层，并以较小学习率微调其余网络。

## 测试

运行以下命令进行模型测试：

```bash
python test.py --data_root path/to/CUB_200_2011 --model_path path/to/trained_model.h5
```

- `data_root`：CUB-200-2011数据集的根目录路径。
- `model_path`：保存的已训练模型路径。

测试脚本将：
1. 加载已训练的模型。
2. 在测试集上评估模型。
3. 输出测试集上的准确率和损失。

## 结果可视化

运行以下命令使用TensorBoard可视化训练过程：

```bash
tensorboard --logdir=logs/fit
```

打开浏览器并访问 `http://localhost:6006/` 查看TensorBoard仪表盘，包括训练集和验证集上的损失和准确率曲线。

## 模型和报告

已训练的模型和实验报告可通过以下链接获取：
- 已训练模型：[Google Drive Link](https://drive.google.com/your_model_link)
- 实验报告：[PDF Report](https://your_report_link)

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

---

### 示例训练脚本 (`train.py`)

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True, help='CUB_200_2011数据集路径')
parser.add_argument('--log_dir', type=str, default='logs/fit', help='保存TensorBoard日志的路径')
args = parser.parse_args()

# 数据增强和预处理
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    args.data_root + '/images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    args.data_root + '/images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 加载预训练的ResNet-50模型
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = Flatten()(base_model.output)
x = Dense(200, activation='softmax')(x)

# 定义模型
model = Model(inputs=base_model.input, outputs=x)

# 冻结除最后一层外的所有层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# TensorBoard回调
tensorboard_callback = TensorBoard(log_dir=args.log_dir, histogram_freq=1)

# 训练模型
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[tensorboard_callback]
)

# 保存模型
model.save('resnet50_finetuned.h5')
```

### 示例测试脚本 (`test.py`)

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True, help='CUB_200_2011数据集路径')
parser.add_argument('--model_path', type=str, required=True, help='已训练模型路径')
args = parser.parse_args()

# 数据预处理
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    args.data_root + '/images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 加载已训练模型
model = tf.keras.models.load_model(args.model_path)

# 评估模型
loss, accuracy = model.evaluate(test_generator)
print(f'测试损失: {loss}')
print(f'测试准确率: {accuracy}')
```

以上是详细的中文README文件，包括了如何进行训练和测试的步骤。根据实际情况调整文件路径和参数。