import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import gzip
from urllib.request import urlretrieve
import os


# 定义激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# 定义交叉熵损失函数
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / m
    return loss


# 定义三层神经网络模型
class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, reg_lambda):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda  # 添加正则化参数
        self.params = self.initialize_parameters()

    def initialize_parameters(self):
        params = {}
        params['W1'] = np.random.randn(self.input_size, self.hidden_size)
        params['b1'] = np.zeros((1, self.hidden_size))
        params['W2'] = np.random.randn(self.hidden_size, self.output_size)
        params['b2'] = np.zeros((1, self.output_size))
        return params

    def forward_propagation(self, X):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        Z1 = np.dot(X, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)
        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return cache

    def backward_propagation(self, X, y, cache):
        m = X.shape[0]
        A1, A2 = cache['A1'], cache['A2']
        W1, W2 = self.params['W1'], self.params['W2']

        dZ2 = A2 - y
        dW2 = (np.dot(A1.T, dZ2) + self.reg_lambda * W2) / m  # 添加正则化项
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(cache['Z1'])
        dW1 = (np.dot(X.T, dZ1) + self.reg_lambda * W1) / m  # 添加正则化项
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.params['W2'] -= self.learning_rate * dW2
        self.params['b2'] -= self.learning_rate * db2
        self.params['W1'] -= self.learning_rate * dW1
        self.params['b1'] -= self.learning_rate * db1

    def train(self, X_train, y_train, X_val, y_val, num_epochs=100):
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')
        best_params = None
        for epoch in range(num_epochs):
            cache = self.forward_propagation(X_train)
            loss = cross_entropy_loss(y_train, cache['A2'])
            val_cache = self.forward_propagation(X_val)
            val_loss = cross_entropy_loss(y_val, val_cache['A2'])
            val_predictions = self.predict(X_val)
            val_accuracy = np.mean(val_predictions == np.argmax(y_val, axis=1))

            train_losses.append(loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = self.params.copy()

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, Training Loss: {loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

            self.backward_propagation(X_train, y_train, cache)

        self.params = best_params

        # 可视化训练过程中的损失和准确率
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(num_epochs), train_losses, label='Training Loss')
        plt.plot(range(num_epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.show()

    def predict(self, X):
        cache = self.forward_propagation(X)
        predictions = np.argmax(cache['A2'], axis=1)
        return predictions


if __name__ == "__main__":
    # 加载数据集并进行预处理
    def load_data():
        def load_images(filename):
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            return data.reshape(-1, 28 * 28) / 255.0  # 归一化到 [0, 1]

        def load_labels(filename):
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            return data

        def download_data(url, filename):
            if not os.path.exists(filename):
                print("Downloading", filename, "...")
                urlretrieve(url, filename)
            print("Data", filename, "is ready!")

        download_data('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
                      'train-images-idx3-ubyte.gz')
        download_data('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
                      'train-labels-idx1-ubyte.gz')
        download_data('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
                      't10k-images-idx3-ubyte.gz')
        download_data('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
                      't10k-labels-idx1-ubyte.gz')

        X_train = load_images('train-images-idx3-ubyte.gz')
        y_train = load_labels('train-labels-idx1-ubyte.gz')
        X_test = load_images('t10k-images-idx3-ubyte.gz')
        y_test = load_labels('t10k-labels-idx1-ubyte.gz')

        # 对标签进行 one-hot 编码
        encoder = OneHotEncoder(categories='auto', sparse=False)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = encoder.transform(y_test.reshape(-1, 1))

        return X_train, y_train, X_test, y_test


    X_train, y_train, X_test, y_test = load_data()

    # 划分训练集和测试集，设置验证集占比为0.1，测试集占比为0.1
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # 设置超参数
    input_size = 784  # 输入特征维度
    hidden_size = 256 # 隐藏层大小
    output_size = 10  # 输出类别数
    learning_rate = 0.7  # 学习率
    reg_lambda = 0.85 # 正则化参数
    num_epochs = 60  # 迭代次数

    # 创建三层神经网络模型
    model = ThreeLayerNN(input_size, hidden_size, output_size, learning_rate, reg_lambda)

    # 训练模型
    model.train(X_train, y_train, X_val, y_val, num_epochs=num_epochs)

    # 在测试集上评估模型性能
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
    print(f"Test Accuracy: {accuracy}")
