"""
MNIST 手写数字识别示例

使用 Bears 框架构建 MLP 模型，在 MNIST 数据集上进行训练和测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from bears import (
    Sequential, Dense, ReLU, Softmax,
    CrossEntropyLoss, SGD, Adam,
    accuracy,
    create_dummy_mnist, normalize, flatten, one_hot_encode
)


def main():
    """主函数"""
    print("=" * 70)
    print("Bears 🐻 - MNIST 手写数字识别示例")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/6] Loading MNIST dataset...")
    try:
        # 尝试加载真实 MNIST 数据
        from bears import load_mnist_simple
        (X_train, y_train), (X_test, y_test) = load_mnist_simple('./data')
        print(f"Loaded real MNIST dataset")
    except:
        # 如果加载失败，使用虚拟数据
        print("Real MNIST not found, using dummy data for demonstration")
        (X_train, y_train), (X_test, y_test) = create_dummy_mnist(
            n_train=1000, n_test=200
        )
    
    print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")
    
    # 2. 数据预处理
    print("\n[2/6] Preprocessing data...")
    
    # 归一化: 将像素值从 [0, 255] 缩放到 [0, 1]
    X_train = normalize(X_train, method='scale')
    X_test = normalize(X_test, method='scale')
    
    # 展平: 将 28x28 图像展平为 784 维向量
    X_train = flatten(X_train)
    X_test = flatten(X_test)
    
    # One-hot 编码: 将标签转换为 10 维向量
    y_train_onehot = one_hot_encode(y_train, num_classes=10)
    y_test_onehot = one_hot_encode(y_test, num_classes=10)
    
    print(f"Preprocessed training data: {X_train.shape}")
    print(f"Preprocessed training labels: {y_train_onehot.shape}")
    
    # 3. 构建模型
    print("\n[3/6] Building model...")
    
    model = Sequential()
    model.add(Dense(784, 128))      # 输入层 -> 隐藏层1: 784 -> 128
    model.add(ReLU())               # ReLU 激活
    model.add(Dense(128, 64))       # 隐藏层1 -> 隐藏层2: 128 -> 64
    model.add(ReLU())               # ReLU 激活
    model.add(Dense(64, 10))        # 隐藏层2 -> 输出层: 64 -> 10
    model.add(Softmax())            # Softmax 激活
    
    print("\nModel architecture:")
    model.summary()
    
    # 4. 编译模型
    print("\n[4/6] Compiling model...")
    
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.001)
    # optimizer = SGD(learning_rate=0.01, momentum=0.9)  # 也可以使用 SGD
    
    model.compile(loss=loss_fn, optimizer=optimizer)
    print(f"Loss function: {loss_fn}")
    print(f"Optimizer: {optimizer}")
    
    # 5. 训练模型
    print("\n[5/6] Training model...")
    
    history = model.fit(
        X_train, y_train_onehot,
        epochs=10,
        batch_size=32,
        verbose=True,
        validation_data=(X_test, y_test_onehot)
    )
    
    # 6. 评估模型
    print("\n[6/6] Evaluating model...")
    
    # 在训练集上评估
    y_train_pred = model.predict(X_train)
    train_acc = accuracy(y_train_pred, y_train_onehot)
    print(f"Training accuracy: {train_acc * 100:.2f}%")
    
    # 在测试集上评估
    y_test_pred = model.predict(X_test)
    test_acc = accuracy(y_test_pred, y_test_onehot)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    
    # 显示一些预测示例
    print("\n" + "=" * 70)
    print("Sample predictions:")
    print("=" * 70)
    n_samples = min(10, len(X_test))
    for i in range(n_samples):
        pred_label = np.argmax(y_test_pred[i])
        true_label = y_test[i]
        confidence = y_test_pred[i][pred_label] * 100
        status = "✓" if pred_label == true_label else "✗"
        print(f"{status} Sample {i+1}: Predicted={pred_label}, True={true_label}, Confidence={confidence:.1f}%")
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    
    # 可选: 保存模型权重
    # model.save_weights('mnist_model.npy')
    # print("\nModel weights saved to 'mnist_model.npy'")


if __name__ == '__main__':
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    main()
