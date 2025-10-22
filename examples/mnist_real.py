"""
MNIST 真实数据集训练示例
使用真实的 MNIST 手写数字数据集训练神经网络
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from bears import Sequential, Dense, ReLU, Softmax
from bears import CrossEntropyLoss, Adam, accuracy
from bears import load_mnist_simple, normalize, flatten, one_hot_encode, get_batches

print("=" * 60)
print("Bears 🐻 - MNIST 真实数据集训练示例")
print("=" * 60)

# 1. 加载真实 MNIST 数据集
print("\n[1/6] 加载 MNIST 数据集...")
try:
    (X_train, y_train), (X_test, y_test) = load_mnist_simple(path='../data', download=True)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
except Exception as e:
    print(f"加载失败: {e}")
    print("将使用虚拟数据进行演示...")
    from bears import create_dummy_mnist
    (X_train, y_train), (X_test, y_test) = create_dummy_mnist(n_train=1000, n_test=200)

# 2. 数据预处理
print("\n[2/6] 数据预处理...")
X_train = normalize(flatten(X_train), method='scale')
X_test = normalize(flatten(X_test), method='scale')
y_train_onehot = one_hot_encode(y_train, num_classes=10)
y_test_onehot = one_hot_encode(y_test, num_classes=10)

print(f"训练数据形状: {X_train.shape}")
print(f"测试数据形状: {X_test.shape}")

# 3. 构建模型
print("\n[3/6] 构建神经网络模型...")
model = Sequential()
model.add(Dense(784, 256))
model.add(ReLU())
model.add(Dense(256, 128))
model.add(ReLU())
model.add(Dense(128, 10))
model.add(Softmax())

# 4. 编译模型
print("\n[4/6] 编译模型...")
model.compile(
    loss=CrossEntropyLoss(),
    optimizer=Adam(learning_rate=0.001)
)

model.summary()

# 5. 训练模型
print("\n[5/6] 开始训练...")
epochs = 10
batch_size = 128

for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0
    
    # 批次训练
    for batch_X, batch_y in get_batches(X_train, y_train_onehot, batch_size=batch_size):
        # 前向传播
        y_pred = model.forward(batch_X)
        
        # 计算损失
        loss = model.loss_fn(y_pred, batch_y)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        model.optimizer.step(model.get_parameters())
        
        # 梯度清零
        for param in model.get_parameters():
            param.zero_grad()
        
        epoch_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
        n_batches += 1
    
    # 计算平均损失
    avg_loss = epoch_loss / n_batches
    
    # 每个 epoch 评估一次
    y_pred = model.predict(X_test)
    test_acc = accuracy(y_pred, y_test_onehot)
    
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Test Acc: {test_acc * 100:.2f}%")

# 6. 最终评估
print("\n[6/6] 最终评估...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy(y_train_pred, y_train_onehot)
test_acc = accuracy(y_test_pred, y_test_onehot)

print(f"\n训练集准确率: {train_acc * 100:.2f}%")
print(f"测试集准确率: {test_acc * 100:.2f}%")

# 显示一些预测示例
print("\n预测示例（前 10 个测试样本）:")
print("真实标签:", y_test[:10])
print("预测标签:", np.argmax(y_test_pred[:10], axis=1))

print("\n" + "=" * 60)
print("训练完成！🎉")
print("=" * 60)
