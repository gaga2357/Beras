# Bears 🐻 - 从零实现的深度学习框架

Bears 是一个完全从零实现的深度学习框架，模仿 Keras API 设计，仅使用 NumPy 作为数值计算库。

## ✨ 特性

- 🔥 自动微分引擎 - 完整的计算图和反向传播
- 🧠 神经网络层 - Dense, ReLU, Sigmoid, Softmax, Dropout
- 📊 损失函数 - MSE, CrossEntropy, BinaryCrossEntropy
- ⚡ 优化器 - SGD, Adam, RMSprop, AdaGrad
- 🎯 Keras 风格 API - 简洁易用
- 📦 零依赖 - 仅使用 NumPy

## 🚀 快速开始

### 简单示例

```python
import bears

# 创建模型
model = bears.Sequential([
    bears.Dense(128, input_dim=784),
    bears.ReLU(),
    bears.Dense(10),
    bears.Softmax()
])

# 编译模型
model.compile(
    loss=bears.CrossEntropyLoss(),
    optimizer=bears.Adam(lr=0.001),
    metrics=['accuracy']
)

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 评估模型
loss, acc = model.evaluate(X_test, y_test)
```

### MNIST 示例

```bash
# 使用真实 MNIST 数据集（自动下载）
python examples/mnist_real.py

# 使用虚拟数据快速测试
python examples/mnist_example.py
```

**真实 MNIST 测试结果：**
- 训练集准确率：99.68%
- 测试集准确率：97.84%

## 📚 API 文档

### 模型

- `Sequential(layers)` - 顺序模型

### 层

- `Dense(units, input_dim=None)` - 全连接层
- `ReLU()` - ReLU 激活层
- `Sigmoid()` - Sigmoid 激活层
- `Softmax()` - Softmax 激活层
- `Dropout(rate)` - Dropout 层

### 损失函数

- `MSELoss()` - 均方误差
- `CrossEntropyLoss()` - 交叉熵损失
- `BinaryCrossEntropyLoss()` - 二元交叉熵

### 优化器

- `SGD(lr, momentum=0)` - 随机梯度下降
- `Adam(lr, beta1=0.9, beta2=0.999)` - Adam 优化器
- `RMSprop(lr, decay_rate=0.9)` - RMSprop 优化器
- `AdaGrad(lr)` - AdaGrad 优化器

## 📖 更多文档

- [快速开始](快速开始.md) - 5 分钟上手指南
- [产品文档](产品文档.md) - 功能详细说明
- [设计需求](设计需求.md) - 技术设计文档
- [执行计划](执行计划.md) - 开发计划

## 🎓 学习价值

通过这个项目，可以深入理解：
- 自动微分原理
- 反向传播算法
- 神经网络实现
- 优化算法
- 深度学习框架设计

## 📄 许可证

MIT License
