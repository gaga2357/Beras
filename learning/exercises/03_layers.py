"""
阶段 3：神经网络层
========================================

学习目标：
1. 实现全连接层（Dense Layer）
2. 理解前向传播和反向传播
3. 掌握权重初始化方法

关键概念：
- 全连接层：y = xW + b
- 权重初始化（Xavier/He）
- 参数管理
"""

import numpy as np
from exercises.01_tensor import Tensor
from exercises.02_activations import relu, sigmoid, softmax


class Layer:
    """
    神经网络层基类
    
    所有层都应该继承这个基类并实现：
    - forward(x): 前向传播
    - backward(grad): 反向传播（可选，如果使用自动微分）
    - get_parameters(): 返回可训练参数
    """
    
    def __init__(self):
        self.trainable = True  # 是否可训练
        self.training = True   # 训练模式 or 推理模式
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入 Tensor
        
        Returns:
            Tensor: 输出
        """
        raise NotImplementedError
    
    def __call__(self, x):
        """使层可以像函数一样调用"""
        return self.forward(x)
    
    def get_parameters(self):
        """
        返回可训练参数
        
        Returns:
            list: 参数列表 [weight, bias, ...]
        """
        return []
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False


class Dense(Layer):
    """
    全连接层 (Dense Layer / Fully Connected Layer)
    
    公式：y = xW + b
    
    其中：
    - x: 输入 [batch_size, input_dim]
    - W: 权重 [input_dim, output_dim]
    - b: 偏置 [output_dim]
    - y: 输出 [batch_size, output_dim]
    
    TODO: 任务 3.2, 3.3, 3.4
    """
    
    def __init__(self, input_dim, output_dim, use_bias=True, 
                 weight_init='xavier'):
        """
        初始化 Dense 层
        
        TODO: 任务 3.4
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            use_bias: 是否使用偏置
            weight_init: 权重初始化方法 ('xavier', 'he', 'normal')
        
        关键问题思考：
        1. 为什么不能全部初始化为 0？
        2. Xavier 和 He 初始化有什么区别？
        3. 偏置通常初始化为什么？
        
        权重初始化方法：
        - Xavier: W ~ N(0, 2/(input_dim + output_dim))
        - He: W ~ N(0, 2/input_dim)  # 适合 ReLU
        - Normal: W ~ N(0, 0.01)
        
        提示：
        - 使用 np.random.randn 生成随机数
        - 权重需要 requires_grad=True
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        # TODO: 初始化权重和偏置
        # self.weight = Tensor(..., requires_grad=True)
        # self.bias = Tensor(..., requires_grad=True) if use_bias else None
        pass
    
    def forward(self, x):
        """
        前向传播
        
        TODO: 任务 3.2
        
        公式：y = xW + b
        
        关键问题思考：
        1. 如何处理批量数据？
        2. 输入形状是什么？输出形状是什么？
        3. 为什么需要缓存输入 x？
        
        Args:
            x: Tensor, shape [batch_size, input_dim]
        
        Returns:
            Tensor, shape [batch_size, output_dim]
        
        提示：
        - 使用 x.matmul(self.weight)
        - 如果有偏置，使用 + self.bias（会自动广播）
        """
        # TODO: 在这里实现
        pass
    
    def get_parameters(self):
        """
        返回可训练参数
        
        Returns:
            list: [weight, bias]（如果有偏置）
        """
        params = [self.weight]
        if self.use_bias:
            params.append(self.bias)
        return params


class ReLU(Layer):
    """
    ReLU 激活层
    
    TODO: 任务 3.5
    将 02_activations.py 中的 relu 函数封装成层
    """
    
    def __init__(self):
        super().__init__()
        self.trainable = False  # 激活层没有可训练参数
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: Tensor
        
        Returns:
            Tensor
        """
        # TODO: 调用 relu 函数
        pass


class Sigmoid(Layer):
    """
    Sigmoid 激活层
    
    TODO: 封装 sigmoid 函数
    """
    
    def __init__(self):
        super().__init__()
        self.trainable = False
    
    def forward(self, x):
        # TODO: 调用 sigmoid 函数
        pass


class Softmax(Layer):
    """
    Softmax 激活层
    
    TODO: 封装 softmax 函数
    """
    
    def __init__(self, axis=-1):
        super().__init__()
        self.trainable = False
        self.axis = axis
    
    def forward(self, x):
        # TODO: 调用 softmax 函数
        pass


# ==================== 测试代码 ====================

def test_dense_forward():
    """测试 Dense 层前向传播"""
    print("=" * 50)
    print("测试 1：Dense 层前向传播")
    print("=" * 50)
    
    # 创建层
    layer = Dense(input_dim=3, output_dim=2)
    
    # 输入数据
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    
    # 前向传播
    y = layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"权重形状: {layer.weight.shape}")
    print(f"输出形状: {y.shape}")
    print(f"期望输出形状: (1, 2)")
    
    assert y.shape == (1, 2), "输出形状错误！"
    print("✅ 形状检查通过！")
    print()


def test_dense_backward():
    """测试 Dense 层反向传播"""
    print("=" * 50)
    print("测试 2：Dense 层反向传播")
    print("=" * 50)
    
    # 创建层
    layer = Dense(input_dim=2, output_dim=1)
    
    # 简单的输入
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    
    # 前向传播
    y = layer(x)
    
    # 反向传播
    y.backward()
    
    print(f"输入: {x.data}")
    print(f"输出: {y.data}")
    print(f"输入梯度: {x.grad}")
    print(f"权重梯度形状: {layer.weight.grad.shape if layer.weight.grad is not None else None}")
    print(f"偏置梯度形状: {layer.bias.grad.shape if layer.bias.grad is not None else None}")
    
    assert layer.weight.grad is not None, "权重梯度未计算！"
    assert layer.bias.grad is not None, "偏置梯度未计算！"
    print("✅ 梯度计算通过！")
    print()


def test_activation_layers():
    """测试激活层"""
    print("=" * 50)
    print("测试 3：激活层")
    print("=" * 50)
    
    x = Tensor([[-1.0, 0.0, 1.0]], requires_grad=True)
    
    # ReLU
    relu_layer = ReLU()
    y_relu = relu_layer(x)
    print(f"ReLU 输入: {x.data}")
    print(f"ReLU 输出: {y_relu.data}")
    print(f"期望: [[0, 0, 1]]")
    
    # Sigmoid
    sigmoid_layer = Sigmoid()
    y_sigmoid = sigmoid_layer(x)
    print(f"\nSigmoid 输入: {x.data}")
    print(f"Sigmoid 输出: {y_sigmoid.data}")
    
    # Softmax
    softmax_layer = Softmax()
    y_softmax = softmax_layer(x)
    print(f"\nSoftmax 输入: {x.data}")
    print(f"Softmax 输出: {y_softmax.data}")
    print(f"Softmax 输出和: {y_softmax.data.sum()}（应该是 1.0）")
    print()


def test_multi_layer():
    """测试多层网络"""
    print("=" * 50)
    print("测试 4：多层网络")
    print("=" * 50)
    
    # 构建一个简单的 2 层网络
    layer1 = Dense(input_dim=3, output_dim=4)
    relu1 = ReLU()
    layer2 = Dense(input_dim=4, output_dim=2)
    
    # 输入
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    
    # 前向传播
    h = layer1(x)
    h = relu1(h)
    y = layer2(h)
    
    print(f"输入形状: {x.shape}")
    print(f"隐藏层形状: {h.shape}")
    print(f"输出形状: {y.shape}")
    
    # 反向传播
    y.backward()
    
    print(f"\n第1层权重梯度形状: {layer1.weight.grad.shape if layer1.weight.grad is not None else None}")
    print(f"第2层权重梯度形状: {layer2.weight.grad.shape if layer2.weight.grad is not None else None}")
    
    assert layer1.weight.grad is not None, "第1层权重梯度未计算！"
    assert layer2.weight.grad is not None, "第2层权重梯度未计算！"
    print("✅ 多层网络测试通过！")
    print()


def test_weight_initialization():
    """测试权重初始化"""
    print("=" * 50)
    print("测试 5：权重初始化")
    print("=" * 50)
    
    # Xavier 初始化
    layer_xavier = Dense(100, 100, weight_init='xavier')
    print(f"Xavier 初始化:")
    print(f"  权重均值: {layer_xavier.weight.data.mean():.6f}")
    print(f"  权重标准差: {layer_xavier.weight.data.std():.6f}")
    print(f"  理论标准差: {np.sqrt(2.0 / (100 + 100)):.6f}")
    
    # He 初始化
    layer_he = Dense(100, 100, weight_init='he')
    print(f"\nHe 初始化:")
    print(f"  权重均值: {layer_he.weight.data.mean():.6f}")
    print(f"  权重标准差: {layer_he.weight.data.std():.6f}")
    print(f"  理论标准差: {np.sqrt(2.0 / 100):.6f}")
    print()


if __name__ == "__main__":
    print("\n")
    print("🐻 Bears 学习之旅 - 阶段 3：神经网络层")
    print("\n")
    
    # TODO: 完成层的实现后，运行以下测试
    # test_dense_forward()
    # test_dense_backward()
    # test_activation_layers()
    # test_multi_layer()
    # test_weight_initialization()
    
    print("\n")
    print("💡 提示：")
    print("1. 先实现 Dense 层的 __init__ 和 forward")
    print("2. 测试前向传播的形状是否正确")
    print("3. 利用自动微分测试反向传播")
    print("4. 实现权重初始化")
    print("5. 封装激活函数为层")
    print("6. 思考：为什么需要不同的初始化方法？")
    print("\n")
