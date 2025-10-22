"""
Layers 模块 - 实现神经网络层
"""

import numpy as np
from .tensor import Tensor, relu, sigmoid, softmax


class Layer:
    """
    层的基类
    
    所有神经网络层都应该继承这个类
    """
    
    def __init__(self):
        self.trainable = True
        self.parameters = []
    
    def forward(self, x):
        """
        前向传播(抽象方法)
        
        Args:
            x: 输入张量
        
        Returns:
            输出张量
        """
        raise NotImplementedError("Subclass must implement forward method")
    
    def __call__(self, x):
        """调用接口"""
        return self.forward(x)
    
    def get_parameters(self):
        """
        获取层的参数
        
        Returns:
            参数列表
        """
        return self.parameters


class Dense(Layer):
    """
    全连接层 (Dense Layer)
    
    实现: y = xW + b
    
    Attributes:
        input_dim: 输入维度
        output_dim: 输出维度
        weights: 权重矩阵 [input_dim, output_dim]
        bias: 偏置向量 [output_dim]
        use_bias: 是否使用偏置
    """
    
    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        初始化全连接层
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            use_bias: 是否使用偏置
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        # Xavier 初始化
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.weights = Tensor(
            np.random.uniform(-limit, limit, (input_dim, output_dim)),
            requires_grad=True
        )
        
        if use_bias:
            self.bias = Tensor(
                np.zeros(output_dim),
                requires_grad=True
            )
        else:
            self.bias = None
        
        # 注册参数
        self.parameters = [self.weights]
        if self.use_bias:
            self.parameters.append(self.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
        
        Returns:
            输出张量 [batch_size, output_dim]
        """
        # y = xW + b
        out = x.matmul(self.weights)
        if self.use_bias:
            out = out + self.bias
        return out
    
    def __repr__(self):
        return f"Dense(input_dim={self.input_dim}, output_dim={self.output_dim}, use_bias={self.use_bias})"


class ReLU(Layer):
    """
    ReLU 激活层
    
    实现: y = max(0, x)
    """
    
    def __init__(self):
        super().__init__()
        self.trainable = False
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            输出张量
        """
        return relu(x)
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid(Layer):
    """
    Sigmoid 激活层
    
    实现: y = 1 / (1 + exp(-x))
    """
    
    def __init__(self):
        super().__init__()
        self.trainable = False
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            输出张量
        """
        return sigmoid(x)
    
    def __repr__(self):
        return "Sigmoid()"


class Softmax(Layer):
    """
    Softmax 激活层
    
    实现: y = exp(x) / sum(exp(x))
    """
    
    def __init__(self, axis=-1):
        """
        初始化 Softmax 层
        
        Args:
            axis: 计算 softmax 的轴
        """
        super().__init__()
        self.trainable = False
        self.axis = axis
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            输出张量
        """
        return softmax(x, axis=self.axis)
    
    def __repr__(self):
        return f"Softmax(axis={self.axis})"


class Dropout(Layer):
    """
    Dropout 层 (可选实现)
    
    训练时随机丢弃部分神经元，测试时不丢弃
    """
    
    def __init__(self, rate=0.5):
        """
        初始化 Dropout 层
        
        Args:
            rate: 丢弃率
        """
        super().__init__()
        self.rate = rate
        self.trainable = False
        self.training = True
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            输出张量
        """
        if self.training and self.rate > 0:
            # 训练模式: 随机丢弃
            mask = np.random.binomial(1, 1 - self.rate, size=x.shape)
            out = Tensor(
                x.data * mask / (1 - self.rate),  # 缩放以保持期望不变
                requires_grad=x.requires_grad,
                _children=(x,)
            )
            
            def _backward():
                if x.requires_grad:
                    x.grad += out.grad * mask / (1 - self.rate)
            
            out._backward = _backward
            return out
        else:
            # 测试模式: 不丢弃
            return x
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False
    
    def __repr__(self):
        return f"Dropout(rate={self.rate})"
