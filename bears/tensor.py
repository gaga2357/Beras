"""
Tensor 模块 - 实现张量和自动微分功能
"""

import numpy as np


class Tensor:
    """
    张量类，封装数据和梯度，支持自动微分
    
    Attributes:
        data: 实际数据 (numpy array)
        grad: 梯度 (numpy array)
        requires_grad: 是否需要计算梯度
        _backward: 反向传播函数
        _prev: 前驱节点集合
    """
    
    def __init__(self, data, requires_grad=False, _children=()):
        """
        初始化张量
        
        Args:
            data: 数据，可以是 list, numpy array 等
            requires_grad: 是否需要梯度
            _children: 子节点(用于构建计算图)
        """
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, (int, float, np.number)):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float32) if requires_grad else None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
    
    @property
    def shape(self):
        """返回张量形状"""
        return self.data.shape
    
    def zero_grad(self):
        """梯度清零"""
        if self.grad is not None:
            self.grad = np.zeros_like(self.data, dtype=np.float32)
    
    def backward(self, grad=None):
        """
        反向传播，计算梯度
        
        Args:
            grad: 上游梯度，默认为 1
        """
        if not self.requires_grad:
            return
        
        # 构建拓扑排序
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        
        build_topo(self)
        
        # 初始化梯度
        if grad is None:
            self.grad = np.ones_like(self.data, dtype=np.float32)
        else:
            self.grad = grad
        
        # 反向传播
        for node in reversed(topo):
            node._backward()
    
    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        """加法运算"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other)
        )
        
        def _backward():
            if self.requires_grad:
                # 处理广播情况
                grad = out.grad
                # 如果形状不同，需要对梯度求和到正确的形状
                ndims_added = grad.ndim - self.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                # 对于广播的维度求和
                for i, (dim, size) in enumerate(zip(grad.shape, self.data.shape)):
                    if size == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad
            
            if other.requires_grad:
                grad = out.grad
                ndims_added = grad.ndim - other.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, (dim, size) in enumerate(zip(grad.shape, other.data.shape)):
                    if size == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """乘法运算"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other)
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data
                ndims_added = grad.ndim - self.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, (dim, size) in enumerate(zip(grad.shape, self.data.shape)):
                    if size == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad
            
            if other.requires_grad:
                grad = out.grad * self.data
                ndims_added = grad.ndim - other.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, (dim, size) in enumerate(zip(grad.shape, other.data.shape)):
                    if size == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
        
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        """减法运算"""
        return self + (-other)
    
    def __neg__(self):
        """取负运算"""
        return self * -1
    
    def __truediv__(self, other):
        """除法运算"""
        return self * (other ** -1)
    
    def __pow__(self, power):
        """幂运算"""
        assert isinstance(power, (int, float)), "Power must be int or float"
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                self.grad += (power * self.data ** (power - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return Tensor(other) - self
    
    def __rtruediv__(self, other):
        return Tensor(other) / self
    
    def matmul(self, other):
        """
        矩阵乘法
        
        Args:
            other: 另一个张量
        
        Returns:
            结果张量
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other)
        )
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        """
        求和运算
        
        Args:
            axis: 求和的轴
            keepdims: 是否保持维度
        
        Returns:
            结果张量
        """
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    if not keepdims:
                        grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.data.shape)
                self.grad += grad
        
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        """
        求平均值
        
        Args:
            axis: 求平均的轴
            keepdims: 是否保持维度
        
        Returns:
            结果张量
        """
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    if not keepdims:
                        grad = np.expand_dims(grad, axis=axis)
                    n = self.data.shape[axis]
                else:
                    n = self.data.size
                grad = np.broadcast_to(grad, self.data.shape) / n
                self.grad += grad
        
        out._backward = _backward
        return out
    
    def reshape(self, *shape):
        """
        改变形状
        
        Args:
            shape: 新的形状
        
        Returns:
            结果张量
        """
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        
        out._backward = _backward
        return out
    
    def transpose(self, *axes):
        """
        转置
        
        Args:
            axes: 轴的顺序
        
        Returns:
            结果张量
        """
        if len(axes) == 0:
            axes = None
        out = Tensor(
            self.data.transpose(*axes) if axes else self.data.T,
            requires_grad=self.requires_grad,
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                if axes:
                    # 计算逆转置
                    inv_axes = np.argsort(axes)
                    self.grad += out.grad.transpose(*inv_axes)
                else:
                    self.grad += out.grad.T
        
        out._backward = _backward
        return out
    
    @property
    def T(self):
        """转置属性"""
        return self.transpose()


def relu(x):
    """
    ReLU 激活函数
    
    Args:
        x: 输入张量
    
    Returns:
        输出张量
    """
    out = Tensor(
        np.maximum(0, x.data),
        requires_grad=x.requires_grad,
        _children=(x,)
    )
    
    def _backward():
        if x.requires_grad:
            x.grad += (x.data > 0).astype(np.float32) * out.grad
    
    out._backward = _backward
    return out


def sigmoid(x):
    """
    Sigmoid 激活函数
    
    Args:
        x: 输入张量
    
    Returns:
        输出张量
    """
    sig = 1 / (1 + np.exp(-x.data))
    out = Tensor(
        sig,
        requires_grad=x.requires_grad,
        _children=(x,)
    )
    
    def _backward():
        if x.requires_grad:
            x.grad += sig * (1 - sig) * out.grad
    
    out._backward = _backward
    return out


def softmax(x, axis=-1):
    """
    Softmax 激活函数
    
    Args:
        x: 输入张量
        axis: 计算 softmax 的轴
    
    Returns:
        输出张量
    """
    # 数值稳定性: 减去最大值
    x_max = np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(x.data - x_max)
    softmax_out = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    out = Tensor(
        softmax_out,
        requires_grad=x.requires_grad,
        _children=(x,)
    )
    
    def _backward():
        if x.requires_grad:
            # Softmax 的梯度: s * (grad - (s * grad).sum())
            s = out.data
            grad_sum = (s * out.grad).sum(axis=axis, keepdims=True)
            x.grad += s * (out.grad - grad_sum)
    
    out._backward = _backward
    return out


def log(x):
    """
    自然对数
    
    Args:
        x: 输入张量
    
    Returns:
        输出张量
    """
    out = Tensor(
        np.log(x.data + 1e-8),  # 添加小常数防止 log(0)
        requires_grad=x.requires_grad,
        _children=(x,)
    )
    
    def _backward():
        if x.requires_grad:
            x.grad += out.grad / (x.data + 1e-8)
    
    out._backward = _backward
    return out


def exp(x):
    """
    指数函数
    
    Args:
        x: 输入张量
    
    Returns:
        输出张量
    """
    out = Tensor(
        np.exp(x.data),
        requires_grad=x.requires_grad,
        _children=(x,)
    )
    
    def _backward():
        if x.requires_grad:
            x.grad += out.data * out.grad
    
    out._backward = _backward
    return out
