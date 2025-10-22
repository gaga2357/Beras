"""
Losses 模块 - 实现损失函数
"""

import numpy as np
from .tensor import Tensor, log


class Loss:
    """
    损失函数基类
    """
    
    def __call__(self, y_pred, y_true):
        """
        计算损失
        
        Args:
            y_pred: 预测值
            y_true: 真实值
        
        Returns:
            损失值(标量张量)
        """
        raise NotImplementedError("Subclass must implement __call__ method")


class MSELoss(Loss):
    """
    均方误差损失 (Mean Squared Error)
    
    公式: MSE = mean((y_pred - y_true)^2)
    
    适用于回归任务
    """
    
    def __call__(self, y_pred, y_true):
        """
        计算 MSE 损失
        
        Args:
            y_pred: 预测值张量
            y_true: 真实值张量或 numpy array
        
        Returns:
            损失值(标量张量)
        """
        # 确保 y_true 是 Tensor
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # MSE = mean((y_pred - y_true)^2)
        diff = y_pred - y_true
        squared_diff = diff * diff
        loss = squared_diff.mean()
        
        return loss
    
    def __repr__(self):
        return "MSELoss()"


class CrossEntropyLoss(Loss):
    """
    交叉熵损失 (Cross Entropy Loss)
    
    公式: CE = -mean(sum(y_true * log(y_pred)))
    
    适用于分类任务
    注意: 输入应该是 softmax 之后的概率分布
    """
    
    def __call__(self, y_pred, y_true):
        """
        计算交叉熵损失
        
        Args:
            y_pred: 预测概率分布 [batch_size, num_classes]
            y_true: 真实标签(one-hot 编码) [batch_size, num_classes]
        
        Returns:
            损失值(标量张量)
        """
        # 确保 y_true 是 Tensor
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # 数值稳定性: 裁剪预测值
        y_pred_clipped = Tensor(
            np.clip(y_pred.data, 1e-8, 1 - 1e-8),
            requires_grad=y_pred.requires_grad,
            _children=(y_pred,)
        )
        
        def _backward():
            if y_pred.requires_grad:
                y_pred.grad += y_pred_clipped.grad
        
        y_pred_clipped._backward = _backward
        
        # CE = -mean(sum(y_true * log(y_pred)))
        log_pred = log(y_pred_clipped)
        loss = -(y_true * log_pred).sum(axis=1).mean()
        
        return loss
    
    def __repr__(self):
        return "CrossEntropyLoss()"


class BinaryCrossEntropyLoss(Loss):
    """
    二分类交叉熵损失
    
    公式: BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    
    适用于二分类任务
    """
    
    def __call__(self, y_pred, y_true):
        """
        计算二分类交叉熵损失
        
        Args:
            y_pred: 预测概率 [batch_size, 1] 或 [batch_size]
            y_true: 真实标签 [batch_size, 1] 或 [batch_size]
        
        Returns:
            损失值(标量张量)
        """
        # 确保 y_true 是 Tensor
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        # 数值稳定性: 裁剪预测值
        y_pred_clipped = Tensor(
            np.clip(y_pred.data, 1e-8, 1 - 1e-8),
            requires_grad=y_pred.requires_grad,
            _children=(y_pred,)
        )
        
        def _backward():
            if y_pred.requires_grad:
                y_pred.grad += y_pred_clipped.grad
        
        y_pred_clipped._backward = _backward
        
        # BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        log_pred = log(y_pred_clipped)
        log_one_minus_pred = log(Tensor(1.0) - y_pred_clipped)
        loss = -(y_true * log_pred + (Tensor(1.0) - y_true) * log_one_minus_pred).mean()
        
        return loss
    
    def __repr__(self):
        return "BinaryCrossEntropyLoss()"
