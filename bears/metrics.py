"""
Metrics 模块 - 实现评估指标
"""

import numpy as np
from .tensor import Tensor


def accuracy(y_pred, y_true):
    """
    计算分类准确率
    
    Args:
        y_pred: 预测值 [batch_size, num_classes] (概率分布或 logits)
        y_true: 真实标签 [batch_size, num_classes] (one-hot 编码)
    
    Returns:
        准确率 (float)
    """
    # 转换为 numpy array
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    
    # 获取预测类别
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    
    # 计算准确率
    acc = np.mean(pred_labels == true_labels)
    
    return float(acc)


def mse_metric(y_pred, y_true):
    """
    计算均方误差(用于评估)
    
    Args:
        y_pred: 预测值
        y_true: 真实值
    
    Returns:
        MSE 值 (float)
    """
    # 转换为 numpy array
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    
    # 计算 MSE
    mse = np.mean((y_pred - y_true) ** 2)
    
    return float(mse)


def mae_metric(y_pred, y_true):
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    Args:
        y_pred: 预测值
        y_true: 真实值
    
    Returns:
        MAE 值 (float)
    """
    # 转换为 numpy array
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    
    # 计算 MAE
    mae = np.mean(np.abs(y_pred - y_true))
    
    return float(mae)


def r2_score(y_pred, y_true):
    """
    计算 R² 分数 (决定系数)
    
    Args:
        y_pred: 预测值
        y_true: 真实值
    
    Returns:
        R² 分数 (float)
    """
    # 转换为 numpy array
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    
    # 计算 R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return float(r2)


def precision(y_pred, y_true, class_idx=None):
    """
    计算精确率
    
    Args:
        y_pred: 预测值 [batch_size, num_classes]
        y_true: 真实标签 [batch_size, num_classes]
        class_idx: 类别索引，None 表示计算所有类别的平均值
    
    Returns:
        精确率 (float)
    """
    # 转换为 numpy array
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    
    if class_idx is not None:
        # 计算特定类别的精确率
        tp = np.sum((pred_labels == class_idx) & (true_labels == class_idx))
        fp = np.sum((pred_labels == class_idx) & (true_labels != class_idx))
        prec = tp / (tp + fp + 1e-8)
    else:
        # 计算所有类别的平均精确率
        num_classes = y_pred.shape[1]
        prec_list = []
        for i in range(num_classes):
            tp = np.sum((pred_labels == i) & (true_labels == i))
            fp = np.sum((pred_labels == i) & (true_labels != i))
            prec_list.append(tp / (tp + fp + 1e-8))
        prec = np.mean(prec_list)
    
    return float(prec)


def recall(y_pred, y_true, class_idx=None):
    """
    计算召回率
    
    Args:
        y_pred: 预测值 [batch_size, num_classes]
        y_true: 真实标签 [batch_size, num_classes]
        class_idx: 类别索引，None 表示计算所有类别的平均值
    
    Returns:
        召回率 (float)
    """
    # 转换为 numpy array
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    
    if class_idx is not None:
        # 计算特定类别的召回率
        tp = np.sum((pred_labels == class_idx) & (true_labels == class_idx))
        fn = np.sum((pred_labels != class_idx) & (true_labels == class_idx))
        rec = tp / (tp + fn + 1e-8)
    else:
        # 计算所有类别的平均召回率
        num_classes = y_pred.shape[1]
        rec_list = []
        for i in range(num_classes):
            tp = np.sum((pred_labels == i) & (true_labels == i))
            fn = np.sum((pred_labels != i) & (true_labels == i))
            rec_list.append(tp / (tp + fn + 1e-8))
        rec = np.mean(rec_list)
    
    return float(rec)


def f1_score(y_pred, y_true, class_idx=None):
    """
    计算 F1 分数
    
    Args:
        y_pred: 预测值 [batch_size, num_classes]
        y_true: 真实标签 [batch_size, num_classes]
        class_idx: 类别索引，None 表示计算所有类别的平均值
    
    Returns:
        F1 分数 (float)
    """
    prec = precision(y_pred, y_true, class_idx)
    rec = recall(y_pred, y_true, class_idx)
    
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    
    return float(f1)
