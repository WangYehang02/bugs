"""
工具函数
"""
import torch
import torch.nn.functional as F


def softmax_with_temperature(input, t=1, axis=-1):
    """
    带温度参数的 softmax 函数
    
    参数:
        input: 输入张量
        t: 温度参数，t > 1 时分布更平滑，t < 1 时分布更尖锐
        axis: 应用 softmax 的维度
    
    返回:
        归一化后的概率分布
    """
    return F.softmax(input / t, dim=axis)
