import torch
import torch.nn.functional as F
import numpy as np

def flow_matching_loss(model, x_1, graph_context, reduction='mean', weight=None):
    """
    优化的流匹配损失函数
    
    Args:
        model: 速度场网络，输入 (x_t, t, context)，输出速度 v
        x_1: 真实节点特征 (Batch, Dim)
        graph_context: 由 GNN 提取的邻域结构信息 (Batch, Dim) 或 (1, Dim) 用于广播
        reduction: 'mean' 或 'none'，用于返回逐样本损失
        weight: 可选的样本权重 (Batch,)
    
    Returns:
        loss: 标量损失（reduction='mean'）或逐样本损失（reduction='none'）
    """
    batch_size = x_1.shape[0]
    
    # 1. 采样时间 t ~ U[0, 1]
    t = torch.rand(batch_size, 1, device=x_1.device)
    
    # 2. 采样高斯噪声 x_0 (先验分布)
    x_0 = torch.randn_like(x_1)
    
    # 3. 构造线性插值路径 x_t
    # 路径变直的关键：x_t = (1-t)*x_0 + t*x_1
    x_t = (1 - t) * x_0 + t * x_1
    
    # 4. 定义目标速度 (Target Velocity)
    # 直线路径的导数永远是 x_1 - x_0
    target_v = x_1 - x_0
    
    # 5. 处理context的广播：如果context是(1, Dim)，需要广播到batch
    if graph_context.dim() == 2:
        if graph_context.shape[0] == 1 and batch_size > 1:
            graph_context = graph_context.repeat(batch_size, 1)
        elif graph_context.shape[0] != batch_size:
            raise ValueError(f"Context shape mismatch: {graph_context.shape[0]} vs batch_size {batch_size}")
    
    # 6. 模型预测速度 v_theta
    # 输入包含当前的 x_t, 时间 t, 以及通过子图获取的结构上下文
    pred_v = model(x_t, t.squeeze(-1) if t.dim() > 1 else t, graph_context)
    
    # 7. 计算逐样本的MSE损失
    mse_per_sample = F.mse_loss(pred_v, target_v, reduction='none').mean(dim=1)
    
    # 8. 应用权重（如果提供）
    if weight is not None:
        mse_per_sample = mse_per_sample * weight
    
    # 9. 根据reduction返回结果
    if reduction == 'mean':
        return mse_per_sample.mean()
    elif reduction == 'none':
        return mse_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def conditional_flow_matching_loss(model, x_1, graph_context, t_sampling='uniform', 
                                    reduction='mean', weight=None):
    """
    增强的条件流匹配损失，支持不同的时间采样策略
    
    Args:
        model: 速度场网络
        x_1: 真实节点特征 (Batch, Dim)
        graph_context: 结构上下文 (Batch, Dim) 或 (1, Dim)
        t_sampling: 'uniform' 或 'logit_normal'，时间采样策略
        reduction: 'mean' 或 'none'
        weight: 可选的样本权重
    
    Returns:
        loss: 损失值
    """
    batch_size = x_1.shape[0]
    
    # 时间采样策略
    if t_sampling == 'uniform':
        t = torch.rand(batch_size, 1, device=x_1.device)
    elif t_sampling == 'logit_normal':
        # 对边界附近的时间步给予更多关注（提升性能的关键）
        t_normal = torch.randn(batch_size, 1, device=x_1.device)
        t = torch.sigmoid(t_normal * 2.0)  # 缩放以控制分布
    else:
        raise ValueError(f"Unknown t_sampling: {t_sampling}")
    
    # 采样噪声
    x_0 = torch.randn_like(x_1)
    
    # 线性插值路径
    x_t = (1 - t) * x_0 + t * x_1
    
    # 目标速度
    target_v = x_1 - x_0
    
    # 处理context广播
    if graph_context.dim() == 2:
        if graph_context.shape[0] == 1 and batch_size > 1:
            graph_context = graph_context.repeat(batch_size, 1)
    
    # 预测速度
    t_input = t.squeeze(-1) if t.dim() > 1 else t
    pred_v = model(x_t, t_input, graph_context)
    
    # 计算损失
    mse_per_sample = F.mse_loss(pred_v, target_v, reduction='none').mean(dim=1)
    
    if weight is not None:
        mse_per_sample = mse_per_sample * weight
    
    if reduction == 'mean':
        return mse_per_sample.mean()
    elif reduction == 'none':
        return mse_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
