from typing import Callable, Union
import math
import torch
import torch.nn as nn
import torch.optim

ModuleType = Union[str, Callable[..., nn.Module]]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FlowMatchingLoss:
    """Flow Matching Loss - learns velocity field instead of denoising"""
    def __init__(self, sigma_min=0.01, sigma_max=1.0, hid_dim=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.hid_dim = hid_dim
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def __call__(self, velocity_fn, data, proto=None, proto_alpha=None):
        """
        Flow Matching loss: learn velocity field v_t(x_t) that transports 
        from noise distribution to data distribution
        
        Path: x_t = (1-t) * x_0 + t * x_1
        where x_0 = data, x_1 = noise
        True velocity: v_t = x_1 - x_0 = noise - data
        """
        batch_size = data.shape[0]
        device = data.device
        
        # Sample time uniformly in [0, 1]
        t = torch.rand(batch_size, device=device)
        
        # Sample noise
        noise = torch.randn_like(data)
        
        # Linear interpolation path: x_t = (1-t) * data + t * noise
        t_expanded = t.unsqueeze(1)  # [B, 1]
        x_t = (1 - t_expanded) * data + t_expanded * noise
        
        # True velocity: v_t = noise - data (constant along path)
        v_true = noise - data
        
        # Predicted velocity
        v_pred = velocity_fn(x_t, t, proto, proto_alpha)
        
        # L2 loss
        loss = torch.mean((v_pred - v_true) ** 2)
        
        # Reconstruction error for scoring (similar to diffusion)
        reconstruction_errors = (v_pred - v_true) ** 2
        score = torch.sqrt(torch.sum(reconstruction_errors, 1))
        
        # Reconstructed data point (for prototype computation)
        # Use the predicted velocity to estimate reconstruction
        # At t=0, we can reconstruct as: x_0 ≈ x_t - t * v_pred
        # But for scoring, we use the error directly
        reconstructed = data - v_pred  # Approximate reconstruction
        
        return loss, score, reconstructed


class MLPFlowMatching(nn.Module):
    """MLP for Flow Matching - predicts velocity field (改进版，学习FMGADv1)"""
    def __init__(self, d_in, dim_t=512, cond_dim=None):
        super().__init__()
        self.dim_t = dim_t
        self.cond_dim = cond_dim
        
        # 时间嵌入层
        self.map_time = PositionalEmbedding(num_channels=dim_t)
        
        # 时间嵌入的MLP
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        
        # 如果使用条件（prototype），添加条件投影层
        if cond_dim is not None:
            self.map_proto = nn.Linear(cond_dim, dim_t, bias=False)
            self.proto_proj = nn.Linear(cond_dim, dim_t)
        
        # 输入投影：将输入特征投影到时间嵌入维度
        self.proj_x = nn.Linear(d_in, dim_t)
        
        # 主要的MLP网络：预测速度向量
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),  # 输出维度与输入维度相同
        )

    def forward(self, x, t, context=None, proto_alpha=None):
        """
        Predict velocity field v_t(x_t) (改进版)
        Args:
            x: current state [B, d]
            t: time [B] or scalar
            context: prototype context (optional) [B, cond_dim] or [1, cond_dim]
            proto_alpha: prototype weight (optional)
        Returns:
            velocity: predicted velocity [B, d]
        """
        # 处理时间维度
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=x.dtype).expand(x.shape[0])
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.dim() > 1:
            t = t.squeeze(-1) if t.shape[-1] == 1 else t.flatten()
        
        # 确保t在[0, 1]范围内
        t = t.clamp(0.0, 1.0)
        
        # 时间嵌入（放大时间以便嵌入更好地工作，FMGADv1的关键改进）
        t_emb = self.map_time(t * 1000.0)
        t_emb = t_emb.reshape(t_emb.shape[0], 2, -1).flip(1).reshape(*t_emb.shape)
        t_emb = self.time_embed(t_emb)
        
        # 输入投影
        x_proj = self.proj_x(x)
        
        # 如果有条件（prototype），添加条件信息
        if context is not None and self.cond_dim is not None:
            if proto_alpha is None:
                proto_alpha = 1.0
            proto_emb = self.proto_proj(context)
            # 融合：输入特征 + 时间嵌入 + 条件嵌入
            h = x_proj + t_emb + proto_alpha * proto_emb
        else:
            # 无条件的融合：输入特征 + 时间嵌入
            h = x_proj + t_emb
        
        # 通过MLP预测速度
        v_t = self.mlp(h)
        return v_t


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2,
                             dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FlowMatchingModel(nn.Module):
    """Flow Matching Model wrapper (改进版，使用FMloss.py中的损失函数)"""
    def __init__(self, velocity_fn, hid_dim, sigma_min=0.01, sigma_max=1.0):
        super().__init__()
        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.velocity_fn = velocity_fn

    def forward(self, x_1, proto=None, proto_alpha=None):
        """
        前向传播：计算流匹配损失（改进版，使用更好的损失函数）
        
        Args:
            x_1: 真实节点特征（目标数据）[batch, hid_dim]
            proto: prototype（上下文条件），可选 [batch, hid_dim] 或 [1, hid_dim]
            proto_alpha: prototype的权重系数
        
        Returns:
            loss: 损失值（标量）
            score: 异常分数（用于评估）[batch]
            reconstructed: 重构的特征（通过流匹配采样得到）[batch, hid_dim]
        """
        from FMloss import flow_matching_loss, conditional_flow_matching_loss
        
        # 使用流匹配损失函数
        # 如果提供了proto，使用条件流匹配，否则使用无条件流匹配
        if proto is not None:
            graph_context = proto
            # 使用logit_normal采样策略（对边界附近的时间步给予更多关注，提升性能）
            loss = conditional_flow_matching_loss(
                self.velocity_fn, 
                x_1, 
                graph_context, 
                t_sampling='logit_normal',  # 关键改进：使用logit_normal而不是uniform
                reduction='mean'
            )
        else:
            # 如果没有proto，使用零向量作为context
            graph_context = torch.zeros(1, x_1.shape[1], device=x_1.device)
            loss = flow_matching_loss(
                self.velocity_fn,
                x_1,
                graph_context,
                reduction='mean'
            )
        
        # 为了兼容原有的接口，计算重构误差作为异常分数
        # 使用采样得到重构值（更准确，FMGADv1的方法）
        with torch.no_grad():
            # 从噪声开始采样
            x_0 = torch.randn_like(x_1)
            # 使用采样得到重构值（更准确）
            reconstructed = sample_flow_matching(
                self.velocity_fn, x_0, num_steps=10,
                proto=proto, proto_alpha=proto_alpha
            )
        
        # 计算重构误差作为异常分数
        score = torch.sqrt(torch.sum((x_1 - reconstructed) ** 2, dim=1))
        
        return loss, score, reconstructed


def sample_flow_matching(velocity_net, x_0, num_steps=50, proto=None, proto_alpha=None):
    """
    Sample from flow matching model using ODE solver (Euler method) (改进版)
    从噪声x_0开始，通过ODE积分得到x_1（正向积分，FMGADv1的方法）
    
    Args:
        velocity_net: 速度场网络
        x_0: 初始噪声 [batch, dim]
        num_steps: 采样步数
        proto: prototype条件，可选
        proto_alpha: prototype权重
    Returns:
        x_1: 生成的数据 [batch, dim]
    """
    device = x_0.device
    batch_size = x_0.shape[0]
    
    # 时间步：从0到1（正向积分，更自然）
    dt = 1.0 / num_steps
    x_t = x_0.clone()
    
    velocity_net.eval()
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device, dtype=x_0.dtype)
            
            # 预测速度
            v_t = velocity_net(x_t, t, context=proto, proto_alpha=proto_alpha)
            
            # Euler方法：x_{t+dt} = x_t + dt * v_t（正向）
            x_t = x_t + dt * v_t
    
    return x_t


def sample_flow_matching_free(proto_net, free_net, x_0, num_steps=50, proto=None, proto_alpha=None, weight=None):
    """
    Sample using combination of prototype and free flow matching models (改进版)
    从噪声x_0开始，正向积分到数据（FMGADv1的方法）
    
    Args:
        proto_net: 条件速度场网络（使用prototype）
        free_net: 自由速度场网络（不使用prototype）
        x_0: 初始噪声 [batch, dim]
        num_steps: 采样步数
        proto: prototype条件
        proto_alpha: prototype权重
        weight: 组合权重，weight越大，prototype的影响越大
    Returns:
        x_1: 生成的数据 [batch, dim]
    """
    device = x_0.device
    batch_size = x_0.shape[0]
    
    # 时间步：从0到1（正向积分）
    dt = 1.0 / num_steps
    x_t = x_0.clone()
    
    proto_net.eval()
    free_net.eval()
    
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device, dtype=x_0.dtype)
            
            # 预测两个速度场
            v_proto = proto_net(x_t, t, context=proto, proto_alpha=proto_alpha)
            v_free = free_net(x_t, t, context=None, proto_alpha=None)
            
            # 组合速度场：weight越大，proto的影响越大
            v_combined = (1.0 + weight) * v_free - weight * v_proto
            
            # Euler方法（正向）
            x_t = x_t + dt * v_combined
    
    return x_t
