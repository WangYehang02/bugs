import torch
import torch.nn as nn


def compute_residuals(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    AnomalyGFM 风格残差：
      r_i = h_i - mean_{j in N(i)} h_j

    约定 edge_index 为 PyG 格式 [2, E]，消息从 src=edge_index[0] 聚合到 dst=edge_index[1]。
    """
    if h.dim() != 2:
        raise ValueError(f"h should be [N, D], got {tuple(h.shape)}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index should be [2, E], got {tuple(edge_index.shape)}")

    src, dst = edge_index[0], edge_index[1]
    n, d = h.size(0), h.size(1)

    # sum_{j->i} h_j
    neigh_sum = torch.zeros((n, d), device=h.device, dtype=h.dtype)
    neigh_sum.index_add_(0, dst, h[src])

    deg = torch.zeros((n,), device=h.device, dtype=h.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
    deg = deg.clamp_min(1.0).unsqueeze(1)  # 避免除0；孤立点视为均值=0

    neigh_mean = neigh_sum / deg
    return h - neigh_mean


def compute_dual_residuals_with_degree(h: torch.Tensor, edge_index: torch.Tensor):
    """
    计算双重残差（全局+局部）并返回节点度数用于自适应门控。
    全局残差：节点与全图均值的差异，适合稀疏/低度节点（稳健）。
    局部残差：节点与邻居均值的差异，适合稠密图上的局部结构异常（敏感）。
    约定 edge_index 为 PyG 格式 [2, E]，消息从 src=edge_index[0] 聚合到 dst=edge_index[1]。

    返回:
        r_global: [N, D] 全局统计残差
        r_local:  [N, D] 局部结构残差
        deg:      [N, 1] 节点度数（用于后续门控）
    """
    if h.dim() != 2:
        raise ValueError(f"h should be [N, D], got {tuple(h.shape)}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index should be [2, E], got {tuple(edge_index.shape)}")

    # 1. 全局残差 (Global Residual)：节点相对全图均值的偏差
    global_mean = torch.mean(h, dim=0, keepdim=True)
    r_global = h - global_mean

    # 2. 局部残差 (Local Residual) 与度数
    src, dst = edge_index[0], edge_index[1]
    n, d = h.size(0), h.size(1)
    neigh_sum = torch.zeros((n, d), device=h.device, dtype=h.dtype)
    neigh_sum.index_add_(0, dst, h[src])

    deg_val = torch.zeros((n,), device=h.device, dtype=h.dtype)
    deg_val.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))

    deg_clamped = deg_val.clamp_min(1.0).unsqueeze(1)
    neigh_mean = neigh_sum / deg_clamped
    r_local = h - neigh_mean

    return r_global, r_local, deg_val.unsqueeze(1)


def compute_multi_scale_residuals(h: torch.Tensor, edge_index: torch.Tensor):
    """
    多尺度残差：全局、局部、结构（度数异常）。
    r_global: 节点与全图均值的差异（属性/分布异常）
    r_local: 节点与邻居均值的差异（局部属性异常）
    r_structural: 度数异常 |deg_i - mean(deg_neighbors)| 扩展到 [N,D]
    返回: r_global [N,D], r_local [N,D], r_structural [N,D], deg [N,1]
    """
    if h.dim() != 2 or edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("h [N,D], edge_index [2,E]")
    src, dst = edge_index[0], edge_index[1]
    n, d = h.size(0), h.size(1)

    global_mean = torch.mean(h, dim=0, keepdim=True)
    r_global = h - global_mean

    neigh_sum = torch.zeros((n, d), device=h.device, dtype=h.dtype)
    neigh_sum.index_add_(0, dst, h[src])
    deg_val = torch.zeros((n,), device=h.device, dtype=h.dtype)
    deg_val.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
    deg_clamped = deg_val.clamp_min(1.0).unsqueeze(1)
    neigh_mean = neigh_sum / deg_clamped
    r_local = h - neigh_mean

    # 邻居度数均值：mean(deg_neighbors) for each node
    deg_src = deg_val[src]
    neigh_deg_sum = torch.zeros((n,), device=h.device, dtype=h.dtype)
    neigh_deg_sum.index_add_(0, dst, deg_src)
    neigh_deg_mean = (neigh_deg_sum / deg_clamped.squeeze(1)).unsqueeze(1)
    deg_expanded = deg_val.unsqueeze(1)
    r_structural_scalar = (deg_expanded - neigh_deg_mean).abs()
    r_structural = r_structural_scalar.expand(-1, d)

    return r_global, r_local, r_structural, deg_val.unsqueeze(1)


class ResidualChannelAttention(nn.Module):
    """学习多通道残差的融合权重（软注意力）。"""
    def __init__(self, num_channels: int = 3, dim: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.w = nn.Sequential(
            nn.Linear(dim * num_channels, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, num_channels),
            nn.Softmax(dim=1),
        )

    def forward(self, residuals: list) -> torch.Tensor:
        """
        residuals: list of [N, D] tensors, length num_channels
        返回: [N, D] 加权和
        """
        cat = torch.cat(residuals, dim=1)
        alpha = self.w(cat)
        out = sum(alpha[:, i : i + 1] * residuals[i] for i in range(len(residuals)))
        return out


def adaptive_residual_scale(deg: torch.Tensor, base_scale: float = 10.0, low_deg: float = 2.0, high_deg: float = 10.0) -> torch.Tensor:
    """
    基于节点度的自适应残差缩放：低度节点用较小 scale，高度节点用较大 scale。
    deg: [N, 1]
    返回: [N, 1] scale 因子
    """
    d = deg.clamp(min=1e-6)
    log_deg = torch.log1p(d)
    low, high = torch.log1p(torch.tensor([low_deg, high_deg], device=deg.device, dtype=deg.dtype))
    t = (log_deg - low) / (high - low + 1e-8).clamp(0.0, 1.0)
    return base_scale * (0.5 + t)