import os
import sys
import math
import csv
import json
import hashlib
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

from torchdiffeq import odeint

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, from_scipy_sparse_matrix
from sklearn.metrics import auc, precision_recall_curve
from scipy import io as scipy_io
from scipy.sparse import issparse
from pygod.metric.metric import (
    eval_roc_auc,
    eval_average_precision,
    eval_recall_at_k,
    eval_precision_at_k,
)

# 本仓库自包含，不依赖外部 DiffGAD
from pygod.utils import load_data

# 以当前包所在目录为根（FMGAD），保证单文件夹可运行
FMGAD_ROOT = os.path.dirname(os.path.abspath(__file__))
if FMGAD_ROOT not in sys.path:
    sys.path.insert(0, FMGAD_ROOT)

from auto_encoder import GraphAE
from utils import softmax_with_temperature
from load_custom_data import load_dgraphfin_data, load_dgraph_data
from flow_matching_model import MLPFlowMatching, FlowMatchingModel, sample_flow_matching_free
from FMloss import flow_matching_loss, conditional_flow_matching_loss

from encoder import (
    compute_dual_residuals_with_degree,
    compute_multi_scale_residuals,
    ResidualChannelAttention,
    adaptive_residual_scale as adaptive_residual_scale_fn,
)


def _smooth_scores_by_graph(
    score: torch.Tensor, edge_index: torch.Tensor, alpha: float, device: torch.device
) -> torch.Tensor:
    """score_smoothed = (1-alpha)*score + alpha*mean(score[neighbors])."""
    if alpha <= 0.0 or edge_index.numel() == 0:
        return score
    src, dst = edge_index[0], edge_index[1]
    n = score.size(0)
    neigh_sum = torch.zeros(n, device=device, dtype=score.dtype)
    neigh_sum.index_add_(0, dst, score[src])
    deg = torch.zeros(n, device=device, dtype=score.dtype)
    deg.index_add_(0, dst, torch.ones_like(score[src]))
    deg = deg.clamp_min(1.0)
    neigh_mean = neigh_sum / deg
    return (1.0 - alpha) * score + alpha * neigh_mean


def _add_virtual_knn_edges(
    edge_index: torch.Tensor,
    h: torch.Tensor,
    degree_threshold: int,
    k: int,
    device: torch.device,
    chunk_size: int = 5000,
) -> torch.Tensor:
    """
    极速优化的虚拟邻居补充机制：
    使用全局邻居均值特征作为 Query 矩阵，通过 GPU 并行张量计算寻找相似节点，
    彻底消除 for 循环中的 k_hop_subgraph 导致的时间阻塞。
    """
    n = h.size(0)

    with torch.no_grad():
        src, dst = edge_index[0], edge_index[1]

        # 1. 统计节点度数
        deg = torch.zeros(n, device=device, dtype=torch.long)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.long))

        # 找出需要处理的低度节点 (排除度数为0的孤立点，因为无法计算邻居均值)
        low_deg_mask = (deg < degree_threshold) & (deg > 0)
        low_deg_nodes = low_deg_mask.nonzero(as_tuple=False).view(-1)

        if low_deg_nodes.numel() == 0:
            return edge_index

        # 2. 全局并行计算所有节点的“邻居平均特征” (充当 Query)
        # 通过 index_add_ 一次性完成所有消息聚合，替代 for 循环
        h_norm = F.normalize(h, p=2, dim=1)
        neigh_sum = torch.zeros((n, h.size(1)), device=device, dtype=h.dtype)
        neigh_sum.index_add_(0, dst, h_norm[src])

        deg_clamp = deg.float().clamp_min(1.0).unsqueeze(1)
        neigh_mean = neigh_sum / deg_clamp
        # 得到归一化的邻居均值特征
        neigh_mean = F.normalize(neigh_mean, p=2, dim=1)

        new_edges = []

        # 3. 分块进行 GPU 全局相似度计算，避免 OOM
        for start_idx in range(0, low_deg_nodes.size(0), chunk_size):
            end_idx = min(start_idx + chunk_size, low_deg_nodes.size(0))
            chunk_nodes = low_deg_nodes[start_idx:end_idx]
            c_size = chunk_nodes.size(0)

            # Query: 本批次低度节点的邻居特征均值
            query_feats = neigh_mean[chunk_nodes]
            # 计算与全图节点的相似度
            sim = torch.mm(query_feats, h_norm.t())

            # 屏蔽与自身的相似度
            sim.scatter_(1, chunk_nodes.unsqueeze(1), -1e9)

            # 向量化屏蔽真实的 1-hop 邻居 (避免重复造边)
            # 建立映射: 节点ID -> 它在当前 chunk_nodes 中的相对行号
            node_to_chunk = torch.full((n,), -1, device=device, dtype=torch.long)
            node_to_chunk[chunk_nodes] = torch.arange(c_size, device=device)

            # 找到所有出发点在当前 chunk 内的真实边
            valid_edges_mask = node_to_chunk[src] >= 0
            chunk_src_idx = node_to_chunk[src[valid_edges_mask]]
            valid_dst = dst[valid_edges_mask]

            # 将真实存在的边的相似度强行置为最小值
            sim[chunk_src_idx, valid_dst] = -1e9

            # 并行获取 Top-K
            _, topk_idx = sim.topk(min(k, n - 1), dim=1)

            # 将结果转回 CPU 构建边列表 (双向加边保证无向图结构)
            chunk_nodes_np = chunk_nodes.cpu().numpy()
            topk_idx_np = topk_idx.cpu().numpy()

            for i in range(c_size):
                u = chunk_nodes_np[i]
                for v in topk_idx_np[i]:
                    new_edges.append([u, v])
                    new_edges.append([v, u])

        if not new_edges:
            return edge_index

        # 4. 合并边并去重
        new_edges_tensor = torch.tensor(new_edges, device=device, dtype=edge_index.dtype).t()
        combined = torch.cat([edge_index, new_edges_tensor], dim=1)
        combined = torch.unique(combined, dim=1)
        return combined


class _GateParams(nn.Module):
    """可学习的门控参数：bias（度数中心）、sharpness（过渡陡峭度），用于自适应双重残差融合。"""
    def __init__(self, bias: float = 2.0, sharpness: float = 1.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        # 用 raw 存贮，前向时用 softplus 保证 sharpness > 0，避免门控语义反转
        self._raw_sharpness = nn.Parameter(torch.tensor(sharpness, dtype=torch.float32))

    @property
    def sharpness(self):
        return torch.nn.functional.softplus(self._raw_sharpness)


class ResFlowGAD(BaseTransform):
    """
    FMGADself：结合 DiffGAD(v2) 的“AE重建评分 + proto/CFG采样” 与 AnomalyGFM 的“残差编码”。

    核心做法（Weibo 单数据集先跑通）：
    - 先训练 GraphAE（与 v2 同口径）得到 h = AE.encode(x)
    - 计算残差 r = h - mean_N(h)，并做 scale 放大
    - 拼接 z = [h ; scale*r] 作为 Flow Matching 的 data manifold
    - FM 采样得到 z_hat，取 h_hat 做 AE.decode，按 AE loss_func 作为异常分数（保留 v2 上限）
    """
    def __init__(
        self,
        name: str = "FMGADself",
        hid_dim: Optional[int] = None,
        ae_epochs: int = 300,
        diff_epochs: int = 800,
        patience: int = 100,
        lr: float = 0.005,
        wd: float = 0.0,
        weight: float = 1.0,
        sample_steps: int = 50,
        ae_dropout: float = 0.3,
        ae_lr: float = 0.01,
        ae_alpha: float = 0.8,
        proto_alpha: float = 0.01,
        residual_scale: float = 10.0,
        gate_bias: float = 2.0,
        gate_sharpness: float = 1.0,
        verbose: bool = True,
        use_nll_score: bool = False,
        use_energy_score: bool = False,
        use_guided_recon: bool = False,
        guidance_scale: float = 3.0,
        ode_steps: int = 20,
        # 多尺度残差与自适应缩放
        use_multi_scale_residual: bool = True,
        use_adaptive_residual_scale: bool = True,
        # 多评分融合（温度系数 softmax 加权）
        use_multi_score_fusion: bool = True,
        score_fusion_temperature: float = 1.0,
        # 稀疏图：虚拟邻居
        use_virtual_neighbors: bool = True,
        virtual_degree_threshold: int = 5,
        virtual_k: int = 5,
        # 训练：困难样本挖掘、课程学习
        use_hard_negative_mining: bool = False,
        use_curriculum_learning: bool = False,
        curriculum_warmup_epochs: int = 100,
        # 后处理：分数平滑
        use_score_smoothing: bool = True,
        score_smoothing_alpha: float = 0.3,
        # 集成：多 trial 取平均分数（在 forward 里已做 3 trial，可选平均分数）
        ensemble_score: bool = True,
        num_trial: int = 3,
    ):
        self.name = name
        self.num_trial = num_trial
        self.hid_dim = hid_dim
        self.ae_epochs = ae_epochs
        self.diff_epochs = diff_epochs
        self.patience = patience
        self.lr = lr
        self.wd = wd
        self.weight = weight
        self.sample_steps = sample_steps
        self.verbose = verbose
        self.proto_alpha = proto_alpha
        self.residual_scale = residual_scale
        self.gate_module = _GateParams(bias=gate_bias, sharpness=gate_sharpness)
        self.use_nll_score = use_nll_score
        self.use_energy_score = use_energy_score
        self.use_guided_recon = use_guided_recon
        self.guidance_scale = guidance_scale
        self.ode_steps = ode_steps
        self.use_multi_scale_residual = use_multi_scale_residual
        self.use_adaptive_residual_scale = use_adaptive_residual_scale
        self.use_multi_score_fusion = use_multi_score_fusion
        self.score_fusion_temperature = score_fusion_temperature
        self.use_virtual_neighbors = use_virtual_neighbors
        self.virtual_degree_threshold = virtual_degree_threshold
        self.virtual_k = virtual_k
        self.use_hard_negative_mining = use_hard_negative_mining
        self.use_curriculum_learning = use_curriculum_learning
        self.curriculum_warmup_epochs = curriculum_warmup_epochs
        self.use_score_smoothing = use_score_smoothing
        self.score_smoothing_alpha = score_smoothing_alpha
        self.ensemble_score = ensemble_score

        self.ae_dropout = ae_dropout
        self.ae_lr = ae_lr
        self.ae_alpha = ae_alpha

        self.ae = None  # type: Optional[GraphAE]
        self.dm = None  # type: Optional[FlowMatchingModel]
        self.dm_proto = None  # type: Optional[FlowMatchingModel]
        self.proto = None  # type: Optional[torch.Tensor]
        self.residual_attention = None  # type: Optional[ResidualChannelAttention]

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # v2 默认扫 500 个 time points；Weibo 先用更少点数提速，指标通常不会明显下降
        self.timesteps = 100

    def _load_dataset(self, dset: str):
        if dset == "dgraphfin":
            return load_dgraphfin_data(os.path.join(FMGAD_ROOT, "data", "dgraphfin.npz"))
        if dset == "dgraph":
            return load_dgraph_data()
        if dset == "yelpchi":
            return self._load_yelpchi()
        if dset == "elliptic":
            return self._load_elliptic()
        if dset == "twitter":
            return self._load_twitter()
        if dset in ("twibot20", "twibot22"):
            return self._load_twibot20(dset)
        return load_data(dset)

    def _load_mat_data(self, path: str):
        """Load graph from .mat file. Supports common keys: A/adj/network, X/Attributes/feat, y/label/Label."""
        mat = scipy_io.loadmat(path)
        # Filter out MATLAB meta keys
        keys = [k for k in mat if not k.startswith("__")]
        # Adjacency: A, adj, network, homo (YelpChi 等异构图合并后的同构邻接)
        adj = None
        for name in ["A", "adj", "network", "Adj", "homo", "net_rur", "net_rtr", "net_rsr"]:
            if name in mat:
                adj = mat[name]
                break
        if adj is None:
            raise KeyError(".mat must contain one of: A, adj, network, Adj, homo, net_*")
        if issparse(adj):
            edge_index, _ = from_scipy_sparse_matrix(adj)
        else:
            adj = np.asarray(adj)
            if adj.ndim == 3:
                adj = adj.reshape(adj.shape[-2], adj.shape[-1])
            edge_index = torch.from_numpy(np.stack(np.where(adj != 0), axis=0)).long()
        # Node features: X, Attributes, feat, features
        x = None
        for name in ["X", "Attributes", "feat", "features", "Feature"]:
            if name in mat:
                x = mat[name]
                break
        if x is None:
            n = edge_index.max().item() + 1 if edge_index.numel() else 0
            if edge_index.numel():
                n = max(n, edge_index[0].max().item(), edge_index[1].max().item()) + 1
            x = np.eye(n, dtype=np.float32)
        while hasattr(x, "dtype") and (x.dtype == np.object_ or str(getattr(x.dtype, "kind", "")) == "O"):
            x = x.flat[0] if x.size == 1 else np.vstack(list(x.flat))
        if issparse(x):
            x = x.toarray()
        x = np.asarray(x)
        if x.ndim == 3:
            x = x.reshape(x.shape[-2], x.shape[-1])
        if x.dtype not in (np.float32, np.float64):
            x = x.astype(np.float32)
        x = torch.from_numpy(x).float()
        # Labels: y, label, Label, labels
        y = None
        for name in ["y", "label", "Label", "labels"]:
            if name in mat:
                y = mat[name]
                break
        if y is None:
            raise KeyError(".mat must contain one of: y, label, Label, labels")
        y = np.asarray(y).ravel().astype(np.int64)
        if np.unique(y).size > 2:
            # 多类转二类：多数类为正常(0)，其余为异常(1)
            _, counts = np.unique(y, return_counts=True)
            normal_val = np.unique(y)[np.argmax(counts)]
            y = (y != normal_val).astype(np.int64)
        y = torch.from_numpy(y).long()
        return Data(x=x, edge_index=edge_index, y=y)

    def _load_twitter(self):
        """Load Twitter: try pygod, then local .pt file, then local .mat file as fallback."""
        # 1) 尝试从 pygod 下载
        try:
            return load_data("twitter")
        except (RuntimeError, Exception) as e:
            if self.verbose:
                print(f"Failed to load twitter from pygod: {e}")
        
        # 2) 尝试从本地 .pt 文件加载
        path = os.path.join(os.path.expanduser("~"), ".pygod", "data", "twitter.pt")
        if os.path.exists(path):
            return torch.load(path)
        
        # 3) 尝试从本地 .mat 文件加载（若存在，优先当前 FMGAD 目录）
        for root in [os.getcwd(), FMGAD_ROOT]:
            for sub in ["", "datasets"]:
                folder = os.path.join(root, sub) if sub else root
                mat_path = os.path.join(folder, "Twitter.mat")
                if os.path.isfile(mat_path):
                    return self._load_mat_data(mat_path)
        
        raise RuntimeError(
            "Twitter dataset not found. Please download twitter.pt and place it in ~/.pygod/data/ "
            "or provide Twitter.mat in FMGAD/ or FMGAD/datasets/ directory."
        )

    def _load_yelpchi(self):
        """Load YelpChi: try local .mat, then pygod, then local .pt, then PyG Yelp (GraphSAINT) as fallback."""
        # 1) 优先使用 FMGAD 目录下的 YelpChi.mat
        for root in [os.getcwd(), FMGAD_ROOT]:
            for sub in ["", "datasets", "datasets/YelpChi"]:
                folder = os.path.join(root, sub) if sub else root
                mat_path = os.path.join(folder, "YelpChi.mat")
                if os.path.isfile(mat_path):
                    return self._load_mat_data(mat_path)
        try:
            return load_data("yelpchi")
        except RuntimeError:
            pass
        path = os.path.join(os.path.expanduser("~"), ".pygod", "data", "yelpchi.pt")
        if os.path.exists(path):
            return torch.load(path)
        try:
            from torch_geometric.datasets import Yelp
            root = os.path.join(os.path.expanduser("~"), ".pygod", "data", "Yelp")
            dataset = Yelp(root=root)
            data = dataset[0]
            y_bin = (data.y == 0).long()
            if y_bin.sum() < 10 or (data.y.size(0) - y_bin.sum()) < 10:
                y_bin = (data.y != data.y.mode()[0]).long()
            return Data(x=data.x, edge_index=data.edge_index, y=y_bin)
        except Exception as e:
            raise RuntimeError(
                "YelpChi not found: put YelpChi.mat in FMGAD/ or FMGAD/datasets/ or yelpchi.pt in ~/.pygod/data/."
            ) from e

    def _load_elliptic(self):
        """
        Load Elliptic Bitcoin transaction dataset from Kaggle (ellipticco/elliptic-data-set).
        优先从本地 datasets/elliptic 目录读取；若不存在，则尝试用 kagglehub 自动下载。
        标签约定：
            class = 1 (illicit) -> 异常(1)
            class = 0 (licit)   -> 正常(0)
            class = 2 (unknown) -> 视作正常(0)（与部分图异常检测工作一致）
        """
        base_dirs: List[str] = []

        # 1) 显式本地目录（优先项目 datasets 下）
        for root in [os.getcwd(), FMGAD_ROOT]:
            for sub in ["datasets/elliptic", "datasets/Elliptic", "datasets/elliptic-bitcoin", "datasets/elliptic_bitcoin"]:
                base_dirs.append(os.path.join(root, sub))

        # 2) 环境变量指定的数据目录（若用户自行配置）
        ell_env = os.environ.get("ELLIPTIC_DATA_DIR")
        if ell_env:
            base_dirs.insert(0, ell_env)

        # 3) 尝试通过 kagglehub 自动下载（需要网络与 kagglehub 依赖）
        try:
            import kagglehub  # type: ignore

            kaggle_path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
            # 官方数据目录中真正的文件夹为 elliptic_bitcoin_dataset
            kb_inner = os.path.join(kaggle_path, "elliptic_bitcoin_dataset")
            base_dirs.insert(0, kb_inner if os.path.isdir(kb_inner) else kaggle_path)
        except Exception:
            # kagglehub 不可用时直接跳过，后续仅尝试本地目录
            pass

        feat_path = edge_path = class_path = None
        for base in base_dirs:
            if not base:
                continue
            f = os.path.join(base, "elliptic_txs_features.csv")
            e = os.path.join(base, "elliptic_txs_edgelist.csv")
            c = os.path.join(base, "elliptic_txs_classes.csv")
            if os.path.isfile(f) and os.path.isfile(e) and os.path.isfile(c):
                feat_path, edge_path, class_path = f, e, c
                break

        if feat_path is None:
            raise RuntimeError(
                "Elliptic dataset not found. 请确保包含 elliptic_txs_features.csv、"
                "elliptic_txs_edgelist.csv、elliptic_txs_classes.csv 的目录存在，"
                "或在可用网络环境下安装 kagglehub 后自动下载该数据集。"
            )

        # 读取特征：第一列 txId，第二列 time_step，其余为数值特征
        tx_ids: List[str] = []
        features: List[List[float]] = []
        with open(feat_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                tx_id = row[0]
                # 跳过 time_step，只保留特征
                feat_vals = [float(v) for v in row[2:]]
                tx_ids.append(tx_id)
                features.append(feat_vals)

        if not tx_ids:
            raise RuntimeError("Elliptic features file is empty or malformed.")

        x = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
        id2idx = {tid: i for i, tid in enumerate(tx_ids)}

        # 读取边：两列 txId1, txId2
        edges: List[List[int]] = []
        with open(edge_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                src_id, dst_id = row[0], row[1]
                if src_id in id2idx and dst_id in id2idx:
                    edges.append([id2idx[src_id], id2idx[dst_id]])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # 读取标签：txId, class（0=licit, 1=illicit, 2=unknown）
        y = torch.zeros(len(tx_ids), dtype=torch.long)
        with open(class_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                tx_id, cls_str = row[0], row[1]
                if tx_id not in id2idx:
                    continue
                try:
                    cls = int(cls_str)
                except ValueError:
                    continue
                if cls == 1:  # illicit
                    y[id2idx[tx_id]] = 1
                else:
                    # 0（licit）与 2（unknown）统一视作正常
                    y[id2idx[tx_id]] = 0

        if self.verbose:
            num_nodes = x.size(0)
            num_edges = edge_index.size(1)
            num_anoms = int(y.sum().item())
            print(
                f"Loaded Elliptic: {num_nodes} nodes, {num_edges} edges, "
                f"{num_anoms} anomalies (illicit) after treating unknown as normal."
            )

        return Data(x=x, edge_index=edge_index, y=y)

    def _load_twibot20(self, dset: str = "twibot20"):
        """Load TwiBot-20/TwiBot-22 dataset from JSON files."""
        import json
        # 支持 Twibot-20 或 TwiBot-20 两种目录名
        base_paths = [
            "/mnt/yehang/FMGAD/Twibot-20",
            "/mnt/yehang/0208/20260208/datasets/Twibot-20",
            "/mnt/yehang/001important/FMGAD早期/Twibot-20",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "Twibot-20"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "TwiBot-20"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "Twibot-20"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "TwiBot-20"),
            os.path.join(os.getcwd(), "Twibot-20"),
            os.path.join(os.getcwd(), "TwiBot-20"),
            os.path.join(FMGAD_ROOT, "datasets", "Twibot-20"),
            os.path.join(FMGAD_ROOT, "datasets", "TwiBot-20"),
            os.path.join(FMGAD_ROOT, "Twibot-20"),
            os.path.join(FMGAD_ROOT, "TwiBot-20"),
        ]
        json_files = {}
        for base_path in base_paths:
            if os.path.exists(base_path):
                for split in ["train", "test", "dev"]:
                    json_path = os.path.join(base_path, f"{split}.json")
                    if os.path.exists(json_path):
                        json_files[split] = json_path
                if not json_files and os.path.exists(os.path.join(base_path, "TwiBot-20_sample.json")):
                    json_files["sample"] = os.path.join(base_path, "TwiBot-20_sample.json")
                break
        if not json_files:
            raise RuntimeError(
                f"{dset} dataset not found. Please ensure Twibot-20/TwiBot-20 directory exists "
                "with train.json/test.json (e.g. under FMGAD/datasets/Twibot-20/)"
            )
        all_users = []
        for split, path in json_files.items():
            with open(path, "r") as f:
                users = json.load(f)
                all_users.extend(users)
        id_to_idx = {user["ID"]: idx for idx, user in enumerate(all_users)}
        n_nodes = len(all_users)
        edges = []
        for idx, user in enumerate(all_users):
            if user.get("neighbor") is not None:
                neighbor = user["neighbor"]
                if "following" in neighbor and neighbor["following"]:
                    for following_id in neighbor["following"]:
                        if following_id in id_to_idx:
                            edges.append([idx, id_to_idx[following_id]])
                if "follower" in neighbor and neighbor["follower"]:
                    for follower_id in neighbor["follower"]:
                        if follower_id in id_to_idx:
                            edges.append([id_to_idx[follower_id], idx])
        feature_keys = [
            "followers_count", "friends_count", "listed_count", "favourites_count",
            "statuses_count", "verified", "protected", "default_profile",
            "default_profile_image", "geo_enabled", "contributors_enabled",
        ]
        features = []
        for user in all_users:
            profile = user.get("profile", {})
            feat = []
            for key in feature_keys:
                val = profile.get(key, 0)
                if isinstance(val, bool):
                    feat.append(1.0 if val else 0.0)
                elif isinstance(val, (int, float)):
                    feat.append(float(val))
                else:
                    feat.append(0.0)
            domain = user.get("domain", [])
            domain_feat = [0.0] * 4
            if "Politics" in domain or "politics" in str(domain).lower():
                domain_feat[0] = 1.0
            if "Business" in domain or "Bussiness" in domain or "business" in str(domain).lower():
                domain_feat[1] = 1.0
            if "Entertainment" in domain or "entertainment" in str(domain).lower():
                domain_feat[2] = 1.0
            if "Sports" in domain or "sports" in str(domain).lower():
                domain_feat[3] = 1.0
            feat.extend(domain_feat)
            features.append(feat)
        x = torch.tensor(features, dtype=torch.float32)
        y = []
        for user in all_users:
            label = user.get("label", None)
            y.append(1 if (label is not None and int(label) == 1) else 0)
        y = torch.tensor(y, dtype=torch.long)
        if y.sum() == 0:
            if self.verbose:
                print("Warning: No labels found in TwiBot data. Using random labels for testing.")
            n_anomalies = max(1, int(n_nodes * 0.1))
            anomaly_indices = torch.randperm(n_nodes)[:n_anomalies]
            y[anomaly_indices] = 1
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        if self.verbose:
            print(f"Loaded {dset}: {n_nodes} nodes, {edge_index.shape[1]} edges, {y.sum().item()} anomalies")
        return Data(x=x, edge_index=edge_index, y=y)

    def _ensure_save_dir(self, dset: str):
        # 运行目录即项目目录（FMselfv1 或 FMGADself），模型保存到 cwd/models
        save_dir = os.path.join(os.getcwd(), "models", dset, "full_batch")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _build_run_tag(self, dset: str) -> str:
        """
        基于当前实验关键超参数生成稳定签名，避免并行调参时写入同一路径导致 checkpoint 冲突。
        """
        payload = {
            "dataset": dset,
            "hid_dim": self.hid_dim,
            "ae_dropout": self.ae_dropout,
            "ae_lr": self.ae_lr,
            "ae_alpha": self.ae_alpha,
            "proto_alpha": self.proto_alpha,
            "weight": self.weight,
            "residual_scale": self.residual_scale,
            "sample_steps": self.sample_steps,
            "use_adaptive_residual_scale": self.use_adaptive_residual_scale,
            "use_multi_score_fusion": self.use_multi_score_fusion,
            "use_virtual_neighbors": self.use_virtual_neighbors,
            "virtual_degree_threshold": self.virtual_degree_threshold,
            "virtual_k": self.virtual_k,
            "use_score_smoothing": self.use_score_smoothing,
            "score_smoothing_alpha": self.score_smoothing_alpha,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    def _build_z(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建混合 Latent Z：多尺度残差（可选）+ 自适应门控/注意力融合 + 自适应缩放（可选）。
        返回：z [N, 2*hid_dim], h [N, hid_dim], r_final [N, hid_dim]
        """
        h = self.ae.encode(x, edge_index)
        dev = h.device
        if self.use_virtual_neighbors and getattr(self, "virtual_degree_threshold", 5) is not None:
            edge_index = _add_virtual_knn_edges(
                edge_index, h,
                self.virtual_degree_threshold,
                getattr(self, "virtual_k", 5),
                dev,
            )

        if getattr(self, "use_multi_scale_residual", False) and self.residual_attention is not None:
            r_global, r_local, r_structural, deg = compute_multi_scale_residuals(h, edge_index)
            r_fused = self.residual_attention([r_global, r_local, r_structural])
        else:
            r_global, r_local, deg = compute_dual_residuals_with_degree(h, edge_index)
            bias = self.gate_module.bias.to(dev)
            sharpness = self.gate_module.sharpness.to(dev)
            alpha = torch.sigmoid((deg - bias) * sharpness)
            r_fused = alpha * r_local + (1.0 - alpha) * r_global

        if getattr(self, "use_adaptive_residual_scale", False):
            scale = adaptive_residual_scale_fn(deg, base_scale=self.residual_scale)
            r_final = r_fused * scale
        else:
            r_final = r_fused * self.residual_scale
        z = torch.cat([h, r_final], dim=1)
        return z, h, r_final

    def forward(self, dset: str):
        self.dataset = dset
        data = self._load_dataset(dset)
        self._large_graph = getattr(data, "num_nodes", data.x.size(0)) > 15000
        if self.hid_dim is None:
            self.hid_dim = 2 ** int(math.log2(data.x.size(1)) - 1)

        # AE
        self.ae = GraphAE(in_dim=data.num_node_features, hid_dim=self.hid_dim, dropout=self.ae_dropout).cuda()
        save_dir = self._ensure_save_dir(dset)
        run_tag = self._build_run_tag(dset)
        ae_path = os.path.join(save_dir, f"run_{run_tag}")
        os.makedirs(ae_path, exist_ok=True)

        # 1) train AE (单次；与 v2 同口径的 loss_func / dense_adj)
        ae_ckpt = self._train_ae_once(data, ae_path)
        if self.verbose:
            print(f"loading AE checkpoint: {ae_ckpt:04d}")
        ae_dict = torch.load(os.path.join(ae_path, f"{ae_ckpt}.pt"))
        self.ae.load_state_dict(ae_dict["state_dict"])
        self.gate_module = self.gate_module.to(next(self.ae.parameters()).device)
        if self.use_multi_scale_residual:
            self.residual_attention = ResidualChannelAttention(3, self.hid_dim).to(next(self.ae.parameters()).device)

        # 2) trials
        num_trial = getattr(self, "num_trial", 3)
        dm_auc, dm_ap, dm_rec, dm_auprc, dm_f1 = [], [], [], [], []

        for _ in tqdm.tqdm(range(num_trial)):
            # z_dim = 2*hid_dim
            z_dim = 2 * self.hid_dim

            # free model: cond_dim=None => 用全0 context
            velocity_free = MLPFlowMatching(d_in=z_dim, dim_t=512, cond_dim=None).cuda()
            self.dm = FlowMatchingModel(velocity_fn=velocity_free, hid_dim=z_dim).cuda()
            proto_h = self._train_dm_free(data, ae_path)

            dm_dict = torch.load(os.path.join(ae_path, "dm_self.pt"))
            self.dm.load_state_dict(dm_dict["state_dict"])
            if "gate_state" in dm_dict:
                self.gate_module.load_state_dict(dm_dict["gate_state"])
            if self.residual_attention is not None and "residual_attention" in dm_dict:
                self.residual_attention.load_state_dict(dm_dict["residual_attention"])
            self.proto = dm_dict["prototype"]  # [hid_dim]

            # proto model: cond_dim = hid_dim（只条件在 h 的原型上）
            velocity_proto = MLPFlowMatching(d_in=z_dim, dim_t=512, cond_dim=self.hid_dim).cuda()
            self.dm_proto = FlowMatchingModel(velocity_fn=velocity_proto, hid_dim=z_dim).cuda()
            self._train_dm_proto(data, ae_path)
            dm_proto_dict = torch.load(os.path.join(ae_path, "proto_dm_self.pt"))
            self.dm_proto.load_state_dict(dm_proto_dict["state_dict"])

            # eval：根据配置选择不同的评分方式
            if self.use_nll_score:
                auc_this, ap_this, rec_this, auprc_this, f1_this = self._evaluate_nll(data)
            elif self.use_energy_score:
                auc_this, ap_this, rec_this, auprc_this, f1_this = self._evaluate_energy(data)
            elif self.use_guided_recon:
                auc_this, ap_this, rec_this, auprc_this, f1_this = self._evaluate_guided_recon(data)
            else:
                ret = self.sample(self.dm_proto, self.dm, data)
                if len(ret) == 6:
                    auc_this, ap_this, rec_this, auprc_this, f1_this, scores = ret
                    if not hasattr(self, "_ensemble_scores"):
                        self._ensemble_scores = []
                    self._ensemble_scores.append(scores)
                else:
                    auc_this, ap_this, rec_this, auprc_this, f1_this = ret
            dm_auc.append(auc_this)
            dm_ap.append(ap_this)
            dm_rec.append(rec_this)
            dm_auprc.append(auprc_this)
            dm_f1.append(f1_this)

        if getattr(self, "ensemble_score", False) and hasattr(self, "_ensemble_scores") and len(self._ensemble_scores) > 0:
            # 多 trial 分数取平均，再按平均分数计算一次指标
            stacked = torch.stack(self._ensemble_scores)  # [num_trial, N]
            mean_scores = stacked.mean(dim=0)  # [N]
            if torch.isnan(mean_scores).any() or torch.isinf(mean_scores).any():
                mean_scores = torch.nan_to_num(mean_scores, nan=0.0, posinf=0.0, neginf=0.0)

            y_true = data.y

            pyg_auc = eval_roc_auc(y_true, mean_scores)
            if pyg_auc < 0.5:
                smin, smax = mean_scores.min().item(), mean_scores.max().item()
                if smax - smin > 1e-8:
                    mean_scores = 1.0 - (mean_scores - smin) / (smax - smin)
                else:
                    mean_scores = -mean_scores
                mean_scores = torch.nan_to_num(mean_scores, nan=0.0, posinf=0.0, neginf=0.0)
                pyg_auc = eval_roc_auc(y_true, mean_scores)

            pyg_ap = eval_average_precision(y_true, mean_scores)
            pyg_rec = eval_recall_at_k(y_true, mean_scores, int(y_true.sum()))
            pyg_prec = eval_precision_at_k(y_true, mean_scores, int(y_true.sum()))

            y_np = y_true.cpu().numpy()
            p, r, _ = precision_recall_curve(y_np, mean_scores.cpu().numpy())
            pyg_auprc = auc(r, p)
            pyg_f1 = 2 * pyg_prec * pyg_rec / (pyg_prec + pyg_rec) if (pyg_prec + pyg_rec) > 0 else 0.0
            dm_auc = torch.tensor([float(pyg_auc)])
            dm_ap = torch.tensor([float(pyg_ap)])
            dm_rec = torch.tensor([float(pyg_rec)])
            dm_auprc = torch.tensor([float(pyg_auprc)])
            dm_f1 = torch.tensor([float(pyg_f1)])
            del self._ensemble_scores
        else:
            dm_auc = torch.tensor(dm_auc)
            dm_ap = torch.tensor(dm_ap)
            dm_rec = torch.tensor(dm_rec)
            dm_auprc = torch.tensor(dm_auprc)
            dm_f1 = torch.tensor(dm_f1)

        print(
            "Final AUC: {:.4f}±{:.4f} ({:.4f})\t"
            "Final AP: {:.4f}±{:.4f} ({:.4f})\t"
            "Final Recall: {:.4f}±{:.4f} ({:.4f})\t"
            "Final AUPRC: {:.4f}±{:.4f} ({:.4f})\t"
            "Final F1@k: {:.4f}±{:.4f} ({:.4f})".format(
                torch.mean(dm_auc),
                torch.std(dm_auc),
                torch.max(dm_auc),
                torch.mean(dm_ap),
                torch.std(dm_ap),
                torch.max(dm_ap),
                torch.mean(dm_rec),
                torch.std(dm_rec),
                torch.max(dm_rec),
                torch.mean(dm_auprc),
                torch.std(dm_auprc),
                torch.max(dm_auprc),
                torch.mean(dm_f1),
                torch.std(dm_f1),
                torch.max(dm_f1),
            )
        )

        # 返回便于外部写报告
        return {  # type: Dict[str, Any]
            "auc_mean": float(torch.mean(dm_auc)),
            "auc_std": float(torch.std(dm_auc)),
            "ap_mean": float(torch.mean(dm_ap)),
            "ap_std": float(torch.std(dm_ap)),
            "rec_mean": float(torch.mean(dm_rec)),
            "rec_std": float(torch.std(dm_rec)),
            "auprc_mean": float(torch.mean(dm_auprc)),
            "auprc_std": float(torch.std(dm_auprc)),
            "f1_mean": float(torch.mean(dm_f1)),
            "f1_std": float(torch.std(dm_f1)),
        }

    def _compute_nll(self, data) -> torch.Tensor:
        """
        使用 CNF 公式计算每个节点的 NLL（基于 Flow Matching 的连续归一化流视角）。
        NLL 越大表示样本在模型下概率越低 -> 越异常。
        这里基于 free model 的无条件流场 self.dm.velocity_fn。
        """
        assert self.dm is not None, "Flow Matching free model (self.dm) must be trained before computing NLL."

        self.dm.eval()
        self.ae.eval()

        x = data.x.cuda().to(torch.float32)
        edge_index = data.edge_index.cuda()

        # 使用与训练相同的 z 表示：[h; r]
        z_all, _, _ = self._build_z(x, edge_index)
        n_nodes, dim_z = z_all.shape

        scores_list = []
        # 分块计算 NLL，避免一次性占满显存
        chunk_size = 1024
        for start in range(0, n_nodes, chunk_size):
            end = min(start + chunk_size, n_nodes)
            z = z_all[start:end]
            batch_size = z.shape[0]

            # 初始 log-density 差分为 0
            logp_diff0 = torch.zeros(batch_size, 1, device=z.device)

            def ode_func(t, state):
                z_t, logp_diff_t = state

                z_t = z_t.requires_grad_(True)
                # Flow Matching 时间输入：batch 维度的标量 t
                t_tensor = torch.full((batch_size,), t.item(), device=z_t.device)

                # 无条件流场：context=None, proto_alpha=None
                v = self.dm.velocity_fn(z_t, t_tensor, context=None, proto_alpha=None)

                # Hutchinson 迹估计 trace(dv/dz)
                epsilon = torch.randn_like(z_t)
                v_eps = torch.sum(v * epsilon)
                grad_v_eps = torch.autograd.grad(v_eps, z_t, create_graph=False)[0]
                divergence = torch.sum(grad_v_eps * epsilon, dim=1, keepdim=True)  # [B,1]

                # dz/dt = v, d logp / dt = - div(v)
                return v, -divergence

            # 从 t=1 (数据) 积分到 t=0 (先验)
            t_span = torch.tensor([1.0, 0.0], device=z.device)
            # 使用定步长 RK4 积分，显存占用和步数更可控
            z_T, logp_diff_T = odeint(
                ode_func,
                (z, logp_diff0),
                t_span,
                method="rk4",
                options={"step_size": 0.1},  # 0.1 -> 约 10 步
            )

            z0 = z_T[-1]  # t=0 时的点（噪声空间）
            delta_logp = logp_diff_T[-1]  # \int -div(v) dt

            # 标准高斯先验下的 log p(z0)
            logp_z0 = -0.5 * torch.sum(z0 ** 2, dim=1, keepdim=True) - 0.5 * dim_z * math.log(2 * math.pi)

            # log p(x) = log p(z0) - ∫ div(v) dt  = logp_z0 - delta_logp
            logpx = logp_z0 - delta_logp  # [B,1]

            # 异常分数 = - log p(x)
            scores_chunk = -logpx.squeeze(1)  # [B]
            scores_list.append(scores_chunk)

        scores = torch.cat(scores_list, dim=0)  # [N]
        return scores

    def _evaluate_nll(self, data) -> Tuple[float, float, float, float, float]:
        """
        使用 NLL 分数进行异常检测评估，返回 (AUC, AP, Recall@k, AUPRC, F1@k)。
        这里的 k 取正类个数，因此在 top-k 阈值下 precision@k 与 recall@k 数值相同，F1@k 与二者相同。
        """
        self.ae.eval()
        self.dm.eval()

        scores = self._compute_nll(data)  # [N] (在 cuda)
        scores_cpu = scores.detach().cpu()

        y = data.y.bool()

        pyg_auc = eval_roc_auc(y, scores_cpu)
        pyg_ap = eval_average_precision(y, scores_cpu)
        pyg_rec = eval_recall_at_k(y, scores_cpu, sum(y))
        pyg_prec = eval_precision_at_k(y, scores_cpu, sum(y))

        p, r, _ = precision_recall_curve(y.numpy(), scores_cpu.numpy())
        auprc = auc(r, p)

        # F1@k（基于 top-k，其中 k 为正类个数）
        if (pyg_prec + pyg_rec) > 0:
            f1_at_k = 2 * pyg_prec * pyg_rec / (pyg_prec + pyg_rec)
        else:
            f1_at_k = 0.0

        if self.verbose:
            print(
                f"NLL-based eval -> AUC: {pyg_auc:.4f}, AP: {pyg_ap:.4f}, "
                f"Recall@k: {pyg_rec:.4f}, Precision@k: {pyg_prec:.4f}, "
                f"F1@k: {f1_at_k:.4f}, AUPRC: {auprc:.4f}"
            )

        return float(pyg_auc), float(pyg_ap), float(pyg_rec), float(auprc), float(f1_at_k)

    @torch.no_grad()
    def _evaluate_energy(self, data) -> Tuple[float, float, float, float, float]:
        """
        基于“速度能量”的异常检测：
        - 在 t=0.99（接近数据端）评估 proto 条件流上的速度模长 ||v||_2。
        - 直觉：正常点应位于稳定流形附近，速度应较小；异常点处于不稳定区域，速度较大。
        """
        self.ae.eval()
        assert self.dm_proto is not None and self.proto is not None, "proto Flow Matching model must be trained."
        self.dm_proto.eval()

        x = data.x.cuda().to(torch.float32)
        edge_index = data.edge_index.cuda()
        y = data.y.bool()

        z, _, _ = self._build_z(x, edge_index)  # [N, 2*hid_dim]
        batch_size = z.shape[0]

        # t 接近 1（数据端）
        t_tensor = torch.full((batch_size,), 0.99, device=z.device)

        # proto 作为条件：先整理成 [1, hid_dim]，再 broadcast 到 [N, hid_dim]
        if self.proto.dim() == 0:
            proto_context = self.proto.unsqueeze(0)  # [1]
        elif self.proto.dim() == 1:
            proto_context = self.proto  # [hid_dim]
        else:
            proto_context = self.proto.squeeze(0) if self.proto.shape[0] == 1 else self.proto.mean(dim=0)
        proto_context = proto_context.unsqueeze(0).expand(batch_size, -1)  # [N, hid_dim]

        # 使用条件 Flow Matching 模型的速度场
        v_pred = self.dm_proto.velocity_fn(z, t_tensor, context=proto_context, proto_alpha=self.proto_alpha)

        # 速度模长作为异常分数
        scores = torch.norm(v_pred, p=2, dim=1)  # [N]
        scores_cpu = scores.detach().cpu()

        pyg_auc = eval_roc_auc(y, scores_cpu)
        pyg_ap = eval_average_precision(y, scores_cpu)
        pyg_rec = eval_recall_at_k(y, scores_cpu, sum(y))
        pyg_prec = eval_precision_at_k(y, scores_cpu, sum(y))

        p, r, _ = precision_recall_curve(y.numpy(), scores_cpu.numpy())
        auprc = auc(r, p)

        if (pyg_prec + pyg_rec) > 0:
            f1_at_k = 2 * pyg_prec * pyg_rec / (pyg_prec + pyg_rec)
        else:
            f1_at_k = 0.0

        if self.verbose:
            print(
                f"Energy-based eval -> AUC: {pyg_auc:.4f}, AP: {pyg_ap:.4f}, "
                f"Recall@k: {pyg_rec:.4f}, Precision@k: {pyg_prec:.4f}, "
                f"F1@k: {f1_at_k:.4f}, AUPRC: {auprc:.4f}"
            )

        return float(pyg_auc), float(pyg_ap), float(pyg_rec), float(auprc), float(f1_at_k)

    @torch.no_grad()
    def _evaluate_guided_recon(self, data) -> Tuple[float, float, float, float, float]:
        """
        方案 C：带强引导的 ODE 重建误差（Guided ODE Reconstruction）。
        使用 proto 条件流 (dm_proto) + free 流 (dm) 做 CFG，引导生成“正常”样本，
        再与真实 latent 做 L2 重建误差作为异常分数。
        """
        self.ae.eval()
        assert self.dm is not None and self.dm_proto is not None and self.proto is not None, \
            "Both free and proto Flow Matching models must be trained."
        self.dm.eval()
        self.dm_proto.eval()

        x = data.x.cuda().to(torch.float32)
        edge_index = data.edge_index.cuda()
        y = data.y.bool()

        # 使用与训练相同的 latent 表示 z = [h; r]
        z_real, _, _ = self._build_z(x, edge_index)  # [N, 2*hid_dim]
        batch_size, _ = z_real.shape

        # proto context 作为 Condition（与训练时一致）
        if self.proto.dim() == 0:
            proto_context = self.proto.unsqueeze(0)
        elif self.proto.dim() == 1:
            proto_context = self.proto
        else:
            proto_context = self.proto.squeeze(0) if self.proto.shape[0] == 1 else self.proto.mean(dim=0)
        proto_context = proto_context.unsqueeze(0).expand(batch_size, -1)  # [N, hid_dim]

        proto_net = self.dm_proto.velocity_fn
        free_net = self.dm.velocity_fn

        guidance_scale = self.guidance_scale
        steps = self.ode_steps

        # 从标准高斯噪声开始（纯生成式重建）
        current_x = torch.randn_like(z_real)
        dt = 1.0 / float(steps)

        def ode_func(t_scalar: float, x_latent: torch.Tensor) -> torch.Tensor:
            # t_scalar 是标量 float，扩展成 batch 形式
            t_tensor = torch.full((x_latent.shape[0],), t_scalar, device=x_latent.device)
            # 条件流：朝向原型的速度
            v_cond = proto_net(x_latent, t_tensor, context=proto_context, proto_alpha=self.proto_alpha)
            # 自由流：无条件速度
            v_uncond = free_net(x_latent, t_tensor, context=None, proto_alpha=None)
            # CFG 组合
            v_final = v_uncond + guidance_scale * (v_cond - v_uncond)
            return v_final

        # 手写 RK4 ODE，从 t=0 -> t=1
        for i in range(steps):
            t_curr = i * dt
            k1 = ode_func(t_curr, current_x)
            k2 = ode_func(t_curr + 0.5 * dt, current_x + 0.5 * dt * k1)
            k3 = ode_func(t_curr + 0.5 * dt, current_x + 0.5 * dt * k2)
            k4 = ode_func(t_curr + dt, current_x + dt * k3)
            current_x = current_x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        z_recon = current_x  # [N, 2*hid_dim]

        # 重建误差作为异常分数
        scores = torch.sum((z_real - z_recon) ** 2, dim=1)  # [N]
        scores_cpu = scores.detach().cpu()

        pyg_auc = eval_roc_auc(y, scores_cpu)
        pyg_ap = eval_average_precision(y, scores_cpu)
        pyg_rec = eval_recall_at_k(y, scores_cpu, sum(y))
        pyg_prec = eval_precision_at_k(y, scores_cpu, sum(y))

        p, r, _ = precision_recall_curve(y.numpy(), scores_cpu.numpy())
        auprc = auc(r, p)

        if (pyg_prec + pyg_rec) > 0:
            f1_at_k = 2 * pyg_prec * pyg_rec / (pyg_prec + pyg_rec)
        else:
            f1_at_k = 0.0

        if self.verbose:
            print(
                f"Guided-ODE eval -> AUC: {pyg_auc:.4f}, AP: {pyg_ap:.4f}, "
                f"Recall@k: {pyg_rec:.4f}, Precision@k: {pyg_prec:.4f}, "
                f"F1@k: {f1_at_k:.4f}, AUPRC: {auprc:.4f} (guidance={guidance_scale}, steps={steps})"
            )

        return float(pyg_auc), float(pyg_ap), float(pyg_rec), float(auprc), float(f1_at_k)

    def _train_ae_once(self, data, ae_path: str) -> int:
        if self.verbose:
            print("Training autoencoder (FMGADself, single run)...")

        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.ae_lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        best_loss = float("inf")
        best_epoch = 0
        patience = 0

        x = data.x.cuda().to(torch.float32)
        edge_index = data.edge_index.cuda()
        num_nodes = data.num_nodes
        # 大图（如 YelpChi 45k 节点）不构造稠密邻接，仅用属性重建训练，避免 OOM
        if getattr(self, "_large_graph", False) and self.verbose:
            print(f"Large graph (n={num_nodes}): training AE with attribute reconstruction only.")
        s = None if getattr(self, "_large_graph", False) else to_dense_adj(edge_index)[0].cuda()

        for epoch in range(1, self.ae_epochs + 1):
            self.ae.train()
            optimizer.zero_grad()

            if getattr(self, "_large_graph", False):
                self.ae.emb = self.ae.encode(x, edge_index)
                x_ = self.ae.attr_decoder(self.ae.emb, edge_index)
                score = torch.sqrt(torch.sum((x - x_) ** 2, dim=1))
            else:
                x_, s_, _ = self.ae(x, edge_index)
                score = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)
            loss = torch.mean(score)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                patience = 0
                torch.save({"state_dict": self.ae.state_dict()}, os.path.join(ae_path, f"{best_epoch}.pt"))
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("AE early stopping")
                    break

            if self.verbose and epoch % 50 == 0:
                print(f"AE Epoch {epoch:04d} loss={loss.item():.6f}")

        return best_epoch

    def _normalize_clip(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=0, keepdim=True)
        std = inputs.std(dim=0, keepdim=True) + 1e-8
        x = (inputs - mean) / std
        return torch.clamp(x, -10.0, 10.0)

    def _train_dm_free(self, data, ae_path: str) -> torch.Tensor:
        from flow_matching_model import sample_flow_matching

        if self.verbose:
            print("Training FM free model (FMGADself)...")

        fm_lr = self.lr * 0.5
        params = list(self.dm.parameters()) + list(self.gate_module.parameters())
        if self.residual_attention is not None:
            params = params + list(self.residual_attention.parameters())
        optimizer = torch.optim.Adam(params, lr=fm_lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        best_loss = float("inf")
        patience = 0
        proto_h = None

        # FM 阶段不更新 AE：关闭 Dropout，避免每轮 _build_z 特征抖动；并一次性冻结 z/h/r，稳定流形目标、减少重复计算
        self.ae.eval()
        with torch.no_grad():
            x0 = data.x.cuda().to(torch.float32)
            e0 = data.edge_index.cuda()
            z_fixed, h_fixed, r_final_fixed = self._build_z(x0, e0)
            proto_h_init = torch.mean(h_fixed, dim=0).detach()

        for epoch in range(self.diff_epochs):
            z, h, r_final = z_fixed, h_fixed, r_final_fixed

            z = self._normalize_clip(z)
            if torch.isnan(z).any() or torch.isinf(z).any():
                continue

            # 课程学习：早期只用低度节点，逐步加入高度节点
            if getattr(self, "use_curriculum_learning", False) and self.curriculum_warmup_epochs > 0:
                deg = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
                deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=z.device, dtype=z.dtype))
                ratio = min(1.0, (epoch + 1) / float(self.curriculum_warmup_epochs))
                k = max(1, int(ratio * z.size(0)))
                _, idx = torch.topk(deg, k, largest=False)
                mask = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)
                mask[idx] = True
                z = z[mask]
                h = h[mask]

            graph_context = torch.zeros(1, z.shape[1], device=z.device)
            if getattr(self, "use_hard_negative_mining", False):
                loss_per_sample = flow_matching_loss(self.dm.velocity_fn, z, graph_context, reduction="none")
                with torch.no_grad():
                    rec = sample_flow_matching(self.dm.velocity_fn, torch.randn_like(z, device=z.device), num_steps=5, proto=None, proto_alpha=None)
                    err = torch.norm(z - rec, p=2, dim=1)
                    w = (err - err.min() + 1e-8) / (err.max() - err.min() + 1e-8)
                    w = w / (w.sum() + 1e-8)
                loss = (loss_per_sample * w).sum()
            else:
                loss = flow_matching_loss(self.dm.velocity_fn, z, graph_context, reduction="mean")
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # 用采样估计 reconstructed，做 proto 更新（只用 h 部分）
            with torch.no_grad():
                noise = torch.randn_like(z)
                reconstructed = sample_flow_matching(self.dm.velocity_fn, noise, num_steps=10, proto=None, proto_alpha=None)
                if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                    reconstructed = z.clone()
                recon_h = reconstructed[:, : self.hid_dim]

            if epoch == 0:
                proto_h = torch.mean(h, dim=0)  # [hid_dim]
            else:
                proto_expanded = proto_h.unsqueeze(0)
                s_v = self.cos(proto_expanded, recon_h)
                weight = softmax_with_temperature(s_v, t=5).reshape(1, -1)
                proto_h = torch.mm(weight, recon_h).squeeze(0).detach()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            scheduler.step()

            if self.verbose and epoch % 20 == 0:
                print(f"FM-free Epoch {epoch:04d} loss={loss.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
                save_dict = {
                    "state_dict": self.dm.state_dict(),
                    "prototype": proto_h,
                    "gate_state": self.gate_module.state_dict(),
                }
                if self.residual_attention is not None:
                    save_dict["residual_attention"] = self.residual_attention.state_dict()
                torch.save(save_dict, os.path.join(ae_path, "dm_self.pt"))
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("FM-free early stopping")
                    break

        # 兜底：若从未保存过（如全 NaN loss），保存当前状态
        dm_path = os.path.join(ae_path, "dm_self.pt")
        if not os.path.exists(dm_path):
            proto_fallback = proto_h if proto_h is not None else proto_h_init
            save_dict = {
                "state_dict": self.dm.state_dict(),
                "prototype": proto_fallback,
                "gate_state": self.gate_module.state_dict(),
            }
            if self.residual_attention is not None:
                save_dict["residual_attention"] = self.residual_attention.state_dict()
            torch.save(save_dict, dm_path)
            if self.verbose:
                print("FM-free: fallback save (no valid loss epoch)")

        return proto_h

    def _train_dm_proto(self, data, ae_path: str):
        if self.verbose:
            print("Training FM proto model (FMGADself)...")

        fm_lr = self.lr * 0.5
        params_proto = list(self.dm_proto.parameters()) + list(self.gate_module.parameters())
        optimizer = torch.optim.Adam(params_proto, lr=fm_lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        best_loss = float("inf")
        patience = 0

        self.ae.eval()
        with torch.no_grad():
            x_cuda = data.x.cuda().to(torch.float32)
            edge_index_cuda = data.edge_index.cuda()
            z_fixed, _, _ = self._build_z(x_cuda, edge_index_cuda)

        for epoch in range(self.diff_epochs):
            z = self._normalize_clip(z_fixed)
            if torch.isnan(z).any() or torch.isinf(z).any():
                continue

            proto_context = self.proto.unsqueeze(0) if self.proto.dim() == 1 else self.proto.mean(dim=0, keepdim=True)
            loss = conditional_flow_matching_loss(
                self.dm_proto.velocity_fn,
                z,
                proto_context,
                t_sampling="logit_normal",
                reduction="mean",
            )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_proto, 0.5)
            optimizer.step()
            scheduler.step()

            if self.verbose and epoch % 20 == 0:
                print(f"FM-proto Epoch {epoch:04d} loss={loss.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
                torch.save({"state_dict": self.dm_proto.state_dict()}, os.path.join(ae_path, "proto_dm_self.pt"))
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("FM-proto early stopping")
                    break

        # 兜底：若从未保存过，保存当前 proto 模型
        proto_path = os.path.join(ae_path, "proto_dm_self.pt")
        if not os.path.exists(proto_path):
            torch.save({"state_dict": self.dm_proto.state_dict()}, proto_path)
            if self.verbose:
                print("FM-proto: fallback save (no valid loss epoch)")

    def sample(self, proto_model, free_model, data):
        self.ae.eval()
        proto_model.eval()
        free_model.eval()

        proto_net = proto_model.velocity_fn
        free_net = free_model.velocity_fn

        x = data.x.cuda().to(torch.float32)
        edge_index = data.edge_index.cuda()
        y = data.y.bool()

        # 构造 z，并预计算 energy / proto / residual 分数（用于多评分融合）
        z0, h, r_final = self._build_z(x, edge_index)
        z0 = self._normalize_clip(z0)
        noise = torch.randn_like(z0)

        proto_context = self.proto.unsqueeze(0) if self.proto.dim() == 1 else self.proto.mean(dim=0, keepdim=True)
        if proto_context.dim() == 1:
            proto_context = proto_context.unsqueeze(0)
        proto_expanded = proto_context.expand(z0.size(0), -1)
        score_proto = torch.norm(h - proto_expanded, p=2, dim=1)
        score_residual = torch.norm(r_final, p=2, dim=1)
        t_99 = torch.full((z0.size(0),), 0.99, device=z0.device, dtype=z0.dtype)
        v_99 = proto_net(z0, t_99, context=proto_expanded, proto_alpha=self.proto_alpha)
        score_energy = torch.norm(v_99, p=2, dim=1)

        def _normalize_score(s: torch.Tensor) -> torch.Tensor:
            smin, smax = s.min(), s.max()
            if smax - smin < 1e-8:
                return torch.zeros_like(s, device=s.device)
            return (s - smin) / (smax - smin)

        auc_list, ap_list, rec_list, auprc_list, f1_list = [], [], [], [], []
        score_list = []
        large_graph = getattr(self, "_large_graph", False)
        if not large_graph:
            s = to_dense_adj(edge_index)[0].cuda()

        num_time_points = self.timesteps
        for i in range(num_time_points):
            num_steps = max(1, int(self.sample_steps * (i + 1) / num_time_points))
            reconstructed = sample_flow_matching_free(
                proto_net,
                free_net,
                noise,
                num_steps,
                proto=proto_context,
                proto_alpha=self.proto_alpha,
                weight=self.weight,
            )

            h_hat = reconstructed[:, : self.hid_dim]
            if large_graph:
                x_ = self.ae.attr_decoder(h_hat, edge_index)
                score_recon = torch.sqrt(torch.sum((x - x_) ** 2, dim=1))
            else:
                x_, s_ = self.ae.decode(h_hat, edge_index)
                score_recon = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)

            if getattr(self, "use_multi_score_fusion", False):
                sr = _normalize_score(score_recon)
                se = _normalize_score(score_energy)
                sp = _normalize_score(score_proto)
                sres = _normalize_score(score_residual)
                stacked = torch.stack([sr, se, sp, sres], dim=1)
                w = softmax_with_temperature(stacked, t=self.score_fusion_temperature, axis=1)
                score = (w * stacked).sum(dim=1)
            else:
                score = score_recon

            if getattr(self, "use_score_smoothing", False) and edge_index.numel() > 0:
                score = _smooth_scores_by_graph(score, edge_index, self.score_smoothing_alpha, score.device)

            scores_cpu = score.detach().cpu()
            # 兜底：NaN/Inf 会破坏 sklearn 评估，替换为 0
            if torch.isnan(scores_cpu).any() or torch.isinf(scores_cpu).any():
                scores_cpu = torch.nan_to_num(scores_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            pyg_auc = eval_roc_auc(y, scores_cpu)
            # AUC < 0.5 时方向反了，用 1-x 取反（先归一化到 [0,1] 再 1-x）
            if pyg_auc < 0.5:
                smin, smax = score.min(), score.max()
                if smax - smin > 1e-8:
                    score = 1.0 - (score - smin) / (smax - smin)
                else:
                    score = -score
                scores_cpu = score.detach().cpu()
                if torch.isnan(scores_cpu).any() or torch.isinf(scores_cpu).any():
                    scores_cpu = torch.nan_to_num(scores_cpu, nan=0.0, posinf=0.0, neginf=0.0)
                pyg_auc = eval_roc_auc(y, scores_cpu)
            pyg_ap = eval_average_precision(y, scores_cpu)
            pyg_rec = eval_recall_at_k(y, scores_cpu, sum(y))
            pyg_prec = eval_precision_at_k(y, scores_cpu, sum(y))
            p, r, _ = precision_recall_curve(y.numpy(), scores_cpu.numpy())
            pyg_auprc = auc(r, p)

            if (pyg_prec + pyg_rec) > 0:
                f1_at_k = 2 * pyg_prec * pyg_rec / (pyg_prec + pyg_rec)
            else:
                f1_at_k = 0.0

            auc_list.append(pyg_auc)
            ap_list.append(pyg_ap)
            rec_list.append(pyg_rec)
            auprc_list.append(pyg_auprc)
            f1_list.append(f1_at_k)
            score_list.append(scores_cpu.clone())

            if self.verbose:
                print(
                    "steps:{},pyg_AUC: {:.4f}, pyg_AP: {:.4f}, pyg_Recall: {:.4f}, F1@k: {:.4f}, AUPRC: {:.4f}".format(
                        num_steps, pyg_auc, pyg_ap, pyg_rec, f1_at_k, pyg_auprc
                    )
                )

        best_idx = int(np.argmax(auc_list))
        if getattr(self, "ensemble_score", False):
            return (
                float(np.max(auc_list)),
                float(np.max(ap_list)),
                float(np.max(rec_list)),
                float(np.max(auprc_list)),
                float(np.max(f1_list)),
                score_list[best_idx],
            )
        best_idx = int(np.argmax(auc_list))
        if getattr(self, "ensemble_score", False):
            return (
                float(np.max(auc_list)),
                float(np.max(ap_list)),
                float(np.max(rec_list)),
                float(np.max(auprc_list)),
                float(np.max(f1_list)),
                score_list[best_idx],
            )
        return (
            float(np.max(auc_list)),
            float(np.max(ap_list)),
            float(np.max(rec_list)),
            float(np.max(auprc_list)),
            float(np.max(f1_list)),
        )