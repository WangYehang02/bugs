"""
图自编码器模块
从DiffGAD移植，提供GraphAE类（别名Graph_AE）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj
from pygod.nn.decoder import DotProductDecoder
from pygod.nn.functional import double_recon_loss


class GraphAE(nn.Module):
    """
    图自编码器（Graph AutoEncoder）
    
    这个类实现了一个图自编码器，用于学习图的低维表示。
    它包含：
    1. 一个共享编码器：将节点特征和结构编码为低维嵌入
    2. 一个属性解码器：从嵌入重构节点特征
    3. 一个结构解码器：从嵌入重构图结构（邻接矩阵）
    
    通过同时重构节点特征和图结构，模型可以学习到图的完整表示。
    """
    
    def __init__(self,
                 in_dim,  # 输入维度（节点特征维度）
                 hid_dim=64,  # 隐藏层维度（嵌入维度）
                 num_layers=4,  # 总层数
                 dropout=0.,  # Dropout 比率，用于防止过拟合
                 act=torch.nn.functional.relu,  # 激活函数
                 sigmoid_s=False,  # 是否对结构解码器输出使用 sigmoid
                 backbone=GCN,  # 使用的图神经网络类型（默认 GCN）
                 **kwargs):  # 其他参数
        super(GraphAE, self).__init__()

        # 将总层数分配给编码器和解码器
        # 编码器和解码器各占一半层数
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)  # 编码器层数（向下取整）
        decoder_layers = math.ceil(num_layers / 2)  # 解码器层数（向上取整）

        # 共享编码器：将节点特征 x 和边信息 edge_index 编码为低维嵌入
        # 输入：节点特征（维度 in_dim）
        # 输出：节点嵌入（维度 hid_dim）
        self.shared_encoder = backbone(in_channels=in_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)

        # 属性解码器：从嵌入重构节点特征
        # 输入：节点嵌入（维度 hid_dim）
        # 输出：重构的节点特征（维度 in_dim）
        self.attr_decoder = backbone(in_channels=hid_dim,
                                     hidden_channels=hid_dim,
                                     num_layers=decoder_layers,
                                     out_channels=in_dim,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)

        # 结构解码器：从嵌入重构图结构（邻接矩阵）
        # 使用点积解码器，通过节点嵌入的点积来预测边是否存在
        self.struct_decoder = DotProductDecoder(in_dim=hid_dim,
                                                hid_dim=hid_dim,
                                                num_layers=decoder_layers - 1,
                                                dropout=dropout,
                                                act=act,
                                                sigmoid_s=sigmoid_s,
                                                backbone=backbone,
                                                **kwargs)

        # 损失函数：同时计算属性重构损失和结构重构损失
        self.loss_func = double_recon_loss
        self.emb = None  # 存储编码后的嵌入

    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x: 节点特征矩阵 [num_nodes, in_dim]
            edge_index: 边的索引 [2, num_edges]
        
        返回:
            x_: 重构的节点特征
            s_: 重构的图结构（邻接矩阵）
            self.emb: 节点嵌入
        """
        # 编码：将节点特征和结构编码为嵌入
        self.emb = self.encode(x, edge_index)
        # 解码：从嵌入重构节点特征和结构
        x_, s_ = self.decode(self.emb, edge_index)
        return x_, s_, self.emb
    
    def encode(self, x, edge_index):
        """
        编码函数：将节点特征编码为低维嵌入
        
        参数:
            x: 节点特征矩阵
            edge_index: 边的索引
        
        返回:
            节点嵌入矩阵 [num_nodes, hid_dim]
        """
        self.emb = self.shared_encoder(x, edge_index)
        return self.emb
    
    def decode(self, emb, edge_index):
        """
        解码函数：从嵌入重构节点特征和图结构
        
        参数:
            emb: 节点嵌入矩阵
            edge_index: 边的索引（用于结构解码）
        
        返回:
            x_: 重构的节点特征
            s_: 重构的图结构（邻接矩阵）
        """
        # 重构节点特征
        x_ = self.attr_decoder(emb, edge_index)
        # 重构图结构
        s_ = self.struct_decoder(emb, edge_index)
        return x_, s_
