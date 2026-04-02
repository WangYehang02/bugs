# FMGAD：论文 Method 可用的理论表述（含公式）

下文采用无监督图异常检测常见设定，符号与当前实现（`res_flow_gad.py`、`encoder.py`、`FMloss.py`、`flow_matching_model.py`、`auto_encoder.py`）一致，可直接裁剪、翻译后用于论文 **Method** 小节。若需与代码逐行对齐，括号内标注函数或文件名。

---

## 1. 问题形式化

给定无向图 \(\mathcal{G}=(\mathcal{V},\mathcal{E})\)，节点特征矩阵 \(\mathbf{X}\in\mathbb{R}^{N\times F}\)，邻接以 COO 形式 \(\mathcal{E}\) 表示。训练阶段**不使用**节点异常标签；推断时为每个节点 \(i\) 输出异常分数 \(s_i\in\mathbb{R}\)，评估阶段用二值真值 \(y_i\in\{0,1\}\)（0 正常、1 异常）计算 AUC / AP 等。

记 \(\mathbf{A}\in\{0,1\}^{N\times N}\) 为（可选构造的）邻接矩阵；**大图**（\(N>15000\)）实现中**不**实例化稠密 \(\mathbf{A}\)，仅用稀疏消息传递与属性重构分支（`_large_graph`）。

---

## 2. 总体框架（两阶段）

1. **结构感知编码**：图自编码器将 \((\mathbf{X},\mathcal{E})\) 映射为嵌入 \(\mathbf{H}\in\mathbb{R}^{N\times d}\)，并在 \(\mathbf{H}\) 上构造结构残差 \(\mathbf{R}\)，拼接得混合表征 \(\mathbf{Z}\in\mathbb{R}^{N\times 2d}\)。
2. **流匹配建模**：在 \(\mathbf{Z}\) 空间学习**无条件**与**原型条件**两类速度场；推断时从噪声经数值积分得到 \(\hat{\mathbf{Z}}\)，再解码为特征重构误差，并与辅助分数融合得到 \(s_i\)。

---

## 3. 图自编码器（Graph AE）

编码器为 \(L/2\) 层 GCN（实现中为 PyG `GCN`），共享编码得到
\[
\mathbf{H} = \mathrm{Enc}(\mathbf{X},\mathcal{E})\in\mathbb{R}^{N\times d}.
\]

解码器含**属性分支** \(\mathrm{Dec}_x\) 与**结构分支** \(\mathrm{Dec}_s\)（点积解码等，与 PyGOD 接口一致）。对小图，重构 \(\hat{\mathbf{X}}\)、\(\hat{\mathbf{A}}\) 与监督信号组合为（标量化的节点级损失再平均，实现为 `double_recon_loss` 与权重 \(\alpha\)）：
\[
\mathcal{L}_{\mathrm{AE}}
= \frac{1}{N}\sum_{i=1}^{N}
\ell_{\mathrm{recon}}\big(\mathbf{x}_i,\hat{\mathbf{x}}_i,\mathbf{A}_{i:},\hat{\mathbf{A}}_{i:};\alpha\big).
\]
**大图**下省略稠密 \(\mathbf{A}\)，仅用 \(\|\mathbf{x}_i-\hat{\mathbf{x}}_i\|_2\) 型重构项（`_train_ae_once`）。

---

## 4. 结构残差与混合 Latent \(\mathbf{Z}\)

### 4.1 双残差（默认）

对节点 \(i\)，令 \(\bar{\mathbf{h}}=\frac{1}{N}\sum_j \mathbf{h}_j\) 为全图均值嵌入，\(\mathcal{N}(i)\) 为一跳邻居（按 `edge_index` 方向聚合），度 \(d_i=|\mathcal{N}(i)|\)（实现中孤立点度截断为 1 避免除零）。

- **全局残差**（偏离全局分布）  
  \[
  \mathbf{r}^{\mathrm{glob}}_i = \mathbf{h}_i - \bar{\mathbf{h}}.
  \]
- **局部残差**（偏离邻居均值）  
  \[
  \mathbf{r}^{\mathrm{loc}}_i = \mathbf{h}_i - \frac{1}{d_i}\sum_{j\in\mathcal{N}(i)}\mathbf{h}_j.
  \]

**度门控**融合（可学习标量偏置 \(b\) 与锐度 \(\kappa\)）：
\[
\alpha_i = \sigma\big(\kappa(d_i-b)\big),\qquad
\tilde{\mathbf{r}}_i = \alpha_i\,\mathbf{r}^{\mathrm{loc}}_i + (1-\alpha_i)\,\mathbf{r}^{\mathrm{glob}}_i,
\]
其中 \(\sigma\) 为 logistic。可选**自适应幅度** \(\phi(d_i)\) 作用于 \(\tilde{\mathbf{r}}_i\)，再乘标量超参 \(\lambda\)（代码中 `residual_scale`）得 \(\mathbf{r}_i = \lambda\,\phi(d_i)\odot \tilde{\mathbf{r}}_i\)（关闭自适应时 \(\phi\equiv 1\)）。

### 4.2 多尺度残差（可选）

在 \(\mathbf{r}^{\mathrm{glob}},\mathbf{r}^{\mathrm{loc}}\) 之外增加结构相关通道 \(\mathbf{r}^{\mathrm{str}}\)，经通道注意力 \(\mathrm{Attn}(\cdot)\) 融合为 \(\tilde{\mathbf{r}}_i\)（`compute_multi_scale_residuals` + `ResidualChannelAttention`）。

### 4.3 虚拟邻居（可选）

在 \(\mathbf{H}\) 空间对高度节点按 KNN 追加边，再计算上述残差（`_add_virtual_knn_edges`）。

### 4.4 拼接与标准化

\[
\mathbf{z}_i = \big[\mathbf{h}_i \,\|\, \mathbf{r}_i\big] \in \mathbb{R}^{2d}.
\]
训练流模型前对 \(\mathbf{Z}\) 按列标准化并截断到 \([-M,M]\)（`_normalize_clip`），以提高数值稳定性。

---

## 5. 流匹配目标（Rectified / 直线路径）

设 \(\boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})\) 与数据点 \(\mathbf{z}\)（此处为单行 \(\mathbf{z}_i\) 或批）同维。采用**线性插值路径**：
\[
\mathbf{z}_t = (1-t)\boldsymbol{\varepsilon} + t\,\mathbf{z},\qquad t\in[0,1].
\]
该路径的**条件速度场**为常向量
\[
\mathbf{v}^*_t = \mathbf{z}-\boldsymbol{\varepsilon}.
\]

速度网络 \(\mathbf{v}_\theta(\mathbf{z}_t,t,\mathbf{c})\) 输入当前状态 \(\mathbf{z}_t\)、时间 \(t\) 与上下文 \(\mathbf{c}\)（时间嵌入 + MLP，见 `MLPFlowMatching`）。**流匹配损失**为 MSE：
\[
\mathcal{L}_{\mathrm{FM}}
= \mathbb{E}_{t,\boldsymbol{\varepsilon}}\Big[\big\|\mathbf{v}_\theta(\mathbf{z}_t,t,\mathbf{c})-(\mathbf{z}-\boldsymbol{\varepsilon})\big\|_2^2\Big],
\]
其中 \(t\) 的采样策略可为 \(\mathcal{U}(0,1)\) 或 **logit-normal** 再经 sigmoid，使边界附近时间步权重更大（`conditional_flow_matching_loss` 中 `t_sampling`）。

---

## 6. 无条件流与图级原型（Prototype）

**无条件**模型令 \(\mathbf{c}=\mathbf{0}\)（广播为与 batch 同形），学习边缘分布（`_train_dm_free` + `flow_matching_loss`）。

**原型** \(\mathbf{p}\in\mathbb{R}^{d}\) 仅在嵌入子空间 \(\mathbf{H}\) 上定义：训练过程中用流从噪声短步积分得到 \(\hat{\mathbf{H}}\)，再对 \(\mathbf{h}_i\) 与 \(\hat{\mathbf{h}}_i\) 的余弦相似度做温度 softmax 加权，更新 \(\mathbf{p}\)（指数滑动/加权重心形式，实现见 `_train_dm_free` 内 `proto_h` 更新）。

---

## 7. 原型条件流

第二支速度场 \(\mathbf{v}^{\mathrm{proto}}_\theta\) 与 \(\mathbf{v}^{\mathrm{free}}_\theta\) 结构相同，但 MLP 输入中注入条件嵌入（`map_proto` / `proto_proj`），上下文取
\[
\mathbf{c}_i \equiv \mathbf{p}\quad(\text{广播到每个节点}).
\]
损失仍为 \(\mathcal{L}_{\mathrm{FM}}\)，时间采样由配置 `flow_t_sampling` 指定（`conditional_flow_matching_loss`）。

---

## 8. 推断：组合速度场与数值积分

给定噪声初值 \(\mathbf{z}^{(0)}\sim\mathcal{N}(\mathbf{0},\mathbf{I})\)，离散步长 \(\Delta t=1/K\)，**组合速度**（实现 `sample_flow_matching_free`）为
\[
\mathbf{v}^{\mathrm{comb}}_t = (1+w)\,\mathbf{v}^{\mathrm{free}}_\theta(\mathbf{z}_t,t,\varnothing) - w\,\mathbf{v}^{\mathrm{proto}}_\theta(\mathbf{z}_t,t,\mathbf{p}),
\]
其中 \(w\ge 0\) 为超参（代码 `weight`）。前向 Euler 积分：
\[
\mathbf{z}_{t+\Delta t} = \mathbf{z}_t + \Delta t\,\mathbf{v}^{\mathrm{comb}}_t,\qquad
\hat{\mathbf{Z}} = \mathbf{z}_{t=1}.
\]

将 \(\hat{\mathbf{Z}}\) 的前 \(d\) 维视为 \(\hat{\mathbf{H}}\)，经 AE 解码得 \(\hat{\mathbf{X}}\)（大图仅属性解码），**重构型分数**
\[
q^{\mathrm{recon}}_i = \ell_{\mathrm{recon},i}(\mathbf{X},\hat{\mathbf{X}}).
\]

---

## 9. 多源异常分数与图平滑（可选）

在 \(\mathbf{Z}_0=\mathbf{Z}\)（观测混合表征）上可预计算辅助标量：

- **原型距离**：\(q^{\mathrm{proto}}_i=\|\mathbf{h}_i-\mathbf{p}\|_2\)。
- **残差能量**：\(q^{\mathrm{res}}_i=\|\mathbf{r}_i\|_2\)。
- **速度范数（能量）**：取 \(t\approx 1\)，\(q^{\mathrm{ene}}_i=\|\mathbf{v}^{\mathrm{proto}}_\theta(\mathbf{z}_{0,i},t,\mathbf{p})\|_2\)。

各通道在节点上 min–max 归一化到 \([0,1]\) 后，以温度 Softmax 得权重 \(w_{i,k}\)，融合为
\[
s^{\mathrm{fuse}}_i = \sum_{k\in\mathcal{K}} w_{i,k}\,\tilde{q}^{(k)}_i,
\]
其中 \(\mathcal{K}\) 默认含 \(\{\mathrm{recon},\mathrm{ene},\mathrm{proto},\mathrm{res}\}\)（`use_multi_score_fusion`）。关闭融合时 \(s_i=q^{\mathrm{recon}}_i\)。

可选**图上分数平滑**（邻居加权一阶平滑，`_smooth_scores_by_graph`）：
\[
s_i \leftarrow (1-\beta)s_i + \beta\cdot \frac{1}{d_i}\sum_{j\in\mathcal{N}(i)} s_j.
\]

---

## 10. 多步数扫描与 Trial 集成

对若干积分步数 \(K\)（由 `timesteps` 与 `sample_steps` 派生）重复积分–打分，保留 **AUC 最优** 步数对应的 \(\mathbf{s}\)（`sample`）。多次随机 trial 可对分数取平均再评 AUC（`ensemble_score`），以降低流采样方差。

---

## 11. 与标准 Flow Matching 的对应关系

- 采用 **conditional flow matching** 中常见的 **高斯噪声到数据** 的直线桥（rectified path），目标速度闭式为 \(\mathbf{z}-\boldsymbol{\varepsilon}\)，训练为回归该速度场。
- **双速度场 + 凸组合型推断** \((1+w)\mathbf{v}^{\mathrm{free}}-w\mathbf{v}^{\mathrm{proto}}\) 可解释为：在无条件动力学与“绕原型”的条件动力学之间做代数混合，以放大正常流形与异常离群的差异（实现动机）；若写论文可表述为 **ensemble of velocity fields** 或 **prototype-guided rectification**。

---

## 12. 复杂度与实现注记

- **训练**：GCN 编码 \(O(|E|d)\) 量级；流网络为 MLP，对 \(N\) 个节点批处理；大图避免 \(O(N^2)\) 稠密邻接。
- **推断**：\(O(K\cdot N\cdot d_{\mathrm{mlp}})\) 量级积分；\(K\) 与 `sample_steps` 相关。

---

## 13. 你可直接粘贴的「小节标题」建议（英文稿）

- *Problem Formulation*  
- *Structure-Aware Encoder and Residual-Augmented Latent*  
- *Flow Matching on the Latent Manifold*  
- *Unconditional and Prototype-Conditional Velocity Fields*  
- *Inference via Combined Integration and Anomaly Scoring*  
- *(Optional) Multi-Cue Score Fusion and Graph Smoothing*

---

*说明：若论文需引用「标准 FM / diffusion」，建议明确写出路径 \(\mathbf{z}_t=(1-t)\boldsymbol{\varepsilon}+t\mathbf{z}\) 与目标 \(\mathbf{v}^*=\mathbf{z}-\boldsymbol{\varepsilon}\)，并注明原型条件通过 \(\mathbf{v}_\theta(\cdot,\mathbf{p})\) 注入；与当前代码一致。*
