# FMGAD 框架说明

本文档概括本仓库 **FMGAD（Flow Matching + Graph AutoEncoder for Graph Anomaly Detection）** 的代码结构、数据流与训练流程，便于二次开发与接新数据集。

**论文 Method 用（公式与理论表述）**：见同目录 [`METHOD_PAPER.md`](./METHOD_PAPER.md)。

---

## 1. 定位与思路

- **任务**：无监督 / 半监督设定下的 **图异常检测**；评估使用节点级真值标签 `y ∈ {0,1}`（0 正常、1 异常）。
- **核心思想**：
  1. 用 **图自编码器（GraphAE）** 学习节点嵌入，并结合 **结构残差**（全局 / 局部等）拼成混合表征 **`z = [h, r]`**。
  2. 在 **`z` 空间** 上训练 **Flow Matching**（速度场拟合噪声→数据的插值路径）：先训练 **无条件** 流，再训练 **以原型嵌入为条件** 的流。
  3. 推断时用 **生成 / 重构相关的异常分数**（默认基于采样与双模型对比），输出 AUC、AP 等指标（借助 **PyGOD** 的 `eval_*`）。

与「仅用 AE 重构误差」或「纯扩散」不同，本实现强调 **残差增强的 latent** + **流模型在 latent 上的密度 / 动力学建模**。

---

## 2. 目录与模块职责

| 路径 | 职责 |
|------|------|
| `main_train.py` | **主入口**：读 YAML、`CUDA_VISIBLE_DEVICES`、构造 `ResFlowGAD`、调用 `model(dataset_name)`、可选写 `result-file` JSON。 |
| `res_flow_gad.py` | **核心类 `ResFlowGAD`**：数据加载、`forward` 全流程（AE → Flow → 评估）、大图分支、多种评分开关。 |
| `auto_encoder.py` | **GraphAE**：GCN 编码器 + 属性 / 结构解码（与 PyGOD 解码与损失衔接）。 |
| `encoder.py` | **残差与门控**：`compute_dual_residuals_with_degree`、多尺度残差、`ResidualChannelAttention`、自适应 `residual_scale` 等。 |
| `flow_matching_model.py` | **Flow Matching**：`MLPFlowMatching`、`FlowMatchingModel`、`FlowMatchingLoss`、`sample_flow_matching_free` 等。 |
| `FMloss.py` | Flow Matching 训练用损失封装（与 `res_flow_gad` 内训练循环配合）。 |
| `utils.py` | 工具函数（如温度 softmax 等）。 |
| `load_custom_data.py` | **特殊数据**：`dgraph` / `dgraphfin` 等 npz / PyGOD 加载。 |
| `configs/*.yaml` | 数据集名、AE / Flow / 机制开关、训练 trial 数等。 |
| `tuning_search_space.py` | **调参空间**：`get_fixed_overrides`、`get_refined_search_space`、`get_detailed_search_space`（按数据集名分支）。 |
| `run_tune_refined.py` | **多 GPU 精细调参**：子进程调 `main_train.py`，写 `tuning_runs.json`、`best_by_dataset.json` 等。 |
| `tune_hyperparams.py` | 另一套调参入口（与 refined / detailed 口径在 `tuning_search_space` 中统一）。 |
| `run_bestcfg_multiseed_sweep.py` | 按 `tuning_runs.json` 最优 `full_config` 在多个 seed 上复跑。 |
| `merge_multiseed_into_combined_report.py` | 将单次 multiseed 输出合并进总 Markdown 报告。 |

---

## 3. 数据约定

- 统一使用 **PyG `Data`**：
  - `x`：节点特征 `[N, F]`
  - `edge_index`：`[2, E]`，COO
  - `y`：节点标签，**二分类** long/float，语义为 **0=正常，1=异常**
- **加载入口**：`ResFlowGAD._load_dataset(dset)`  
  - 内置：`yelpchi`、`elliptic`、`twitter`、`twibot20/22`、`dgraph`、`dgraphfin`、`questions`（Heterophilous）等。  
  - 其余名字默认走 **`pygod.utils.load_data(dset)`**（若 PyGOD 支持）。
- **大图**：当节点数 **> 15000** 时置 `_large_graph`，避免对全图建 **稠密邻接**（训练 AE 时走稀疏路径），与 YelpChi / Questions 等规模相匹配。

---

## 4. 训练与推断流水线（`ResFlowGAD.forward`）

以下为默认配置下的逻辑顺序（具体分支受 YAML 中布尔开关控制）。

1. **加载数据** → 定 `hid_dim`（若未指定则随特征维自适应）、设 `_large_graph`。
2. **训练 GraphAE**（`_train_ae_once`）：重构特征与结构；得到 `encode` 用于后续 **`h`**。
3. **构造 `z`**（`_build_z`）：
   - `h = AE.encode(x, edge_index)`；
   - 可选 **虚拟邻居**（在嵌入空间 KNN 加边）；
   - **双残差或多尺度残差** + **度相关门控** + 可选 **自适应 residual 缩放**；
   - `z = cat([h, r], dim=1)`，维度 **`2 * hid_dim`**。
4. **Trial 循环**（次数 = `num_trial`）：
   - 训练 **无条件 Flow**（`_train_dm_free`），得到 **`dm`** 与 **原型相关状态**（如 `prototype`、门控等 checkpoint）；
   - 训练 **条件 Flow**（`_train_dm_proto`，条件为 **`h` 空间原型**），得到 **`dm_proto`**；
   - **评估**：默认走 `sample(dm_proto, dm, data)`；可选 NLL / energy / guided recon 等分支。
5. **多 trial 集成**：若 `ensemble_score`，可对各 trial 分数 **平均** 后再算 AUC/AP。
6. **返回**字典：含 `auc_mean`、`ap_mean` 等（供 `main_train` 写入 JSON）。

---

## 5. 配置与调参

- **单次实验**：`python main_train.py --config configs/<name>.yaml --device 0 --seed 42 --num_trial 1`  
  - `dataset` 在 YAML 的 `dataset` 字段；`--num_trial` 可覆盖 YAML。
- **机制开关**：如 `use_multi_scale_residual`、`use_virtual_neighbors`、`flow_t_sampling`、`weight`、`proto_alpha` 等均来自 YAML，传入 `ResFlowGAD` 构造函数。
- **调参**：
  - 搜索空间与 **数据集固定锁** 在 `tuning_search_space.py`（避免无效模块进网格）。
  - `run_tune_refined.py` 负责并行、超时、`tmp` 配置与结果汇总。

---

## 6. 依赖与环境要点

- **PyTorch** + **PyTorch Geometric**（图与 `Data`）。
- **PyGOD**：`load_data`（部分数据集）、`eval_*` 指标、`DotProductDecoder` / `double_recon_loss` 等与 AE 衔接。
- **torchdiffeq**：流模型 ODE / 采样相关（见 `res_flow_gad` 引用）。
- 缓存与数据根目录常通过环境变量 **`FMGAD_MODEL_ROOT`**、`TMPDIR`、`XDG_CACHE_HOME`**（调参脚本中为多进程隔离会单独设置）。

---

## 7. 扩展新数据集（Checklist）

1. 实现 **`_load_xxx` → `Data(x, edge_index, y)`**，`y` 为二分类；在 `_load_dataset` 中注册名字。
2. 在 **`configs/xxx.yaml`** 中设置 `dataset` 与超参初值。
3. 若参与调参，在 **`tuning_search_space.py`** 中为该名字增加 `get_fixed_overrides` / 搜索空间分支（或走 **Discovery** 默认分支）。
4. 若节点数很大，确认 **>15000** 时行为符合预期（稀疏邻接路径）。

---

## 8. 相关文档与脚本（可选）

- 批量调参、复现 YAML 导出：`export_repro_yamls_from_tune_dir.py`、`build_repro_config_bundle.py` 等。
- 消融：`run_ablation.py`。

---

*文档版本：与仓库 `res_flow_gad.py` / `main_train.py` 当前结构对应；若接口变更，请同步更新本节。*
