# DM-QST: Denoising Diffusion Probabilistic Models for Quantum State Tomography

将经典去噪扩散概率模型（DDPM）应用于量子态层析（QST）的探索性研究项目。

## 核心思路

1. **Cholesky 表示**：密度矩阵 ρ = LL†/Tr(LL†)，自动保证物理约束（半正定、迹为1）
2. **条件扩散模型**：在 Cholesky 空间中运行 DDPM，以 Pauli 测量结果为条件
3. **合成训练数据**：使用 Qiskit/NumPy 生成多种类型的随机量子态

## 环境配置

```bash
# 创建 conda 环境
conda env create -f environment.yml
conda activate dm-qst

# 或使用 pip
pip install -e .
```

## 快速验证

```bash
# 运行所有单元测试
python tests/test_cholesky.py
python tests/test_measurements.py
python tests/test_diffusion.py

# 运行端到端验证（1 qubit, 少量数据）
python experiments/quick_validate.py
```

## 使用方法

### 1. 生成训练数据

```bash
python experiments/generate_data.py --n_qubits 2 --n_train 50000 --data_dir ./data
```

### 2. 训练模型

```bash
python experiments/train.py --config configs/default.yaml --n_qubits 2
```

### 3. 评估模型

```bash
python experiments/evaluate.py \
  --checkpoint outputs/abl_stage0/checkpoints/best.pt \
  --n_states 100 \
  --ddim_steps 100
```

## 项目结构

```
DM-QST/
├── configs/              # 配置文件
├── src/
│   ├── representation/   # Cholesky 表示 + 约束验证
│   ├── data/             # 随机态生成 + 测量模拟 + Dataset
│   ├── models/           # 噪声调度 + 条件网络 + UNet + DDPM
│   ├── training/         # 损失函数 + 训练器
│   ├── evaluation/       # 指标 + MLE基线 + 可视化
│   └── utils/            # 配置管理
├── experiments/          # 训练/评估/消融脚本
├── tests/                # 单元测试
└── notebooks/            # Jupyter 分析笔记本
```

## 量子态类型

| 类型 | 描述 | 生成方法 |
|------|------|---------|
| `pure_haar` | Haar 随机纯态 | i.i.d. 复高斯 + 归一化 |
| `mixed_hs` | Hilbert-Schmidt 混合态 | G G†/Tr(G G†), G 复高斯 |
| `mixed_ginibre` | Ginibre 系综 | G G†/Tr(G G†), G 为 d×r 矩阵 |
| `thermal` | 热态 | exp(-βH)/Tr(exp(-βH)) |
| `product` | 积态 | Bloch 球均匀采样 + 张量积 |
| `rqc` | 随机电路输出态 | Qiskit `random_circuit` + `Statevector`（需 qiskit，深度可用 `set_rqc_depth_range` 配置） |

## 真实噪声数据（借鉴 DD-QST）

测量模拟支持两类真实噪声（`--use-noise` / `--noise-model`）：

- **Fake 后端**（推荐，无需 IBM 账号）：`get_fake_backend_noise_model('FakeTorino')`
  复现 IBM 真实设备校准噪声（门误差 + T1/T2 + 读出误差）。注意 127+ qubit 的
  Fake 后端模拟较慢（约 3.6 s/态），适合小规模数据集或测试集。
- **轻量噪声模型**：`get_realistic_noise_model()`，基于典型 transmon 参数的
  depolarizing + readout 噪声，速度快，适合大规模训练集。
- **真实 IBM 设备**：`get_ibm_noise_model('ibm_brisbane')`（需 IBM Quantum 账号）。

用法：

```bash
# RQC 态 + FakeTorino 真实设备噪声
python experiments/generate_data.py --config configs/default.yaml --use-noise \
  --noise-model FakeTorino --n_train 500 --data_dir ./data_rqc_noisy
```

## 低 shot 表示（方案 1：arcsin-sqrt + counts 双通道）

低 shot（100-1000 shots）时频率稀疏、Fisher z 在 0/1 处 clip 且方差爆炸。
方案 1 从表示层缓解：

- **arcsin-sqrt 变换**：`z = arcsin(√p)`，对二项分布方差恒定（Var ≈ 1/(4N)），
  0/1 处良定义、无 clip 信息损失；
- **counts 通道**：条件向量拼接 log1p 归一化的原始 counts，保留 shot 预算
  （置信度）这一频率丢弃的充分统计量。条件维度 6^n → 2·6^n。

配置：`configs/n2_lowshot.yaml`（`use_arcsin_sqrt: true` + `use_counts_channel: true`）
与 `configs/n2_edm.yaml`（Fisher z，无 counts 通道）形成低 shot 消融对比。

```bash
# 生成方案 1 数据集
python experiments/generate_data.py --config configs/n2_lowshot.yaml --n_train 5000

# 训练
python experiments/train.py --config configs/n2_lowshot.yaml
```

## 评估指标

- **Fidelity**: F(ρ, σ) = [Tr √(√ρ σ √ρ)]²
- **Trace distance**: T(ρ, σ) = ½ Tr |ρ - σ|
- **Purity**: Tr(ρ²)

## 参考文献

- VDM for QST (IFAC 2023): Learning Quantum Distributions with Variational Diffusion Models
- CCMQD (arXiv:2511.12221): Channel-Constrained Markovian Quantum Diffusion
- Structure-Preserving Diffusion (arXiv:2404.06336): Mirror maps for density matrix constraints
- QuaDiM (ICLR 2025): Conditional diffusion for quantum property estimation
- Ho et al. (2020): Denoising Diffusion Probabilistic Models
- Nichol & Dhariwal (2021): Improved Denoising Diffusion Probabilistic Models

## License

MIT

## 开发日志

详见 [CHANGELOG.md](CHANGELOG.md) 记录每次代码更新和修复。
