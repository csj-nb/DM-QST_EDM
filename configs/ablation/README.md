# 逐级消融配置集（修正建议落地）

> 背景：`configs/n2_lowshot.yaml` 是 v3 **全叠加**态（表示层 5 件套 + 似然三级 + 损失 2 项）。
> 全开意味着结果无法归因、失败无法定位。本目录把技术线**拆成阶梯**，
> 每级只在前一级基础上加一项，逐级跑即可得到论文的消融表。

## 阶梯定义（每级 = 上一级 + 一项新技术）

| 配置 | 技术线 | 验证目标 |
|------|--------|----------|
| `stage0_baseline.yaml` | arcsin + counts（最 solid 表示层），其余全关 | **干净基线**：EDM vs MLE 的第一组数字 |
| `stage1_shot_channel.yaml` | + shot 通道（73 维） | shot 预算特征的独立增量 |
| `stage2_fid_loss.yaml` | + fidelity 辅助损失 | 增量 + **纯度偏置检查**（见下） |
| `stage3_meas_loss.yaml` | + measurement-consistency 损失 | 训练端似然的独立增量 |
| `stage3b_nll.yaml` | **stage3 的 NLL 变体**（clipped NLL 替代 masked-L2） | masked-L2 vs NLL 对比（见下） |
| `stage4_full_ladder.yaml` | + DPS + IW-MMSE | 采样/评估端似然（= v3 全开） |

保留在全部 stage 的公共基础（非"新技术"，是数据/训练必需）：
- `use_arcsin_sqrt` / `use_counts_channel`（表示层核心）
- `use_variable_shots` + log-uniform 采样（shot 覆盖必需）
- `use_loss_reweighting`（ERDM 重加权，成熟技术）
- sigma 校准值（sigma_max 1.22 等）
- `cfg_mode: none`（stage4 用 DPS 覆盖 CFG，CFG 本身不作为消融项）

## 运行方式（每个 stage 一次完整循环）

```bash
# 1. 生成数据（5k 态小规模信号验证）
python experiments/generate_data.py --config configs/ablation/stage0_baseline.yaml \
  --n_train 5000 --data_dir ./data_abl_stage0

# 2. 训练（200 epoch 左右，看 val 收敛）
python experiments/train.py --config configs/ablation/stage0_baseline.yaml \
  --data_dir ./data_abl_stage0 --output_dir ./outputs/abl_stage0 --device cpu

# 3. 评估（[50,100,300] shots 对比 MLE）
python experiments/evaluate.py --config configs/ablation/stage0_baseline.yaml \
  --data_dir ./data_abl_stage0 --ckpt ./outputs/abl_stage0/checkpoints/best.pt \
  --n_states 100 --skip_mle false --skip_linear true
```

> 注意：`train.py` / `evaluate.py` 的确切 CLI 参数以脚本 `--help` 为准；
> 关键是**每个 stage 用独立的 data_dir / output_dir**，避免缓存串扰。

## 消融表记录（每级填一行）

| stage | 50 shots | 100 shots | 300 shots | vs 上一级增量 | 备注 |
|-------|----------|-----------|-----------|---------------|------|
| 0 baseline | | | | — | EDM vs MLE 第一组数字 |
| 1 +shot | | | | | |
| 2 +fid | | | | | 检查纯度偏置 |
| 3 +meas | | | | | |
| 4 full | | | | | = v3 全开 |

## 三个必须的伴随检查

**stage3b 专用**：stage3（masked-L2）与 stage3b（clipped NLL）对比时，
NLL 的 λ 已重标定为 0.03（log 尺度 ≠ 平方尺度）；若训练不稳，扫描
{0.01, 0.03, 0.1}。NLL 对"观测到但预测概率低"的事件惩罚更重——若
低 shot 档位 fidelity 反而下降，说明 NLL 在过拟合 shot 噪声，保留
masked-L2。

1. **MLE 收敛性**（结果可信度命门，跑消融前先做一次）：
   ```bash
   python experiments/check_mle.py --n-states 30
   # 期望：300→~0.93, 1000→~0.97, 10000→~0.99（单调收敛，VERDICT: PASS）
   ```

2. **stage2 的纯度偏置检查**：`lambda_fid` 与已禁用的 `lambda_rank` 同源
   （都激励纯态）。stage2 评估时额外记录预测态纯度 vs 真态纯度：
   ```bash
   python experiments/diagnose_posterior.py --ckpt ./outputs/abl_stage2/checkpoints/best.pt \
     --n-states 30 --shots 100,300 --K 20
   # 看 diversity.json 里预测态纯度分布；若明显高于真态纯度 => lambda_fid 引入偏置
   ```

3. **stage3 的噪声过拟合检查**：`lambda_meas` 把观测频率当目标，低 shot 时
   观测本身是噪声。stage3 评估时对比 stage2：若 50 shots 档位 fidelity 反而
   下降，说明训练端似然在低 shot 过拟合 shot 噪声。
