# Confidence-Aware Wan Future 特征抽取说明

本文档说明当前 DiffSynth-Studio 中 Wan2.2-TI2V-5B LIBERO future teacher 的 confidence-aware 特征抽取实现。它对应顶层的 `confidence_aware_teacher_distillation.md`，只负责 Wan future（F）通道的离线 teacher cache，不修改 Wan 微调、视频质量验证、StarVLA alignment loss 或部署推理。

## 已实现内容

`examples/wanvideo/model_inference/extract_libero_future_features.py` 保留两条路径：

- legacy 路径：每个 `(suite, episode, anchor)` 执行一次教师推理，写入原有 `features`、`anchor_frames` 和 `valid_future_frames`。
- confidence-aware 路径：使用 `--confidence-aware --num-teacher-samples 4`。同一 anchor 复用相同的输入图像、prompt、分辨率、17-frame rollout、CFG、denoising step、block 和 positive branch，只改变显式 diffusion seed。

每次推理结果到达后立即转成 CPU float32，并加入当前 anchor 的流式累加器。抽取器不保存 K 份完整 feature，只保留：

```text
feature_sum
normalized_sum
normalized_sqnorm_sum
sample_count
```

最终 target 是四次输出的均值，写盘时转换回现有 `bfloat16`。统计使用最后一维 LayerNorm 和 L2 normalization；先计算每个 token 的 pairwise cosine disagreement，再对 token 取 median，得到每个 anchor 一个 raw disagreement scalar。

## Seed 协议

第 `sample_index` 次推理使用：

```text
(base_seed
 + suite_id * 100000000
 + episode_index * 1000003
 + anchor_frame * 9176
 + sample_index * 97531) % 2**31
```

默认 `base_seed=42`。`suite_id` 必须参与 seed，避免不同 suite 中相同 episode index 发生碰撞。所有 sample seed 会写入 episode metadata，便于中断后复算和审计。

## 缓存格式

legacy 目录保持不变：

```text
wan2_2_future_latent_h16_step20_block20z_stride8_480x480
```

confidence-aware 默认目录为：

```text
wan2_2_future_latent_h16_step20_block20z_stride8_480x480_samples4_confv1
```

每个 suite 下仍按 LeRobot `chunks_size` 保存：

```text
<suite-root>/wan2_2_future_latent_h16_step20_block20z_stride8_480x480_samples4_confv1/
  chunk-000/
    episode_000000.safetensors
  ...
```

每个 confidence-aware episode 文件包含：

```text
features              [A, 1125, 3072] bfloat16  # K 次 teacher 输出的均值
anchor_frames         [A]             int64
valid_future_frames   [A]             int16
confidence            [A]             float32   # 校准后的 [0, 1]
disagreement          [A]             float32   # raw disagreement
```

其中 `A = ceil(episode_length / 8)`，第一个 anchor 必须是 0，后续 anchor 间隔为 8 帧。confidence 和 disagreement 都是每个 anchor 一个 scalar，不是 token-level tensor。

confidence-aware metadata 至少包括：

```text
confidence_aware=true
confidence_version=v1
extraction_complete=true
num_teacher_samples=4
target_aggregation=mean
uncertainty_metric=pairwise_cosine_disagreement
feature_normalization=layernorm_last_dim
uncertainty_normalization=l2_last_dim
uncertainty_aggregation=median_token
seed_policy=explicit_per_anchor
calibration=train_quantile_05_95
```

同时保留原有 teacher、checkpoint、episode、shape、stride、step、block、CFG、token order 和 dtype metadata。

## Confidence 校准

当前 LIBERO 流程中所有发现的 episode 都作为训练数据使用，没有独立 validation split。因此无过滤的全量运行会收集全部 episode 的 `disagreement`，计算全局 q05/q95：

```text
r = clip((disagreement - q05) / (q95 - q05 + 1e-8), 0, 1)
confidence = 1 - r
```

校准参数写入 `confidence_manifest.json`，内容包括：

```text
confidence_aware
confidence_version
num_teacher_samples
calibration
calibration_scope=all_training_episodes
q_low
q_high
eps
episode_count
anchor_count
checkpoint_path
checkpoint_sha256
```

不能对单独 shard 使用局部分位数作为正式 confidence。带 `--suites`、`--episode-start`、`--episode-end` 或 `--episode-indices` 的运行只负责抽取 feature/disagreement，不进行全局校准。

## 中断恢复与跳过规则

抽取使用同目录 `.tmp` 文件写入，验证通过后通过 `os.replace()` 原子替换正式文件。重新运行相同命令时：

- 文件不存在、为空、损坏、shape/dtype 错误、anchor 错误或 checkpoint metadata 不匹配：重新抽取该 episode；
- confidence-aware pending 文件（`complete=false`, `calibration=pending`）在普通抽取重启时可直接跳过，不重复运行 K 次 teacher；
- 已完成文件（`complete=true`, `calibration=train_quantile_05_95`）直接跳过；
- `--overwrite` 强制重新抽取所有选中的文件；
- 如果进程在单个 episode 完成前中断，该 episode 按文件粒度整体重算，不支持从单个 anchor 的中间状态继续。

无过滤的 confidence-aware 全量命令会在抽取阶段结束后检查 `confidence_manifest.json`：

- manifest 和所有 episode 都已 finalized：只校验并跳过，不重复校准；
- manifest 缺失或任一 episode 未 finalized：重新收集 raw disagreement 并原子重写全部 episode。

因此，分片任务中断后可以直接重启同样的命令；所有合法 safetensor 会被跳过，剩余 episode 才会加载 Wan 模型并执行推理。

## 运行方式

### 单 episode smoke

```bash
python examples/wanvideo/model_inference/extract_libero_future_features.py \
  --data-root "${DATA_ROOT}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --checkpoint-type lora \
  --confidence-aware \
  --num-teacher-samples 4 \
  --smoke-test \
  --suites libero_goal \
  --episode-indices 125 \
  --output-root "${RUN_ROOT}/smoke/features" \
  --progress-dir "${RUN_ROOT}/smoke/extraction_progress"
```

### 多卡正式抽取

推荐使用入口脚本：

```bash
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
CHECKPOINT_TYPE=lora \
VALIDATION_REPORT="${VALIDATION_REPORT}" \
RUN_ROOT="${RUN_ROOT}" \
bash examples/wanvideo/model_inference/run_extract_libero_future_features.sh
```

入口脚本默认添加：

```text
--confidence-aware
--num-teacher-samples 4
```

正式抽取前仍要求 validation report 的 loss/video 双门禁通过。输出默认直接写入四个 suite 的数据集根目录；进度写入 `${RUN_ROOT}/extraction_progress/`。

## 审计

不加载 Wan 模型的完整性审计：

```bash
python examples/wanvideo/model_inference/extract_libero_future_features.py \
  --data-root "${DATA_ROOT}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --checkpoint-type lora \
  --confidence-aware \
  --num-teacher-samples 4 \
  --audit-only \
  --progress-dir "${RUN_ROOT}/audit_progress"
```

审计会检查 tensor key、shape、dtype、anchor、valid future、finite value、confidence 范围、complete 状态、校准 metadata 和 checkpoint hash。正式 confidence-aware 数据要求所有 episode 都是 finalized；pending 文件只能用于恢复阶段，不能交付给训练 loader。

同时检查临时文件：

```bash
find "${DATA_ROOT}" \
  -path '*wan2_2_future_latent_h16_step20_block20z_stride8_480x480_samples4_confv1*' \
  -type f -name '*.tmp' -print
```

正常情况下没有输出。

## 与上层训练的边界

本项目只生成：

1. 均值 Wan future teacher feature；
2. raw disagreement；
3. calibration 后的 confidence；
4. 可复现和可审计的 safetensors metadata。

StarVLA 消费端仍需自行实现 confidence-aware loader 和 alignment loss weighting。confidence 不输入 Wan、Qwen backbone 或 AISER，也不增加部署推理依赖。旧缓存只适用于 legacy/equal-weight 对照，开启 confidence-aware alignment 时必须使用新目录并严格检查新增 tensor 和 metadata。

## 验证状态与限制

已覆盖：

- 流式均值和 disagreement 公式的合成 tensor 验证；
- confidence/disagreement shape、dtype、finite 和 `[0, 1]` 校验；
- pending/finalized cache 的状态转换；
- atomic write、resume 和 overwrite 路径的静态检查；
- shell 语法和 Python 编译检查。

真实 Wan 抽取仍需要集群 CUDA、已通过视频验证的 checkpoint 和完整 LIBERO 数据。完整 unittest 应在项目 Python 3.11 集群环境执行；当前开发环境缺少与 Torch 匹配的 torchvision，无法在本地完成模型相关导入测试。
