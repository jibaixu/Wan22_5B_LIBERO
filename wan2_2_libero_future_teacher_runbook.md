# Wan2.2-TI2V-5B LIBERO Future Teacher 启动手册

本文档说明如何从 LIBERO manifest 构建开始，完成 Wan2.2-TI2V-5B 的 LoRA 或 full fine-tuning、checkpoint 验证、teacher feature 抽取、断点续跑和完整性审计。

### Confidence-aware future teacher

正式 future teacher 抽取使用 `--confidence-aware --num-teacher-samples 4`。每个 anchor 的四次推理只改变显式 diffusion seed，抽取器以 float32 流式累加器保存均值 feature 和 token-level pairwise cosine disagreement，不写入四份中间 feature。分片运行会先写 `complete=false, calibration=pending` 的缓存；没有 episode 范围或索引过滤的运行会按 suite 分别从该 suite 全部训练 episode 的 disagreement 计算 q05/q95，并原子重写各自的 `confidence`、`complete=true` 及 suite-local `confidence_manifest.json`。不传 `--suites` 时仍一次处理四个 suite，但不会混合它们的 calibration；传入 `--suites` 时会校准每个选中 suite。

confidence-aware 缓存目录为 `wan2_2_future_latent_h16_step20_block20z_stride8_480x480_samples4_confv1`，新增 tensor 为 `confidence [A] float32` 和 `disagreement [A] float32`。旧目录仍保留为 legacy 单次 target，不能与 confidence-aware 缓存混用。

正式长训练和全量特征抽取建议在目标服务器执行。所有 checkpoint、日志和验证视频写入 `/data/jibaixu/Ckpts`，预训练模型写入 `/data/jibaixu/Models`，不要写入系统盘或仓库目录。

## 1. 启动入口总览

| 入口 | 目的 | 典型使用场景 |
| --- | --- | --- |
| `tools/build_libero_future_manifest.py` | 扫描四个 LIBERO suite，按 episode 分层划分 train/val，并生成 stride-4 在线窗口索引 | 第一次准备数据，或数据根目录/划分参数变化后重建 manifest |
| `examples/wanvideo/model_training/train_libero_future.py` | LIBERO 专用 Python 训练入口；加载在线 17 帧窗口并复用现有 Wan SFT loss | 通常由 LoRA/full shell 脚本调用，不建议手写全部参数 |
| `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-LIBERO.sh` | 以 8 GPU、micro-batch 1、gradient accumulation 4 训练默认 LoRA | 默认训练方案，优先使用 |
| `examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-LIBERO.sh` | 使用 DeepSpeed ZeRO-3 对 DiT 做 full fine-tuning | 容量对照实验；不阻塞 LoRA 主流程 |
| `examples/wanvideo/model_training/validate_libero_future_parallel.py` | 多 GPU 按 checkpoint 分片计算确定性 held-out flow loss，并选择 mean loss 最低 checkpoint | 多个 `step-*.safetensors` / `epoch-*.safetensors` 候选筛选时优先使用 |
| `examples/wanvideo/model_training/validate_libero_future.py` | 单进程 loss validation，并生成 base/tuned/GT 对照视频 | 调试小规模 loss，或对选中 checkpoint 执行视频门禁 |
| `examples/wanvideo/model_inference/extract_libero_future_features.py` | 单进程或 `torchrun` 下执行 feature 抽取、原子写入、resume 和 audit | 单卡 smoke、定向重算、调试或完整性审计 |
| `examples/wanvideo/model_inference/run_extract_libero_future_features.sh` | 封装 8 GPU 正式独立 episode 分片抽取 | validation 通过后的正式全量抽取 |
| `tests/test_libero_future_teacher.py` | 检查数据统计、窗口尾部 padding、capture branch/step/block 和缓存契约 | 迁移服务器后先执行的 CPU 回归测试 |

## 2. 环境和固定路径

进入仓库并激活环境：

```bash
cd /home/jbx/Codes/DiffSynth-Studio
source /home/jbx/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth-studio-py311
```

需要下载模型时，在交互 shell 中执行代理命令：

```bash
proxy
```

设置公共路径。后续命令默认复用这些变量：

```bash
export DATA_ROOT=/data/jibaixu/Datasets/LIBERO/libero_no_noop
export RUN_ROOT=/data/jibaixu/Ckpts/wan2_2_ti2v_5b_libero
export DIFFSYNTH_MODEL_BASE_PATH=/data/jibaixu/Models
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope

# 防止 Hugging Face/ModelScope 缓存落到系统盘。
export HF_HOME=/data/jibaixu/Ckpts/huggingface
export MODELSCOPE_CACHE=/data/jibaixu/Ckpts/modelscope
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

确认数据、GPU 和磁盘：

```bash
test -d "${DATA_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
df -h /data
```

## 3. CPU 回归测试

目的：在占用 GPU 前确认四个 suite 的统计、真实 AV1 解码、17 帧尾部 padding、feature capture 位置和 safetensors validator 均正常。

```bash
python -m unittest tests.test_libero_future_teacher -v
```

预期结果为 `Ran 8 tests ... OK`。ImageIO/FFmpeg 退出时可能打印 `ResourceWarning`，只要测试为 `OK` 且真实帧测试通过，不影响训练和抽取。

## 4. 构建 Manifest

目的：生成训练/验证在线窗口索引，不复制或预导出 68991 个窗口视频。划分以 episode 为单位，并按 `(suite, task)` 分层。

```bash
python tools/build_libero_future_manifest.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${RUN_ROOT}/manifests" \
  --train-stride 4 \
  --val-stride 4 \
  --val-ratio 0.1 \
  --split-seed 42 \
  --video-key observation.images.image \
  --num-frames 17
```

输出：

```text
${RUN_ROOT}/manifests/train.jsonl
${RUN_ROOT}/manifests/val.jsonl
${RUN_ROOT}/manifests/all.jsonl
${RUN_ROOT}/manifests/summary.json
```

检查固定总数：

```bash
python -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["totals"])' \
  "${RUN_ROOT}/manifests/summary.json"
```

必须包含：

```text
episodes=1693
all_windows=68991
stride8_anchors=34918
```

## 5. LoRA 训练分支（默认）

### 5.1 单 GPU 一步 smoke

目的：用一个真实 LIBERO 17 帧窗口执行完整 forward、backward、optimizer step 和 checkpoint 保存，提前发现模型加载、显存、数据解码或 loss 问题。

```bash
export CUDA_VISIBLE_DEVICES=0

NUM_PROCESSES=1 \
bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-LIBERO.sh \
  --max-samples 1 \
  --num_epochs 1 \
  --dataset_num_workers 0 \
  --save_steps 1 \
  --output_path "${RUN_ROOT}/smoke/lora_one_step"
```

预期输出：

```text
${RUN_ROOT}/smoke/lora_one_step/step-1.safetensors
${RUN_ROOT}/smoke/lora_one_step/run_metadata.json
${RUN_ROOT}/smoke/lora_one_step/training_args.json
```

训练入口会在 loss 出现 NaN/Inf 时直接失败，因此命令正常结束即表示本次 smoke loss finite。

### 5.2 正式 8 GPU LoRA SFT

目的：以文档锁定的默认配置训练 LoRA：3 epochs、LR `1e-4`、rank 32、每卡 micro-batch 1、累积 4 步，effective global batch 32。

先清除 smoke 时设置的单卡限制：

```bash
unset CUDA_VISIBLE_DEVICES
```

启动：

```bash
NUM_PROCESSES=8 \
bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-LIBERO.sh
```

输出目录：

```text
${RUN_ROOT}/lora_sft/
```

默认每 500 个 dataloader step 保存一个 `step-*.safetensors`，训练结束若不在保存边界也会保存最终 step。

## 6. Full Fine-Tuning 分支（可选）

### 6.1 Full smoke

目的：只验证 ZeRO-3 full fine-tuning 初始化、真实数据 forward/backward 和保存路径。full smoke 的资源消耗明显高于 LoRA。

```bash
bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-LIBERO.sh \
  --max-samples 8 \
  --num_epochs 1 \
  --dataset_num_workers 0 \
  --save_steps 1 \
  --output_path "${RUN_ROOT}/smoke/full_one_step"
```

### 6.2 正式 8 GPU full fine-tuning

目的：使用 DeepSpeed ZeRO-3 训练完整 DiT，作为 LoRA 的容量对照。配置为 2 epochs、LR `1e-5`、BF16、gradient accumulation 4，并在 CPU 初始化模型。

```bash
bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-LIBERO.sh
```

输出目录：

```text
${RUN_ROOT}/full_sft/
```

LoRA 与 full checkpoint 不能在同一次 validation 调用中混合。分别验证，并使用各自独立的 validation 输出目录。

## 7. Validation 与 checkpoint 选择

正式 feature extraction 有两个门禁：

1. `loss_passed=true`：选中 checkpoint 的 256-window deterministic validation loss 必须低于 base model，且所有 loss finite。
2. `video_passed=true`：完成 32 个 base/tuned/GT 20-step 对照，自动检查无完全静态复制，并由操作者确认没有普遍结构崩坏、闪烁或颜色坍缩。

只有最终报告中 `loss_passed`、`video_passed` 和 `passed` 全为 `true`，正式抽取入口才会继续。

以下先以 LoRA 为例。full checkpoint 的文件名和 validation 命令在 7.3 节单独给出。

### 7.1 确定性 validation loss

目的：在完全相同的 256 个 val windows、timesteps 和 noise 上比较 base 与所有候选 checkpoint，并自动选择 mean loss 最低者。默认使用 8 GPU checkpoint-parallel validation：rank 0 评估 base model，每个 rank 评估一部分 LoRA checkpoint，最后合并为与原流程兼容的 `validation_report.json` 和 `validation_summary.csv`。

```bash
torchrun --standalone --nproc_per_node=8 \
  examples/wanvideo/model_training/validate_libero_future_parallel.py \
  --val-manifest "${RUN_ROOT}/manifests/val.jsonl" \
  --output-dir "${RUN_ROOT}/validation/lora" \
  --checkpoint-type lora \
  --checkpoints "${RUN_ROOT}"/lora_sft/step-*.safetensors \
  --mode loss \
  --loss-samples 256 \
  --seed 42
```

若只调试少量 checkpoint 或只申请到单 GPU，可退化为原单进程入口：

```bash
python examples/wanvideo/model_training/validate_libero_future.py \
  --val-manifest "${RUN_ROOT}/manifests/val.jsonl" \
  --output-dir "${RUN_ROOT}/validation/lora" \
  --checkpoint-type lora \
  --checkpoints "${RUN_ROOT}"/lora_sft/step-*.safetensors \
  --mode loss \
  --loss-samples 32 \
  --seed 42
```

设置后续使用的报告和选中 checkpoint：

```bash
export CHECKPOINT_TYPE=lora
export VALIDATION_REPORT="${RUN_ROOT}/validation/lora/validation_report.json"
export CHECKPOINT_PATH="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"]["path"])' "${VALIDATION_REPORT}")"

echo "${CHECKPOINT_PATH}"
```

此时 `loss_passed` 应为 `true`，但在视频验证完成前总体 `passed` 仍为 `false`。

### 7.2 20-step 视频验证

目的：为固定的 32 个 val anchors 分别生成 base 和 tuned 结果，同时保存 GT，报告首帧保持、future/final-frame PSNR/SSIM 和 temporal change。

```bash
python examples/wanvideo/model_training/validate_libero_future.py \
  --val-manifest "${RUN_ROOT}/manifests/val.jsonl" \
  --output-dir "${RUN_ROOT}/validation/lora" \
  --checkpoint-type "${CHECKPOINT_TYPE}" \
  --selected-checkpoint "${CHECKPOINT_PATH}" \
  --mode video \
  --video-samples 32 \
  --seed 42 \
  --approve-video-quality
```

检查：

```text
${RUN_ROOT}/validation/lora/videos/index.html
${RUN_ROOT}/validation/lora/videos/metrics.json
${RUN_ROOT}/validation/lora/validation_report.json
```

`--approve-video-quality` 表示操作者承诺在启动正式抽取前检查完整的 32-sample index。若发现明显崩坏，即使报告已写出也不要执行抽取；移除该参数重新生成时，`video_passed` 会保持为 `false`。

最终门禁检查：

```bash
python -c 'import json,sys; x=json.load(open(sys.argv[1])); print({k:x[k] for k in ("loss_passed","video_passed","passed")})' \
  "${VALIDATION_REPORT}"
```

### 7.3 Full checkpoint 的对应验证命令

目的：对 full fine-tuning 产生的 `epoch-*.safetensors` 使用独立报告完成相同的 loss/video 门禁，避免和 LoRA checkpoint 混合比较。

先执行 loss validation。默认使用 8 GPU checkpoint-parallel validation：

```bash
torchrun --standalone --nproc_per_node=8 \
  examples/wanvideo/model_training/validate_libero_future_parallel.py \
  --val-manifest "${RUN_ROOT}/manifests/val.jsonl" \
  --output-dir "${RUN_ROOT}/validation/full" \
  --checkpoint-type full \
  --checkpoints "${RUN_ROOT}"/full_sft/epoch-*.safetensors \
  --mode loss \
  --loss-samples 256 \
  --seed 42
```

切换公共变量：

```bash
export CHECKPOINT_TYPE=full
export VALIDATION_REPORT="${RUN_ROOT}/validation/full/validation_report.json"
export CHECKPOINT_PATH="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"]["path"])' "${VALIDATION_REPORT}")"
```

再执行 32-sample 视频验证：

```bash
python examples/wanvideo/model_training/validate_libero_future.py \
  --val-manifest "${RUN_ROOT}/manifests/val.jsonl" \
  --output-dir "${RUN_ROOT}/validation/full" \
  --checkpoint-type full \
  --selected-checkpoint "${CHECKPOINT_PATH}" \
  --mode video \
  --video-samples 32 \
  --seed 42 \
  --approve-video-quality
```

后续 smoke、正式抽取和 audit 命令不需要改变；它们读取 `CHECKPOINT_TYPE`、`CHECKPOINT_PATH` 和 `VALIDATION_REPORT`。

## 8. Feature Extraction Smoke

目的：在正式 225 GiB 抽取前，用选中 checkpoint 验证真实 20-step rollout、step 19/block 20 positive hidden capture、原子 safetensors 写入和 resume。

### 8.1 单 GPU 完整 episode

下面使用已知较短的 `libero_goal` episode 125。它有 75 帧和 10 个 stride-8 anchors。

```bash
export CUDA_VISIBLE_DEVICES=0

python examples/wanvideo/model_inference/extract_libero_future_features.py \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --checkpoint-type "${CHECKPOINT_TYPE}" \
  --smoke-test \
  --suites libero_goal \
  --episode-indices 125 \
  --output-root "${RUN_ROOT}/smoke/features" \
  --progress-dir "${RUN_ROOT}/smoke/extraction_progress"
```

预期 feature 文件：

```text
${RUN_ROOT}/smoke/features/libero_goal/
  wan2_2_future_latent_h16_step20_block20z_stride8_480x480/
    chunk-000/episode_000125.safetensors
```

该文件应包含：

```text
features            [10, 1125, 3072] BF16
anchor_frames       [10]             int64
valid_future_frames [10]             int16
```

### 8.2 Resume 检查

目的：确认完整文件会在重新打开、校验 tensor/metadata/checkpoint hash 后跳过，而不是只根据文件名跳过。

原样重复上一条命令。输出 summary 中应出现：

```text
skipped_episodes=1
completed_episodes=1
invalid_episodes=[]
```

### 8.3 8-rank 分片 dry-run

目的：不加载模型，只检查 8 个 rank 的 episode 分片没有重复或遗漏。

```bash
unset CUDA_VISIBLE_DEVICES

torchrun --standalone --nproc_per_node=8 \
  examples/wanvideo/model_inference/extract_libero_future_features.py \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --checkpoint-type "${CHECKPOINT_TYPE}" \
  --dry-run \
  --smoke-test \
  --suites libero_10 \
  --episode-start 0 \
  --episode-end 8 \
  --progress-dir "${RUN_ROOT}/smoke/shard_dry_run"
```

预期 rank 0--7 各得到一个不同 episode。

## 9. 正式 8 GPU 全量 Feature Extraction

目的：对四个 suite 的全部 1693 episodes、34918 anchors 执行 20-step rollout，每个 GPU 独立处理 `global_episode_ordinal % 8 == rank` 的 episode。

启动前再次确认：

```bash
unset CUDA_VISIBLE_DEVICES
test -f "${CHECKPOINT_PATH}"
test -f "${VALIDATION_REPORT}"
df -h /data
```

正式命令：

```bash
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
CHECKPOINT_TYPE="${CHECKPOINT_TYPE}" \
VALIDATION_REPORT="${VALIDATION_REPORT}" \
RUN_ROOT="${RUN_ROOT}" \
bash examples/wanvideo/model_inference/run_extract_libero_future_features.sh
```

正式入口会校验 validation report 中的 checkpoint 绝对路径、SHA256、类型以及 loss/video 双门禁。任何一项不一致都会在加载大模型前停止。

默认 feature 直接写入每个 suite 根目录：

```text
${DATA_ROOT}/libero_10_no_noops_1.0.0_lerobot/
  wan2_2_future_latent_h16_step20_block20z_stride8_480x480/
${DATA_ROOT}/libero_goal_no_noops_1.0.0_lerobot/
  wan2_2_future_latent_h16_step20_block20z_stride8_480x480/
${DATA_ROOT}/libero_object_no_noops_1.0.0_lerobot/
  wan2_2_future_latent_h16_step20_block20z_stride8_480x480/
${DATA_ROOT}/libero_spatial_no_noops_1.0.0_lerobot/
  wan2_2_future_latent_h16_step20_block20z_stride8_480x480/
```

confidence-aware 模式下，每个 suite 的 `wan2_2_future_latent_h16_step20_block20z_stride8_480x480_samples4_confv1/` 目录还会包含该 suite 专属的 `confidence_manifest.json`。已有全局 calibration 缓存可直接用完整 suite 运行原地重校准，复用已保存的 `disagreement`，不会重新跑 teacher。

全局进度位于：

```text
${RUN_ROOT}/extraction_progress/rank-000.json
...
${RUN_ROOT}/extraction_progress/rank-007.json
${RUN_ROOT}/extraction_progress/extraction_summary.json
```

## 10. 中断恢复和定向重算

### 10.1 Resume 全量任务

目的：进程中断后继续未完成 episode。直接原样重新运行第 9 节命令即可。合法缓存会被校验并跳过；空文件、错误 shape/dtype/metadata/hash 或不完整文件会自动重算。

### 10.2 定向重算某些 episode

目的：只修复指定 suite/episode，同时强制覆盖已有文件。该命令只产生 pending confidence 缓存；完成全部修复后，使用不带 episode filter 的完整 suite 运行进行最终校准。

```bash
torchrun --standalone --nproc_per_node=8 \
  examples/wanvideo/model_inference/extract_libero_future_features.py \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --checkpoint-type "${CHECKPOINT_TYPE}" \
  --validation-report "${VALIDATION_REPORT}" \
  --suites libero_goal \
  --episode-indices 12 125 300 \
  --overwrite \
  --progress-dir "${RUN_ROOT}/extraction_progress_repair"
```

`--episode-start` 为 inclusive，`--episode-end` 为 exclusive，也可以用于连续范围重算。

## 11. 全局完整性审计

目的：不加载 Wan 模型、不执行 rollout，只重新打开全部 safetensors，核对 keys、shape、dtype、anchors、valid-future、metadata 和 checkpoint hash。

```bash
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
CHECKPOINT_TYPE="${CHECKPOINT_TYPE}" \
VALIDATION_REPORT="${VALIDATION_REPORT}" \
RUN_ROOT="${RUN_ROOT}" \
bash examples/wanvideo/model_inference/run_extract_libero_future_features.sh \
  --audit-only \
  --progress-dir "${RUN_ROOT}/audit_progress"
```

通过标准：

```text
complete=true
completed_episodes=1693
anchors=34918
invalid_episodes=[]
```

同时检查 suite 目录中不存在临时文件：

```bash
find "${DATA_ROOT}" -path '*wan2_2_future_latent_h16_step20_block20z_stride8_480x480*' \
  -type f -name '*.tmp' -print
```

正常情况下该命令没有输出。最终 feature 总磁盘占用预计约 225 GiB。

## 12. 交付给 StarVLA

完整审计通过后，StarVLA Stage 2 使用以下契约：

```bash
export SPACE2ACT_STAGE=2
export FUTURE_LATENT_DIR=wan2_2_future_latent_h16_step20_block20z_stride8_480x480
export FUTURE_TARGET_TOKENS=1125
export FUTURE_TARGET_DIM=3072
```

需要交付和记录：

```text
选中 checkpoint 的绝对路径和 SHA256
checkpoint 类型：lora 或 full
validation_report.json
四个 suite 的 extraction_summary.json
全局 audit_progress/extraction_summary.json
实际 episode 数、anchor 数和磁盘占用
```

不要交付 smoke 目录作为正式 teacher 数据；正式消费端读取的是四个 suite 根目录下相同 relative feature directory。
