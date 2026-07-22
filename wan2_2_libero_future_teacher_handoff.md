# Wan2.2-TI2V-5B LIBERO Future Teacher 实现交接

## 1. 文档用途

本文档是给在 `/home/jbx/Codes/DiffSynth-Studio` 中工作的下一轮 Codex 的实现规格。目标是：

1. 使用 LIBERO 真实机器人轨迹对 `Wan-AI/Wan2.2-TI2V-5B` 做领域微调，使其能够根据当前主相机图像和语言指令预测接下来 0.8 秒的视频。
2. 使用微调后的 Wan 做 20-step 条件 rollout。
3. 只缓存最终 denoise step 中 zero-based block 20 的正条件分支 hidden state，作为 Space2Act Stage 2 的 future teacher。
4. 不为训练窗口复制视频，不保存批量生成的视频或最终 VAE latent，把额外存储主要用于最终 teacher hidden。

这不是只写一个能跑的训练命令。实现必须同时覆盖：数据索引、在线窗口读取、LoRA/full 训练入口、验证、显式 hidden capture、分布式提取、原子缓存、断点续跑和测试。

## 2. 已锁定的决定

以下参数已经确定，不要在实现时自行改回 Wan 示例默认值。

| 项目 | 决定 |
| --- | --- |
| 目标仓库 | `/home/jbx/Codes/DiffSynth-Studio` |
| 已检查的仓库提交 | `fb337fbb90945ff829de69dbd44ded618f73e889` |
| Base model | `Wan-AI/Wan2.2-TI2V-5B` |
| 默认微调方式 | LoRA SFT；同时提供 full fine-tuning 可选脚本 |
| 相机 | 只使用 `observation.images.image`，不使用 wrist image |
| 输入条件 | 当前图像 + episode 原始 task instruction |
| 预测长度 | 17 个原生 20 FPS 连续帧，即 `[t, ..., t+16]`，0.8 秒未来 |
| 训练 anchor stride | 4 帧，即 0.2 秒 |
| teacher 提取 stride | 8 帧，即一个 teacher feature 共享给 0.4 秒内的 VLA 样本 |
| 分辨率 | `480x480`，保持 LIBERO 正方形视野，不做 `480x832` center crop |
| teacher rollout | 20 inference steps；不采用未经蒸馏的 4-step rollout |
| rollout guidance | `cfg_scale=5.0`、空 negative prompt、`rand_device="cpu"` |
| 抓取层 | zero-based block 20，即第 21 个 transformer block |
| 抓取 denoise step | zero-based step 19，即 20-step rollout 的最后一次 transformer forward |
| CFG 分支 | 只保存 positive conditional branch |
| temporal token | 保存全部 5 个 VAE 时间组，不做 temporal mean pooling |
| 输出 dtype | BF16 |
| 批量提取并行方式 | 8 个 GPU 独立处理 episode 分片，不使用 sequence-parallel all-gather |

不要使用 `/home/jbx/Codes/lingbot-world/wan` 实现本阶段。该目录对应 LingBot/Wan2.1 A14B 推理代码，维度、层数和本阶段的 5B teacher 均不同。

## 3. 已确认的现有代码依据

实现前先重新检查下列文件，若目标仓库之后发生变化，应保持本文档定义的行为，而不是机械依赖旧行号：

- `examples/wanvideo/model_training/train.py`
  - 已有 `WanTrainingModule`、SFT loss、LoRA/full 参数和 `--extra_inputs input_image`。
  - 当前 `parse_extra_inputs()` 使用 `data["video"][0]` 作为 `input_image`。
- `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh`
  - 已有 TI2V-5B LoRA 的 model loading 和 target module 范例。
- `examples/wanvideo/model_training/full/Wan2.2-TI2V-5B.sh`
  - 已有 full fine-tuning 范例。
- `diffsynth/core/data/unified_dataset.py` 和 `diffsynth/core/data/operators.py`
  - 现有 `LoadVideo` 只能从视频前缀读取，不能表示同一 episode 中任意 anchor 的 17 帧窗口。
  - 现有固定 `height/width` 预处理使用 center crop。对 `256x256 -> 480x832` 会丢弃大量上下画面，本任务禁止这样做。
- `diffsynth/pipelines/wan_video.py`
  - pipeline 每个 denoise step 先调用 positive model forward，再在 `cfg_scale != 1` 时调用 negative forward。
  - 普通 hook 如果只保留“最后一次调用”，会误保存 negative branch，因此必须显式标识 branch。
- `diffsynth/models/wan_video_dit.py`
  - `WanModel.blocks` 是 transformer block 列表，block 输出是 `[B, F*H*W, D]`。

官方 TI2V-5B 架构契约为 30 blocks、24 attention heads、head dim 128、hidden dim 3072、DiT patch size `(1,2,2)`。实现时仍应从实际已加载模型读取并 assert，不能只信硬编码。

## 4. 数据 Material Passport

数据根目录：

```text
/data/jibaixu/Datasets/LIBERO/libero_no_noop
```

四个数据集：

```text
libero_10_no_noops_1.0.0_lerobot
libero_goal_no_noops_1.0.0_lerobot
libero_object_no_noops_1.0.0_lerobot
libero_spatial_no_noops_1.0.0_lerobot
```

每个数据集均为 LeRobot v2.1 结构，关键文件为：

```text
meta/info.json
meta/episodes.jsonl
meta/tasks.jsonl
videos/chunk-{episode_chunk:03d}/observation.images.image/episode_{episode_index:06d}.mp4
```

已确认统计：

| Suite | Episodes | Frames | stride-4 windows | stride-8 anchors |
| --- | ---: | ---: | ---: | ---: |
| libero_10 | 379 | 101469 | 25505 | 12844 |
| libero_goal | 428 | 52042 | 13156 | 6687 |
| libero_object | 454 | 66984 | 16920 | 8570 |
| libero_spatial | 432 | 52970 | 13410 | 6817 |
| Total | 1693 | 273465 | 68991 | 34918 |

视频为 20 FPS、`256x256`、AV1、无音频。episode index 只在 suite 内唯一，因此任何 seed、manifest key、日志 key 或全局索引都必须同时包含 suite id。

当前机器有 8 张 `NVIDIA A100-PCIE-40GB`。`/data` 当前约有 2.2 TiB 可用，系统盘只剩约 9 GiB。checkpoint、缓存和验证视频必须写入 `/data`，不能写到仓库目录。

## 5. 目标文件与修改边界

建议在 DiffSynth-Studio 中新增以下文件；如果仓库已有更合适的测试目录，可以调整测试文件位置，但不要改变功能边界。

```text
examples/wanvideo/model_training/libero_future_dataset.py
examples/wanvideo/model_training/train_libero_future.py
examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-LIBERO.sh
examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-LIBERO.sh
examples/wanvideo/model_training/validate_libero_future.py
examples/wanvideo/model_inference/extract_libero_future_features.py
examples/wanvideo/model_inference/run_extract_libero_future_features.sh
tools/build_libero_future_manifest.py
tests/test_libero_future_teacher.py
```

允许对 `diffsynth/pipelines/wan_video.py` 的 denoise loop 和对应 model function 做小范围、默认关闭、向后兼容的 feature-capture 扩展。不要复制整套 Wan 模型实现，不要修改正常 inference/training 的默认返回值。

当前任务不修改 starVLA 仓库中的训练代码。Wan 项目只负责输出满足第 10 节契约的 safetensors。

## 6. Manifest 与在线窗口数据集

### 6.1 Manifest 生成

`tools/build_libero_future_manifest.py` 至少提供：

```text
--data-root
--output-dir
--train-stride 4
--val-stride 4
--val-ratio 0.1
--split-seed 42
--video-key observation.images.image
--num-frames 17
```

生成：

```text
train.jsonl
val.jsonl
all.jsonl
summary.json
```

每个 JSONL row 至少包含：

```json
{
  "suite": "libero_goal",
  "suite_id": 1,
  "dataset_root": "/absolute/path/to/suite",
  "episode_index": 12,
  "episode_length": 155,
  "anchor_frame": 40,
  "video_path": "/absolute/path/to/episode_000012.mp4",
  "prompt": "put the wine bottle on the rack",
  "valid_future_frames": 16
}
```

要求：

- `anchor_frame = 0, stride, 2*stride, ... < episode_length`。
- `valid_future_frames = min(16, episode_length - 1 - anchor_frame)`。
- train/val 必须以 episode 为单位划分，不能以 window 随机划分。
- 按 `(suite, task)` 分层，seed 固定为 42；同样输入重复运行必须逐字节生成相同 manifest。
- `summary.json` 记录 suite 统计、episode split、window 数、参数和生成时间。
- 生成结束 assert：`train windows + val windows == 68991`，`all episodes == 1693`。

### 6.2 Dataset 行为

实现 `LiberoFutureWindowDataset(torch.utils.data.Dataset)`，返回：

```python
{
    "video": list[PIL.Image],  # exactly 17 RGB images
    "prompt": str,
    "sample_id": str,
    "suite": str,
    "episode_index": int,
    "anchor_frame": int,
    "valid_future_frames": int,
}
```

`video` 的帧索引必须严格为：

```python
frame_ids = [min(anchor_frame + i, episode_length - 1) for i in range(17)]
```

即：

- 不做 temporal subsampling。
- episode 尾部用最后一帧重复 padding。
- 不跨 episode。
- 所有帧先转 RGB，再使用 bilinear resize 到 `480x480`。
- 不做 random crop、horizontal flip、颜色增强或帧率转换。
- 不提前导出 68991 个窗口视频。

优先使用 PyAV 或仓库已有可可靠 random seek 的 decoder。先保证正确性，再做有限的 per-worker decoder/LRU 优化。任何缓存必须有上限，不能将所有 episode 解码到内存。

`train_libero_future.py` 应复用现有 `WanTrainingModule`，不要复制 loss 或 pipeline。它只负责选择 `LiberoFutureWindowDataset` 并调用现有 training launcher。若现有 launcher 依赖 `dataset.load_from_cache`，dataset 需明确设置为 `False` 或做最小兼容处理。

## 7. Wan 微调

### 7.1 条件与监督

每个样本的条件和 target 为：

```text
condition image = video[0]
condition text  = 原始 LIBERO task string，不额外改写
target video    = video[0:17]
```

使用现有 `FlowMatchSFTLoss`。`--extra_inputs input_image` 必须开启，使 TI2V 使用首帧条件。训练时只有 target video 包含真实未来帧；推理和 teacher extraction 绝对不能读取真实未来帧。

### 7.2 默认 LoRA 配置

默认启动脚本使用：

```text
model: Wan-AI/Wan2.2-TI2V-5B
height: 480
width: 480
num_frames: 17
dataset_repeat: 1
learning_rate: 1e-4
weight_decay: 0.01
num_epochs: 3
lora_base_model: dit
lora_target_modules: q,k,v,o,ffn.0,ffn.2
lora_rank: 32
lora inference alpha: 1.0
gradient_accumulation_steps: 4
use_gradient_checkpointing: true
dataset_num_workers: 2 per process initially
mixed_precision: bf16
seed: 42
save_steps: 500
```

8 GPU 下每 GPU micro-batch 为 1，effective global batch 为 `8 * 1 * 4 = 32`。现有 runner 的 DataLoader 没有显式 batch size，默认就是 1，不要为了追求更大 batch 先改通用训练框架。

训练输出：

```text
/data/jibaixu/Models/wan2_2_ti2v_5b_libero/lora_sft
```

必须保存实际 CLI 参数、git commit、manifest hash、base model identifier、CUDA/PyTorch 版本和随机种子。

### 7.3 可选 full fine-tuning

提供但不默认执行：

```text
trainable_models: dit
learning_rate: 1e-5
weight_decay: 0.01
num_epochs: 2
gradient_accumulation_steps: 4
mixed_precision: bf16
DeepSpeed: examples/wanvideo/model_training/full/accelerate_config_zero3.yaml
initialize_model_on_cpu: true
```

输出：

```text
/data/jibaixu/Models/wan2_2_ti2v_5b_libero/full_sft
```

full 脚本用于后续容量对照，不应阻塞 LoRA 的首个训练和提取闭环。

### 7.4 本轮明确不做的事情

- 不做 4-step direct distillation。
- 不用未经蒸馏的 4-step model 做全量 teacher extraction。
- 不训练 T5 或 VAE。
- 不加入 action/state/wrist camera 条件。
- 不把 17 帧降采样成 5 帧；5 是 VAE temporal latent group 数，不是输入视频帧数。

## 8. Validation 与 checkpoint 选择

新增 `validate_libero_future.py`，同时支持 base model 和 LoRA/full checkpoint。

### 8.1 确定性 validation loss

- 从 val manifest 以固定 seed 选择 256 个窗口，覆盖四个 suite 和所有 task。
- 对每个 checkpoint 使用相同样本、相同 timestep 集和相同 noise seed 计算 SFT flow loss。
- 不允许 validation 调用改变训练 checkpoint。
- 输出 JSON/CSV，至少包含 checkpoint、mean/std loss、suite 分项、task 分项和 non-finite count。
- 默认选择 mean validation loss 最低的 checkpoint。
- 选中的 checkpoint 必须比未微调 base model 的 loss 更低，且所有值 finite。

### 8.2 20-step 视频验证

固定选择 32 个 val anchors，并用 base 与 tuned checkpoint 分别生成：

```text
input_image: anchor 当前帧
prompt: 原始 task
height/width: 480/480
num_frames: 17
num_inference_steps: 20
sigma_shift: 5.0
cfg_scale: 5.0
negative_prompt: ""
rand_device: cpu
tiled: false
seed: 固定且 base/tuned 相同
```

这些少量验证视频可以保存到：

```text
/data/jibaixu/Models/wan2_2_ti2v_5b_libero/validation
```

至少报告 PSNR/SSIM 的 final-frame 和全 future-frame 均值，并生成 base/tuned/GT 对照索引。若环境已有 LPIPS，可额外报告，但不能让 LPIPS 成为新的强制依赖。

启动全量特征提取前必须确认：

- 首帧条件保持正常。
- 未来帧不是完全静态复制。
- 没有普遍的结构崩坏、闪烁或颜色坍缩。
- tuned validation loss 优于 base。

若这些条件不满足，停止在验证阶段，不要生成 225 GiB 无效 teacher cache。

## 9. 显式 DiT Feature Capture

### 9.1 为什么不能只用普通 hook

当前 pipeline 在每个 denoise step 中按以下顺序执行：

```text
positive conditional forward
negative conditional forward  # cfg_scale != 1 时
scheduler step
```

因此仅在 `pipe.dit.blocks[20]` 注册 hook 并保存“最后一次输出”，得到的是 negative branch。实现必须由 denoise loop 明确告诉 model function 当前是哪个 step 和 branch。

### 9.2 建议接口

可以采用等价设计，但行为必须如下。建议新增不可变配置：

```python
@dataclass(frozen=True)
class WanFeatureCaptureSpec:
    block_index: int = 20
    denoise_step: int = 19
    branch: str = "positive"
    to_cpu: bool = True
    dtype: torch.dtype = torch.bfloat16
```

并提供不破坏普通 `pipe(...)` 返回值的专用方法：

```python
feature = pipe.extract_dit_feature(
    prompt=prompt,
    input_image=image,
    height=480,
    width=480,
    num_frames=17,
    seed=seed,
    num_inference_steps=20,
    sigma_shift=5.0,
    cfg_scale=5.0,
    negative_prompt="",
    rand_device="cpu",
    tiled=False,
    capture=WanFeatureCaptureSpec(),
    decode_video=False,
)
```

实现要求：

- capture 参数默认关闭，普通 inference 和 training 行为完全不变。
- denoise loop 只在目标 step 的 positive call 中向 model function 传入 collector。
- collector 在目标 block forward 完成后接收 `x`。
- 保存 `x.detach().to(torch.bfloat16).cpu()`，随后释放 GPU 引用。
- 前 19 个 denoise step 不保留任何 block hidden。
- 最终不运行 VAE decoder；方法只返回 feature 和必要的 shape metadata。
- 使用 `torch.inference_mode()`。
- assert 目标 model 有 30 blocks、目标 block index 合法、hidden dim 为 3072。
- 若启用 sequence parallel，必须正确 all-gather 后再返回；但本项目的默认提取脚本禁止启用 sequence parallel，因此首版不需要为吞吐使用它。

### 9.3 Feature shape 与顺序

`480x480`、17 帧时，预期：

```text
VAE temporal groups: 5
VAE spatial grid: 30 x 30
DiT patch grid: 15 x 15
hidden dim: 3072
tokens: 5 * 15 * 15 = 1125
```

block 输出必须解释为：

```python
hidden.reshape(5, 15, 15, 3072).reshape(1125, 3072)
```

展平顺序固定为 temporal-major，然后 height-major，再 width-major。保存全部 5 个时间组，不做 temporal pooling，不删除第一个时间组。

如果运行时实际 shape 不是 `[1,1125,3072]`，必须报错并停止，不能静默 reshape、截断或 padding。

## 10. 全量 Teacher Extraction

### 10.1 Anchor 定义

每个 episode：

```python
anchor_frames = np.arange(0, episode_length, 8, dtype=np.int64)
```

总 anchor 数必须为 34918。每个 anchor 的 inference 输入只有：

- `observation.images.image[anchor_frame]`
- episode task instruction
- deterministic noise seed

不得读取 `[t+1, ..., t+16]` 真实帧。`valid_future_frames` 只用于分析 episode 尾部距离，不参与 rollout。

### 10.2 Seed

suite id 固定：

```text
libero_10      -> 0
libero_goal    -> 1
libero_object  -> 2
libero_spatial -> 3
```

建议公式：

```python
seed = (
    base_seed
    + suite_id * 100_000_000
    + episode_index * 1_000_003
    + anchor_frame * 9_176
) % (2**31)
```

`base_seed=42`。公式及最终 seed 必须进入 metadata 或 extraction manifest。

### 10.3 8 GPU 分片

使用 `torchrun --nproc_per_node=8` 启动 8 个独立 worker：

```python
assigned = global_episode_ordinal % world_size == rank
```

要求：

- 每个 rank 设置自己的 CUDA device 并各自加载 base + selected LoRA/full checkpoint。
- LoRA checkpoint 以 `alpha=1.0` 加载；提取 summary 必须记录实际 alpha。
- episode 是最小任务单位，同一 episode 不能被多个 rank 同时写入。
- 不需要 DDP gradient、NCCL all-gather 或 sequence parallel。
- rank 0 写全局 manifest/summary，其他 rank 写 rank-local progress；结束后 rank 0 汇总。
- 单个 episode 内逐 anchor 推理，feature 先放 CPU，episode 完成后一次性写 safetensors。
- 若单 episode CPU feature 堆叠导致峰值过高，可以使用有界 CPU buffer，但最终仍保持每 episode 一个文件。

### 10.4 输出目录

每个 suite 根目录下固定使用相同 relative directory：

```text
wan2_2_future_latent_h16_step20_block20z_stride8_480x480
```

文件结构：

```text
<suite-root>/wan2_2_future_latent_h16_step20_block20z_stride8_480x480/
  chunk-000/
    episode_000000.safetensors
    episode_000001.safetensors
    ...
  extraction_summary.json
```

chunk id 按 `meta/info.json` 的 `chunks_size` 和 `video_path` 规则计算，不能假设永远只有 `chunk-000`。

### 10.5 Safetensors tensor 契约

每个 episode 文件必须包含：

```text
features            [A, 1125, 3072]  bfloat16
anchor_frames       [A]              int64
valid_future_frames [A]              int16
```

其中：

```python
A = ceil(episode_length / 8)
valid_future_frames[i] = min(16, episode_length - 1 - anchor_frames[i])
```

feature 不允许量化成 FP8/INT8，不允许 temporal mean，不允许只保存未来 4 个时间组。

### 10.6 Metadata 契约

safetensors metadata 值均为字符串。至少包含：

```text
complete="true"
teacher="Wan-AI/Wan2.2-TI2V-5B"
checkpoint_path=<selected checkpoint absolute path>
checkpoint_sha256=<checkpoint hash>
suite=<suite name>
suite_id=<0..3>
episode_index=<id>
episode_length=<length>
fps="20"
video_key="observation.images.image"
prompt_source="meta/episodes.jsonl:tasks[0]"
share_stride="8"
future_horizon_frames="16"
num_frames="17"
height="480"
width="480"
num_inference_steps="20"
saved_denoise_step="19"
saved_denoise_step_indexing="zero_based"
block_index="20"
block_indexing="zero_based"
cfg_branch="positive"
cfg_scale="5.0"
negative_prompt=""
rand_device="cpu"
tiled="false"
sigma_shift="5.0"
temporal_tokens="5"
token_h="15"
token_w="15"
target_tokens="1125"
feature_dim="3072"
token_order="temporal_height_width"
dtype="bfloat16"
base_seed="42"
```

可以增加 model revision、DiffSynth commit、torch version、CUDA version 和 extraction command，但不能省略上述字段。

### 10.7 原子写入和 resume

- 先写同目录临时文件，例如 `episode_000000.safetensors.tmp`。
- 写完并重新打开验证 tensor shape、anchors 和 metadata 后，用 `os.replace()` 原子替换正式文件。
- 正式文件只有满足以下全部条件才能跳过：
  - `complete == "true"`
  - tensor keys 完整
  - shape 与 episode 预期一致
  - anchor 严格递增且等于 `np.arange(0,length,8)`
  - checkpoint hash、block、step、stride、resolution、token order 全部一致
- 空文件、`.tmp`、metadata 不一致或 shape 不一致均视为未完成，默认重算。
- 提供 `--overwrite`、`--episode-start`、`--episode-end`、`--episode-indices` 和 `--dry-run`。

## 11. 存储预算

单 anchor feature：

```text
1125 * 3072 * 2 bytes = 6,912,000 bytes ≈ 6.59 MiB
```

预计总缓存：

| Suite | Anchors | 预计 feature 大小 |
| --- | ---: | ---: |
| libero_10 | 12844 | 约 82.7 GiB |
| libero_goal | 6687 | 约 43.0 GiB |
| libero_object | 8570 | 约 55.2 GiB |
| libero_spatial | 6817 | 约 43.9 GiB |
| Total | 34918 | 约 224.8 GiB |

metadata 和 anchor tensor 的开销相对很小。禁止额外保存 34918 个生成视频、最终 VAE latent、每个 denoise step 的 hidden 或多个 transformer layer，否则存储会成倍增长。

## 12. StarVLA 消费端契约

输出将被 `/home/jbx/Codes/starVLA_origin/starVLA` 中现有 Stage 2 代码消费。不要修改 feature shape 来迎合其他 Wan 示例。

启动 Stage 2 时的关键参数：

```bash
SPACE2ACT_STAGE=2 \
FUTURE_LATENT_DIR=wan2_2_future_latent_h16_step20_block20z_stride8_480x480 \
FUTURE_TARGET_TOKENS=1125 \
FUTURE_TARGET_DIM=3072 \
./examples/simBenchmarks/LIBERO/train_files/run_space2act_train_jbx.sh
```

当前 StarVLA 侧的 future channel 应使用：

```text
typed F tokens: 32
teacher target tokens: 1125
teacher target dim: 3072
alignment dim: 1024
attention heads: 8
query chunk size: 128
feature key: features
anchor key: anchor_frames
```

StarVLA loader 会对每个 VLA frame 使用 latest anchor `<= frame_index`，因此 stride-8 teacher 自然共享给后续 8 帧。必须保证每个 episode 的第一个 anchor 是 0。

建议后续在 StarVLA YAML 的 future `expected_metadata` 中检查：

```yaml
complete: "true"
share_stride: "8"
block_index: "20"
block_indexing: "zero_based"
num_inference_steps: "20"
saved_denoise_step: "19"
target_tokens: "1125"
feature_dim: "3072"
token_order: "temporal_height_width"
```

teacher mapper 是训练期辅助层，推理时不会加载 Wan 或读取 teacher safetensors。

## 13. 测试与执行顺序

必须按以下顺序推进，不要直接启动全量缓存。

### 13.1 CPU/unit tests

1. Manifest：四 suite 统计正确，总 episode 1693，总 stride-4 windows 68991。
2. Split：train/val episode 集合无交集，同 seed 可复现，不同 suite 的同 episode id 不冲突。
3. Window：正常窗口恰好 17 帧；尾部窗口重复最后帧；不跨 episode。
4. Resize：输出均为 RGB `480x480`，完整正方形画面，无 center crop。
5. Anchor：stride-8 anchors 严格等于 `arange(0,length,8)`，总数 34918。
6. Cache validator：能拒绝错误 dtype、shape、metadata、缺 key、无 anchor 0 和未完成文件。

### 13.2 模型 capture tests

用 mock blocks 或极小输入验证：

1. `cfg_scale=5` 时只保存 positive branch，而不是 negative branch。
2. 只在 zero-based denoise step 19 抓取。
3. 只在 zero-based block 20 抓取。
4. 普通 `pipe(...)` 在 capture 关闭时返回和修改前一致。
5. `extract_dit_feature(..., decode_video=False)` 不调用 VAE decoder。
6. 返回 shape 严格为 `[1125,3072]` BF16 CPU tensor，顺序为 `T,H,W`。

### 13.3 GPU smoke tests

1. 解码一个真实 window，执行一次 LoRA training forward/backward。
2. 单 GPU 跑 1 optimizer step，确认无 OOM、loss finite、checkpoint 可写。
3. 8 GPU 跑 10--50 optimizer steps，确认样本正确分片、effective batch 和保存逻辑正确。
4. 用选中的 checkpoint 对一个 anchor 跑 20-step capture，只生成 feature，不 decode 视频。
5. 完整提取一个 episode，重新打开 safetensors 并逐项验证。
6. 用 8 GPU 各分配至少一个 episode，确认无重复写入、无遗漏、resume 可跳过完整文件。

### 13.4 正式顺序

```text
build manifests
-> LoRA 1-step smoke
-> LoRA short multi-GPU smoke
-> complete LoRA SFT run
-> validate all checkpoints against base
-> select checkpoint
-> generate 32 held-out comparison videos
-> one-window feature smoke
-> one-episode cache smoke
-> 8-GPU shard smoke
-> full 34918-anchor extraction
-> global integrity audit
-> hand output directory to StarVLA Stage 2
```

## 14. 完成标准

本任务只有同时满足以下条件才算完成：

- LoRA 训练能在真实 LIBERO window 上运行并保存 checkpoint。
- 验证脚本证明 selected checkpoint 的 held-out loss 优于 base，且 20-step 视频没有明显坍缩。
- feature capture 明确命中 final positive branch / step 19 / block 20。
- 单 anchor feature 为 `[1125,3072]` BF16，包含 5 个 temporal groups。
- 四个 suite 共生成 1693 个完整 episode 文件、34918 个 anchors。
- 所有文件通过独立 integrity audit，支持中断后 resume。
- 全量 feature 数据约 225 GiB，不包含批量视频、VAE latent、其他 layer 或其他 denoise step。
- StarVLA 可以使用 `FUTURE_TARGET_TOKENS=1125` 和 `FUTURE_TARGET_DIM=3072` 完成一次 Stage 2 one-step smoke training。

最后一项需要在 Wan 输出完成后回到 starVLA 仓库执行；Wan 工程中的下一轮 Codex 应在最终报告中给出 checkpoint 路径、relative feature directory、实际文件/anchor 数、实际磁盘占用和完整性审计结果。
