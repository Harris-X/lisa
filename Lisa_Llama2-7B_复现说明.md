# Lisa（Llama2-7B）一键复现说明（Linux 远程服务器）

本文档给出从环境、数据、训练到评测的完整流程。

参数策略更新为：**同名参数优先采用 `T-Vaccine复现.md`，其余使用 Lisa 默认参数**。

## 1. 目标与默认设置

- 基础模型（本地目录默认）：`/data_nvme1n1/xieqiuhao/tjy/downloaded_models/Llama-2-7b-hf`
- 评估模型（自动下载/复用）：`/data_nvme1n1/xieqiuhao/tjy/downloaded_models/beaver-dam-7b`
- HF 镜像（默认）：`HF_ENDPOINT=https://hf-mirror.com`
- 阶段一（安全对齐 SFT）：`lr=1e-3`，`batch_size=10`，`epochs=20`，`weight_decay=0.1`
- 阶段二（Lisa 防御微调）：`lr=1e-5`，`batch_size=10`，`epochs=20`，`weight_decay=0.1`
- 攻击设置：`sample_num=1000`，`poison_ratio=0.1`
- 数据规模（同名参数优先 T-Vaccine）：`ALIGN_SAFE_NUM=2000`，`GUIDE_DATA_NUM=2000`
- Lisa 特有超参：`alignment_step=100`，`finetune_step=900`
- `rho`：按你要求采用 T-Vaccine 数值 `3`（注意其语义与 Lisa 论文中的 proximal `rho` 不同）

## 2. 文件说明（本次新增）

- 一键数据准备脚本：`script/repro/prepare_datasets.sh`
- 一键端到端复现脚本：`script/repro/run_lisa_llama2_7b_oneclick.sh`

## 3. 运行前准备

### 3.1 进入项目目录

```bash
cd /path/to/Lisa
```

### 3.2 创建环境并安装依赖

```bash
conda env create -f lisa.yml
conda activate vaccine
pip install -r lisa_pip.txt
```

### 3.3 配置 HuggingFace Token

在仓库根目录创建（或覆盖）文件：`huggingface_token.txt`

```bash
echo "hf_xxx_your_token" > huggingface_token.txt
```

同时确认该账号已经获得 `meta-llama/Llama-2-7b-hf` 的访问权限。

### 3.4 给脚本执行权限

```bash
chmod +x script/repro/prepare_datasets.sh
chmod +x script/repro/run_lisa_llama2_7b_oneclick.sh
```

### 3.5（可选）显式设置 HF 镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 4. 一键复现（推荐）

在项目根目录运行：

```bash
bash script/repro/run_lisa_llama2_7b_oneclick.sh
```

如果担心 SSH/远程终端断开，使用后台模式（推荐）：

```bash
DETACH_RUN=1 GPU_ID=0 bash script/repro/run_lisa_llama2_7b_oneclick.sh
```

后台模式会自动使用 `nohup` 启动并写入：

- 日志：`logs/oneclick_*.nohup.log`
- PID：`logs/oneclick_*.pid`

该命令会自动完成：

1. 下载/缓存数据集并构建 `data/sst2.json`、`data/agnews.json`、`data/gsm8k.json`
2. 阶段一：安全对齐 SFT（BeaverTails_safe）
3. 阶段一后有害评测（HS）
4. 阶段二：Lisa 防御微调（SST2 + 有害样本，10% poison）
5. 阶段二后有害评测（HS）+ SST2 准确率评测（FA）
6. 可选：自动跑普通 SFT 对照组（默认开启）

## 5. 可配置项（环境变量）

可在运行前覆盖默认值：

```bash
DOWNLOADED_MODELS_DIR=/data_nvme1n1/xieqiuhao/tjy/downloaded_models \
MODEL_PATH=/data_nvme1n1/xieqiuhao/tjy/downloaded_models/Llama-2-7b-hf \
EVAL_MODEL_LOCAL_DIR=/data_nvme1n1/xieqiuhao/tjy/downloaded_models/beaver-dam-7b \
HF_ENDPOINT=https://hf-mirror.com \
POISON_RATIO=0.1 \
SAMPLE_NUM=1000 \
RHO=3 \
ALIGN_EPOCHS=20 \
ALIGN_BS=10 \
ALIGN_SAFE_NUM=2000 \
FINETUNE_BS=10 \
GUIDE_DATA_NUM=2000 \
ALIGN_STEP=100 \
FINETUNE_STEP=900 \
RUN_BASELINE_SFT=1 \
RUN_DATA_PREP=1 \
DETACH_RUN=1 \
GPU_ID=0 \
bash script/repro/run_lisa_llama2_7b_oneclick.sh
```

说明：

- `RUN_BASELINE_SFT=1`：同时跑普通 SFT 对照组
- `RUN_DATA_PREP=1`：先执行数据准备（第一次跑建议保持 1）
- `DETACH_RUN=1`：后台运行，终端断开后任务继续
- `GPU_ID=0`：指定可见显卡（脚本会检查并提示是否为 H20）
- `DOWNLOADED_MODELS_DIR`：模型统一下载/存放目录
- `EVAL_MODEL_LOCAL_DIR`：评估模型本地目录（不存在会自动下载）
- `HF_ENDPOINT`：HF 镜像地址

## 6. 关键输出路径

### 6.1 模型权重

- 对齐模型（阶段一）：`ckpt/Llama-2-7b-hf_sft`
- Lisa 模型（阶段二）：
  `ckpt/sst2/Llama-2-7b-hf_lisa_f_3_0.1_1000_100_900_2000`
- 普通 SFT 对照模型（可选）：
  `ckpt/sst2/Llama-2-7b-hf_sft_f_0_0.1_1000_100_900_0`

### 6.1.1 预训练/评估模型目录

- Llama2 本地目录（默认要求存在）：
  `/data_nvme1n1/xieqiuhao/tjy/downloaded_models/Llama-2-7b-hf`
- Moderation 评估模型目录（脚本自动下载）：
  `/data_nvme1n1/xieqiuhao/tjy/downloaded_models/beaver-dam-7b`

### 6.2 评测结果

- 阶段一有害评测：
  `data/poison/Llama-2-7b-hf_sft_sentiment_eval.json`
- Lisa 有害评测：
  `data/poison/sst2/Llama-2-7b-hf_lisa_f_3_0.1_1000_100_900_2000_sentiment_eval.json`
- Lisa SST2 评测：
  `data/sst2/Llama-2-7b-hf_lisa_f_3_0.1_1000_100_900_2000`

### 6.3 日志与显存

- 全流程日志目录：`logs/`
- 阶段一显存监控：`logs/gpu/stage1_alignment_gpu_mem.csv`
- 终端会打印阶段一峰值显存（MiB）

## 7. 分步运行（手动模式）

如果你不想一键跑完，可以手动分步执行：

### 步骤 A：准备数据

```bash
cd /path/to/Lisa
bash script/repro/prepare_datasets.sh
```

### 步骤 B：阶段一安全对齐

```bash
cd /path/to/Lisa
python train.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --data_path PKU-Alignment/BeaverTails_safe \
  --bf16 True \
  --output_dir ckpt/Llama-2-7b-hf_sft \
  --num_train_epochs 20 \
  --per_device_train_batch_size 10 \
  --per_device_eval_batch_size 10 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 100000 \
  --learning_rate 1e-3 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --cache_dir cache \
  --guide_data_num 2000 \
  --optimizer normal
```

### 步骤 C：阶段一 HS 评测

```bash
cd /path/to/Lisa/poison/evaluation
python pred.py \
  --lora_folder ../../ckpt/Llama-2-7b-hf_sft \
  --model_folder meta-llama/Llama-2-7b-hf \
  --output_path ../../data/poison/Llama-2-7b-hf_sft

python eval_sentiment.py \
  --input_path ../../data/poison/Llama-2-7b-hf_sft \
  --moderation_model_path /data_nvme1n1/xieqiuhao/tjy/downloaded_models/beaver-dam-7b
```

### 步骤 D：阶段二 Lisa 微调

```bash
cd /path/to/Lisa
python train.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_folder ckpt/Llama-2-7b-hf_sft \
  --data_path PKU-Alignment/BeaverTails_dangerous \
  --bf16 True \
  --output_dir ckpt/sst2/Llama-2-7b-hf_lisa_f_3_0.1_1000_100_900_2000 \
  --num_train_epochs 20 \
  --per_device_train_batch_size 10 \
  --per_device_eval_batch_size 10 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 100000 \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type constant \
  --logging_steps 10 \
  --tf32 True \
  --eval_steps 2000 \
  --cache_dir cache \
  --optimizer lisa \
  --evaluation_strategy steps \
  --sample_num 1000 \
  --poison_ratio 0.1 \
  --label_smoothing_factor 0 \
  --benign_dataset data/sst2.json \
  --rho 3 \
  --alignment_step 100 \
  --finetune_step 900 \
  --guide_data_num 2000
```

### 步骤 E：阶段二评测（HS + FA）

```bash
cd /path/to/Lisa/poison/evaluation
python pred.py \
  --lora_folder ../../ckpt/Llama-2-7b-hf_sft \
  --lora_folder2 ../../ckpt/sst2/Llama-2-7b-hf_lisa_f_3_0.1_1000_100_900_2000 \
  --model_folder meta-llama/Llama-2-7b-hf \
  --output_path ../../data/poison/sst2/Llama-2-7b-hf_lisa_f_3_0.1_1000_100_900_2000

python eval_sentiment.py \
  --input_path ../../data/poison/sst2/Llama-2-7b-hf_lisa_f_3_0.1_1000_100_900_2000 \
  --moderation_model_path /data_nvme1n1/xieqiuhao/tjy/downloaded_models/beaver-dam-7b

cd ../../sst2
python pred_eval.py \
  --lora_folder ../ckpt/Llama-2-7b-hf_sft \
  --lora_folder2 ../ckpt/sst2/Llama-2-7b-hf_lisa_f_3_0.1_1000_100_900_2000 \
  --model_folder meta-llama/Llama-2-7b-hf \
  --output_path ../data/sst2/Llama-2-7b-hf_lisa_f_3_0.1_1000_100_900_2000
```

## 8. 与论文结果对齐时的检查建议

- HS：以 `*_sentiment_eval.json` 中最终分数为准
- FA：以 `sst2/pred_eval.py` 输出准确率为准
- 显存：关注 `logs/gpu/stage1_alignment_gpu_mem.csv` 对应峰值
- 对照组：建议开启 `RUN_BASELINE_SFT=1`，便于比较 Lisa 与普通 SFT
- 设备：脚本会提示是否使用 H20；若不是 H20，会给出警告
- 模型：脚本限制为 Llama2，避免当前依赖版本下其他模型不兼容

## 9. 故障排除与修复

### 9.1 进程卡死问题修复

**现象**：脚本启动后只输出初始信息就卡住，日志停留在 `preflight_checks` 阶段。

**原因**：
- `nvidia-smi` 查询在某些驱动环境下会永久阻塞
- `import transformers` 时可能触发网络请求导致卡死
- `set -euo pipefail` 中的 `-u` 选项导致未定义变量直接退出（无错误输出）

**修复方案**（已应用到脚本）：
- `nvidia-smi` 加 `timeout 10` 防止阻塞
- `python -c` 加 `timeout 30` 防止网络卡死
- 去掉 `set -u`，改为 `set -eo pipefail`
- 每个检查步骤前加调试 `echo` 输出，便于定位卡点
- 主流程加 `[Step N/5]` 进度标记

### 9.2 杀死卡住进程

创建了专用杀进程脚本：`script/repro/kill_lisa.sh`

```bash
# 预览相关进程（不杀）
bash script/repro/kill_lisa.sh

# 强杀所有相关进程
bash script/repro/kill_lisa.sh --force
```

该脚本会：
- 从 `logs/*.pid` 文件读取记录的 PID
- 通过关键词搜索相关进程（`oneclick`、`train.py`、`prepare_datasets` 等）
- 显示进程树和 GPU 占用
- 支持优雅终止（SIGTERM）或强制终止（SIGKILL）

### 9.3 重新启动流程

```bash
# 1. 确认进程已杀干净
bash script/repro/kill_lisa.sh

# 2. 重新启动（后台模式）
GPU_ID=1 bash script/repro/run_lisa_llama2_7b_oneclick.sh

# 3. 查看新日志
tail -f logs/oneclick_*.nohup.log
```

### 9.4 其他常见问题

- **模型下载失败**：检查 `huggingface_token.txt` 和 HF 权限
- **显存不足**：减少 `ALIGN_BS` 或 `FINETUNE_BS`
- **数据集下载慢**：确认 `HF_ENDPOINT` 设置正确
- **评估模型缺失**：脚本会自动下载到 `EVAL_MODEL_LOCAL_DIR`

### 9.5 数据集名称冲突（sst2 加载报错 FileNotFoundError）

**现象**：`prepare_datasets.sh` 下载 BeaverTails 成功后，加载 `sst2` 时报错：
```
FileNotFoundError: No (supported) data files or dataset script found in sst2
```

**原因**：项目根目录下有 `sst2/`、`agnews/`、`gsm8k/` 文件夹（存放 `build_dataset.py`），而 `datasets` 库的 `load_dataset("sst2")` 会**优先匹配本地同名目录**，将其当作本地数据集解析，自然找不到数据文件。

**修复方案**（已应用）：
- `prepare_datasets.sh`：数据集名改为完整 Hub 路径（`stanfordnlp/sst2`、`fancyzhx/ag_news`、`openai/gsm8k`）
- `sst2/build_dataset.py`：`load_dataset("sst2")` → `load_dataset("stanfordnlp/sst2")`
- `agnews/build_dataset.py`：`load_dataset("ag_news")` → `load_dataset("fancyzhx/ag_news")`
- `gsm8k/build_dataset.py`：`load_dataset("gsm8k", "main")` → `load_dataset("openai/gsm8k", "main")`
