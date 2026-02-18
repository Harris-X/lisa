#!/usr/bin/env bash
set -eo pipefail
# 注意：去掉了 -u（nounset），某些环境变量判断在 unset 时会误报
# pipefail 保留，确保管道中任一命令失败都能被捕获

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export ROOT_DIR

DOWNLOADED_MODELS_DIR="${DOWNLOADED_MODELS_DIR:-/data_nvme1n1/xieqiuhao/tjy/downloaded_models}"
MODEL_PATH="${MODEL_PATH:-${DOWNLOADED_MODELS_DIR}/Llama-2-7b-hf}"
MODEL_SHORT="$(basename "$MODEL_PATH")"
EVAL_MODEL_REPO="${EVAL_MODEL_REPO:-PKU-Alignment/beaver-dam-7b}"
EVAL_MODEL_LOCAL_DIR="${EVAL_MODEL_LOCAL_DIR:-${DOWNLOADED_MODELS_DIR}/beaver-dam-7b}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ===== 按“同名参数优先采用 T-Vaccine，其他保留 Lisa 默认”的核心参数 =====
# T-Vaccine 同名参数：epochs=20, batch=10, lr(stage1)=1e-3, lr(stage2)=1e-5,
# weight_decay=0.1, poison_ratio=0.1, sample_num=1000, align_data_num=2000
# Lisa 特有参数：rho/alignment_step/finetune_step/guide_data_num
# - 注意：rho 在两篇含义不同（Lisa: proximal, T-Vaccine: perturbation），按你的要求此处采用 T-Vaccine 数值 3。
ALIGN_EPOCHS="${ALIGN_EPOCHS:-20}"
ALIGN_BS="${ALIGN_BS:-10}"
ALIGN_LR="${ALIGN_LR:-1e-3}"
ALIGN_WEIGHT_DECAY="${ALIGN_WEIGHT_DECAY:-0.1}"
ALIGN_SAFE_NUM="${ALIGN_SAFE_NUM:-2000}"

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
FINETUNE_BS="${FINETUNE_BS:-10}"
FINETUNE_LR="${FINETUNE_LR:-1e-5}"
FINETUNE_WEIGHT_DECAY="${FINETUNE_WEIGHT_DECAY:-0.1}"

POISON_RATIO="${POISON_RATIO:-0.1}"
SAMPLE_NUM="${SAMPLE_NUM:-1000}"
RHO="${RHO:-3}"
ALIGN_STEP="${ALIGN_STEP:-100}"
FINETUNE_STEP="${FINETUNE_STEP:-900}"
GUIDE_DATA_NUM="${GUIDE_DATA_NUM:-2000}"

# ===== 运行控制 =====
RUN_BASELINE_SFT="${RUN_BASELINE_SFT:-1}"    # 1: 运行对照组 SFT；0: 跳过
RUN_DATA_PREP="${RUN_DATA_PREP:-1}"          # 1: 自动准备数据；0: 跳过
DETACH_RUN="${DETACH_RUN:-1}"                # 1: nohup 后台运行，断开终端后继续
GPU_ID="${GPU_ID:-1,2}"                     # 支持多卡，如 "0,1" 或单卡 "0"

# ===== 路径 =====
CACHE_DIR="${CACHE_DIR:-cache}"
ALIGN_OUT="ckpt/${MODEL_SHORT}_sft"
LISA_OUT="ckpt/sst2/${MODEL_SHORT}_lisa_f_${RHO}_${POISON_RATIO}_${SAMPLE_NUM}_${ALIGN_STEP}_${FINETUNE_STEP}_${GUIDE_DATA_NUM}"
SFT_OUT="ckpt/sst2/${MODEL_SHORT}_sft_f_0_${POISON_RATIO}_${SAMPLE_NUM}_${ALIGN_STEP}_${FINETUNE_STEP}_0"

ALIGN_POISON_PRED="data/poison/${MODEL_SHORT}_sft"
LISA_POISON_PRED="data/poison/sst2/${MODEL_SHORT}_lisa_f_${RHO}_${POISON_RATIO}_${SAMPLE_NUM}_${ALIGN_STEP}_${FINETUNE_STEP}_${GUIDE_DATA_NUM}"
SFT_POISON_PRED="data/poison/sst2/${MODEL_SHORT}_sft_f_0_${POISON_RATIO}_${SAMPLE_NUM}_${ALIGN_STEP}_${FINETUNE_STEP}_0"

LISA_SST2_PRED="data/sst2/${MODEL_SHORT}_lisa_f_${RHO}_${POISON_RATIO}_${SAMPLE_NUM}_${ALIGN_STEP}_${FINETUNE_STEP}_${GUIDE_DATA_NUM}"
SFT_SST2_PRED="data/sst2/${MODEL_SHORT}_sft_f_0_${POISON_RATIO}_${SAMPLE_NUM}_${ALIGN_STEP}_${FINETUNE_STEP}_0"

mkdir -p logs ckpt ckpt/sst2 data data/poison data/sst2

if [[ "$DETACH_RUN" == "1" && -z "${_LISA_DAEMONIZED:-}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="logs/oneclick_${RUN_ID}.nohup.log"
  echo "[INFO] 以后台模式启动（nohup），日志: $LOG_FILE"
  nohup env _LISA_DAEMONIZED=1 DETACH_RUN=0 GPU_ID="$GPU_ID" \
    DOWNLOADED_MODELS_DIR="$DOWNLOADED_MODELS_DIR" MODEL_PATH="$MODEL_PATH" EVAL_MODEL_REPO="$EVAL_MODEL_REPO" EVAL_MODEL_LOCAL_DIR="$EVAL_MODEL_LOCAL_DIR" HF_ENDPOINT="$HF_ENDPOINT" \
    ALIGN_EPOCHS="$ALIGN_EPOCHS" ALIGN_BS="$ALIGN_BS" ALIGN_LR="$ALIGN_LR" ALIGN_WEIGHT_DECAY="$ALIGN_WEIGHT_DECAY" ALIGN_SAFE_NUM="$ALIGN_SAFE_NUM" \
    FINETUNE_EPOCHS="$FINETUNE_EPOCHS" FINETUNE_BS="$FINETUNE_BS" FINETUNE_LR="$FINETUNE_LR" FINETUNE_WEIGHT_DECAY="$FINETUNE_WEIGHT_DECAY" \
    POISON_RATIO="$POISON_RATIO" SAMPLE_NUM="$SAMPLE_NUM" RHO="$RHO" ALIGN_STEP="$ALIGN_STEP" FINETUNE_STEP="$FINETUNE_STEP" GUIDE_DATA_NUM="$GUIDE_DATA_NUM" \
    RUN_BASELINE_SFT="$RUN_BASELINE_SFT" RUN_DATA_PREP="$RUN_DATA_PREP" CACHE_DIR="$CACHE_DIR" \
    bash "$0" > "$LOG_FILE" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "logs/oneclick_${RUN_ID}.pid"
  echo "[INFO] 已启动，PID=$PID"
  echo "[INFO] 查看日志: tail -f $LOG_FILE"
  exit 0
fi

if [[ ! -f "$ROOT_DIR/huggingface_token.txt" ]]; then
  echo "[ERROR] 缺少 huggingface_token.txt，请在仓库根目录创建该文件并写入你的 HuggingFace Token。"
  exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ENDPOINT
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU_ID"

preflight_checks() {
  echo "[Check 1/4] 检查模型名..."
  if [[ "$MODEL_PATH" != *"Llama-2"* ]]; then
    echo "[ERROR] 当前脚本只支持 Llama2（MODEL_PATH 需包含 Llama-2）。"
    echo "        原因：当前工程里其他模型在既有 transformers 版本下可能不兼容。"
    exit 1
  fi
  echo "[Check 1/4] OK: Llama-2 模型"

  echo "[Check 2/4] 检查 GPU..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_name
    # 加 timeout 防止 nvidia-smi 挂起（驱动异常时会阻塞）
    gpu_name="$(timeout 10 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/^ *//;s/ *$//')" || true
    if [[ -z "$gpu_name" ]]; then
      echo "[WARN] nvidia-smi 超时或无输出，跳过 GPU 型号检查。"
    else
      echo "[GPU] 检测到: ${gpu_name}"
      if [[ "$gpu_name" != *"H20"* ]]; then
        echo "[WARN] 当前不是 H20（检测到: ${gpu_name}）。你要求使用 H20，建议确认 GPU 资源分配。"
      fi
    fi
  else
    echo "[WARN] 未检测到 nvidia-smi，跳过 GPU 型号检查。"
  fi

  echo "[Check 3/4] 检查 transformers 版本..."
  # 用 timeout 防止 import transformers 访问网络时卡死
  timeout 30 python -c "
import transformers
v = transformers.__version__
parts = [int(x) for x in v.split('.')[:3]]
if parts < [4,31,0]:
    raise SystemExit(f'[ERROR] transformers>=4.31.0 required, current={v}')
print(f'[OK] transformers version: {v}')
" || {
    echo "[WARN] transformers 版本检查超时或失败（exit=$?），继续运行..."
  }

  echo "[Check 4/4] 检查模型目录..."
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] 未找到本地 Llama2 模型目录: $MODEL_PATH"
    echo "        请确认模型已下载到 /data_nvme1n1/xieqiuhao/tjy/downloaded_models/Llama-2-7b-hf"
    exit 1
  fi
  echo "[Check 4/4] OK: 模型目录存在"
  echo "[Preflight] 全部检查通过"
}

prepare_eval_model() {
  echo "[EvalModel] 检查评估模型..."
  mkdir -p "$DOWNLOADED_MODELS_DIR"
  if [[ -d "$EVAL_MODEL_LOCAL_DIR" ]] && [[ -n "$(find "$EVAL_MODEL_LOCAL_DIR" -maxdepth 1 -type f 2>/dev/null)" ]]; then
    echo "[EvalModel] 已存在本地评估模型目录: $EVAL_MODEL_LOCAL_DIR"
    return
  fi

  echo "[EvalModel] 开始下载评估模型到: $EVAL_MODEL_LOCAL_DIR （这可能需要较长时间...）"
  python -c "
import os
from huggingface_hub import snapshot_download

repo_id = os.environ['EVAL_MODEL_REPO']
local_dir = os.environ['EVAL_MODEL_LOCAL_DIR']
token_path = os.path.join(os.environ['ROOT_DIR'], 'huggingface_token.txt')
with open(token_path, 'r', encoding='utf-8') as f:
    token = f.readline().strip()

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=token,
)
print(f'downloaded eval model: {repo_id} -> {local_dir}')
"
  echo "[EvalModel] 下载完成"
}

GPU_MONITOR_PID=""
GPU_MONITOR_LOG=""

start_gpu_monitor() {
  local tag="$1"
  mkdir -p logs/gpu
  GPU_MONITOR_LOG="logs/gpu/${tag}_gpu_mem.csv"
  nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total --format=csv,noheader,nounits -l 5 > "$GPU_MONITOR_LOG" 2>/dev/null &
  GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    GPU_MONITOR_PID=""
  fi
}

print_gpu_peak() {
  local tag="$1"
  local log_file="logs/gpu/${tag}_gpu_mem.csv"
  if [[ -f "$log_file" ]]; then
    local peak
    peak=$(awk -F', ' 'BEGIN{m=0} {if ($4+0>m) m=$4+0} END{print m}' "$log_file")
    echo "[GPU] ${tag} 峰值显存约: ${peak} MiB"
  fi
}

run_stage1_alignment() {
  printf "\n========== Stage 1: 安全对齐 SFT ==========\n"
  start_gpu_monitor "stage1_alignment"
  python train.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path PKU-Alignment/BeaverTails_safe \
    --output_dir "$ALIGN_OUT" \
    --num_train_epochs "$ALIGN_EPOCHS" \
    --per_device_train_batch_size "$ALIGN_BS" \
    --per_device_eval_batch_size "$ALIGN_BS" \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 100000 \
    --save_total_limit 0 \
    --learning_rate "$ALIGN_LR" \
    --weight_decay "$ALIGN_WEIGHT_DECAY" \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False \
    --cache_dir "$CACHE_DIR" \
    --guide_data_num "$ALIGN_SAFE_NUM" \
    --optimizer normal | tee "logs/stage1_alignment.log"
  stop_gpu_monitor
  print_gpu_peak "stage1_alignment"
}

eval_harmful() {
  local lora_1="$1"
  local lora_2="$2"
  local out_path="$3"
  printf "\n[Eval-HS] 输出到: %s\n" "$out_path"

  pushd poison/evaluation >/dev/null
  if [[ -n "$lora_2" ]]; then
    python pred.py \
      --lora_folder "$ROOT_DIR/$lora_1" \
      --lora_folder2 "$ROOT_DIR/$lora_2" \
      --model_folder "$MODEL_PATH" \
      --output_path "$ROOT_DIR/$out_path" | tee "$ROOT_DIR/logs/eval_hs_$(basename "$out_path").log"
  else
    python pred.py \
      --lora_folder "$ROOT_DIR/$lora_1" \
      --model_folder "$MODEL_PATH" \
      --output_path "$ROOT_DIR/$out_path" | tee "$ROOT_DIR/logs/eval_hs_$(basename "$out_path").log"
  fi

  python eval_sentiment.py \
    --input_path "$ROOT_DIR/$out_path" \
    --moderation_model_path "$EVAL_MODEL_LOCAL_DIR" | tee -a "$ROOT_DIR/logs/eval_hs_$(basename "$out_path").log"
  popd >/dev/null
}

eval_sst2() {
  local lora_1="$1"
  local lora_2="$2"
  local out_path="$3"
  printf "\n[Eval-FA] 输出到: %s\n" "$out_path"
  pushd sst2 >/dev/null
  python pred_eval.py \
    --lora_folder "$ROOT_DIR/$lora_1" \
    --lora_folder2 "$ROOT_DIR/$lora_2" \
    --model_folder "$MODEL_PATH" \
    --output_path "$ROOT_DIR/$out_path" | tee "$ROOT_DIR/logs/eval_fa_$(basename "$out_path").log"
  popd >/dev/null
}

run_stage2_lisa() {
  printf "\n========== Stage 2: Lisa 防御微调 ==========\n"
  python train.py \
    --model_name_or_path "$MODEL_PATH" \
    --lora_folder "$ALIGN_OUT" \
    --data_path PKU-Alignment/BeaverTails_dangerous \
    --output_dir "$LISA_OUT" \
    --num_train_epochs "$FINETUNE_EPOCHS" \
    --per_device_train_batch_size "$FINETUNE_BS" \
    --per_device_eval_batch_size "$FINETUNE_BS" \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 100000 \
    --save_total_limit 0 \
    --learning_rate "$FINETUNE_LR" \
    --weight_decay "$FINETUNE_WEIGHT_DECAY" \
    --warmup_ratio 0.1 \
    --lr_scheduler_type constant \
    --logging_steps 10 \
    --tf32 False \
    --eval_steps 2000 \
    --cache_dir "$CACHE_DIR" \
    --optimizer lisa \
    --evaluation_strategy steps \
    --sample_num "$SAMPLE_NUM" \
    --poison_ratio "$POISON_RATIO" \
    --label_smoothing_factor 0 \
    --benign_dataset data/sst2.json \
    --rho "$RHO" \
    --alignment_step "$ALIGN_STEP" \
    --finetune_step "$FINETUNE_STEP" \
    --guide_data_num "$GUIDE_DATA_NUM" | tee "logs/stage2_lisa.log"

  eval_harmful "$ALIGN_OUT" "$LISA_OUT" "$LISA_POISON_PRED"
  eval_sst2 "$ALIGN_OUT" "$LISA_OUT" "$LISA_SST2_PRED"
}

run_stage2_sft_baseline() {
  printf "\n========== Stage 2 Baseline: 普通 SFT 微调 ==========\n"
  python train.py \
    --model_name_or_path "$MODEL_PATH" \
    --lora_folder "$ALIGN_OUT" \
    --data_path PKU-Alignment/BeaverTails_dangerous \
    --output_dir "$SFT_OUT" \
    --num_train_epochs "$FINETUNE_EPOCHS" \
    --per_device_train_batch_size "$FINETUNE_BS" \
    --per_device_eval_batch_size "$FINETUNE_BS" \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 100000 \
    --save_total_limit 0 \
    --learning_rate "$FINETUNE_LR" \
    --weight_decay "$FINETUNE_WEIGHT_DECAY" \
    --warmup_ratio 0.1 \
    --lr_scheduler_type constant \
    --logging_steps 10 \
    --tf32 False \
    --eval_steps 1000 \
    --cache_dir "$CACHE_DIR" \
    --optimizer normal \
    --evaluation_strategy steps \
    --sample_num "$SAMPLE_NUM" \
    --poison_ratio "$POISON_RATIO" \
    --label_smoothing_factor 0 \
    --benign_dataset data/sst2.json \
    --rho 0 \
    --alignment_step "$ALIGN_STEP" \
    --finetune_step "$FINETUNE_STEP" \
    --guide_data_num 0 | tee "logs/stage2_sft_baseline.log"

  eval_harmful "$ALIGN_OUT" "$SFT_OUT" "$SFT_POISON_PRED"
  eval_sst2 "$ALIGN_OUT" "$SFT_OUT" "$SFT_SST2_PRED"
}

echo "========== Lisa Llama2-7B 复现实验启动 =========="
echo "ROOT_DIR=$ROOT_DIR"
echo "MODEL_PATH=$MODEL_PATH"
echo "EVAL_MODEL_LOCAL_DIR=$EVAL_MODEL_LOCAL_DIR"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "POISON_RATIO=$POISON_RATIO, SAMPLE_NUM=$SAMPLE_NUM, RHO=$RHO"
echo "ALIGN_STEP=$ALIGN_STEP, FINETUNE_STEP=$FINETUNE_STEP, GUIDE_DATA_NUM=$GUIDE_DATA_NUM, ALIGN_SAFE_NUM=$ALIGN_SAFE_NUM"

echo "[Step 1/5] Preflight checks..."
preflight_checks
echo "[Step 2/5] 准备评估模型..."
prepare_eval_model

if [[ "$RUN_DATA_PREP" == "1" ]]; then
  echo "[Step 3/5] 准备数据集..."
  bash script/repro/prepare_datasets.sh 2>&1 | tee logs/data_prepare.log
else
  echo "[Skip] RUN_DATA_PREP=0，跳过数据准备。"
fi

echo "[Step 4/5] Stage 1: 安全对齐..."
run_stage1_alignment
eval_harmful "$ALIGN_OUT" "" "$ALIGN_POISON_PRED"
echo "[Step 5/5] Stage 2: 微调 + 评测..."
run_stage2_lisa

if [[ "$RUN_BASELINE_SFT" == "1" ]]; then
  run_stage2_sft_baseline
else
  echo "[Skip] RUN_BASELINE_SFT=0，跳过普通 SFT 对照组。"
fi

printf "\n========== 全流程完成 ==========\n"
echo "关键输出目录："
echo "- 对齐模型: $ALIGN_OUT"
echo "- Lisa 模型: $LISA_OUT"
echo "- 有害评测结果(对齐): ${ALIGN_POISON_PRED}_sentiment_eval.json"
echo "- 有害评测结果(Lisa): ${LISA_POISON_PRED}_sentiment_eval.json"
echo "- SST2 评测结果(Lisa): $LISA_SST2_PRED"
if [[ "$RUN_BASELINE_SFT" == "1" ]]; then
  echo "- 有害评测结果(SFT): ${SFT_POISON_PRED}_sentiment_eval.json"
  echo "- SST2 评测结果(SFT): $SFT_SST2_PRED"
fi
