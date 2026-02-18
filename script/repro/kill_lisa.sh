#!/usr/bin/env bash
# ============================================================
# kill_lisa.sh — 查找并杀死 Lisa 一键复现相关的所有进程
# 用法:
#   bash script/repro/kill_lisa.sh          # 列出匹配进程（预览，不杀）
#   bash script/repro/kill_lisa.sh --kill   # 实际杀死进程
#   bash script/repro/kill_lisa.sh --force  # SIGKILL 强杀
# ============================================================

set -euo pipefail

ACTION="${1:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "===== 查找 Lisa 相关进程 ====="

# 1. 从 pid 文件中读取记录的 PID
echo ""
echo "--- 从 logs/*.pid 文件中查找 ---"
for pid_file in "$ROOT_DIR"/logs/oneclick_*.pid; do
  [[ -f "$pid_file" ]] || continue
  RECORDED_PID="$(cat "$pid_file")"
  if ps -p "$RECORDED_PID" &>/dev/null; then
    echo "[ALIVE] PID=$RECORDED_PID (来自 $pid_file)"
    # 同时列出其子进程树
    pstree -p "$RECORDED_PID" 2>/dev/null || ps --ppid "$RECORDED_PID" -o pid,ppid,cmd 2>/dev/null || true
  else
    echo "[DEAD]  PID=$RECORDED_PID (来自 $pid_file) — 进程已不存在"
  fi
done

# 2. 通过 ps 搜索关键字匹配的进程
echo ""
echo "--- 通过关键字搜索相关进程 ---"
PATTERNS=(
  "run_lisa_llama2_7b_oneclick"
  "_LISA_DAEMONIZED"
  "prepare_datasets.sh"
  "train.py.*--optimizer"
  "pred.py.*--lora_folder"
  "eval_sentiment.py"
  "pred_eval.py.*--lora_folder"
  "nvidia-smi.*-l"
)

FOUND_PIDS=()
for pat in "${PATTERNS[@]}"; do
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    pid=$(echo "$line" | awk '{print $2}')
    # 排除 grep 自身和当前脚本
    echo "$line" | grep -q "grep" && continue
    echo "$line" | grep -q "kill_lisa.sh" && continue
    FOUND_PIDS+=("$pid")
    echo "  $line"
  done < <(ps aux 2>/dev/null | grep -E "$pat" || true)
done

# 去重
UNIQUE_PIDS=($(printf '%s\n' "${FOUND_PIDS[@]}" | sort -un))

if [[ ${#UNIQUE_PIDS[@]} -eq 0 ]]; then
  echo ""
  echo "[INFO] 未找到任何 Lisa 相关进程。可能已退出。"
  echo ""
  echo "提示: 也可手动检查 GPU 占用:"
  echo "  nvidia-smi"
  echo "  fuser -v /dev/nvidia*   # 查看占用 GPU 的进程"
  exit 0
fi

echo ""
echo "共找到 ${#UNIQUE_PIDS[@]} 个相关进程: ${UNIQUE_PIDS[*]}"

# 3. 执行杀进程
if [[ "$ACTION" == "--kill" ]]; then
  echo ""
  echo "[ACTION] 发送 SIGTERM (优雅终止)..."
  for pid in "${UNIQUE_PIDS[@]}"; do
    echo "  kill $pid"
    kill "$pid" 2>/dev/null || echo "  (PID $pid 已不存在)"
  done
  sleep 2
  # 检查是否还活着
  STILL_ALIVE=()
  for pid in "${UNIQUE_PIDS[@]}"; do
    if ps -p "$pid" &>/dev/null; then
      STILL_ALIVE+=("$pid")
    fi
  done
  if [[ ${#STILL_ALIVE[@]} -gt 0 ]]; then
    echo "[WARN] 以下进程未响应 SIGTERM: ${STILL_ALIVE[*]}"
    echo "       使用 --force 强杀。"
  else
    echo "[OK] 所有进程已终止。"
  fi

elif [[ "$ACTION" == "--force" ]]; then
  echo ""
  echo "[ACTION] 发送 SIGKILL (强杀)..."
  for pid in "${UNIQUE_PIDS[@]}"; do
    echo "  kill -9 $pid"
    kill -9 "$pid" 2>/dev/null || echo "  (PID $pid 已不存在)"
  done
  sleep 1
  echo "[OK] 已发送 SIGKILL。"

else
  echo ""
  echo "以上为预览，未杀死任何进程。"
  echo "操作方式:"
  echo "  bash script/repro/kill_lisa.sh --kill    # 优雅终止 (SIGTERM)"
  echo "  bash script/repro/kill_lisa.sh --force   # 强制终止 (SIGKILL)"
fi

# 4. 检查 GPU 占用
echo ""
echo "===== 当前 GPU 占用 ====="
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || echo "(nvidia-smi 不可用)"
