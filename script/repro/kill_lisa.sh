#!/usr/bin/env bash
# ============================================================
# kill_lisa.sh — 查找并杀死【当前项目目录下】Lisa 相关的所有进程
# 通过项目路径精确匹配，不会误杀其他用户/其他目录的同名进程
#
# 用法:
#   bash script/repro/kill_lisa.sh          # 列出匹配进程（预览，不杀）
#   bash script/repro/kill_lisa.sh --kill   # 实际杀死进程
#   bash script/repro/kill_lisa.sh --force  # SIGKILL 强杀
# ============================================================

set -eo pipefail

ACTION="${1:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "===== 查找 Lisa 相关进程（限定项目路径: $ROOT_DIR）====="

FOUND_PIDS=()

# ---------- 方法 1: 从 pid 文件 + 进程树 ----------
echo ""
echo "--- 从 logs/*.pid 文件中查找 ---"
for pid_file in "$ROOT_DIR"/logs/oneclick_*.pid; do
  [[ -f "$pid_file" ]] || continue
  RECORDED_PID="$(cat "$pid_file")"
  if ps -p "$RECORDED_PID" &>/dev/null; then
    echo "[ALIVE] PID=$RECORDED_PID (来自 $(basename "$pid_file"))"
    FOUND_PIDS+=("$RECORDED_PID")
    # 收集其所有子进程
    while IFS= read -r child_pid; do
      [[ -n "$child_pid" ]] && FOUND_PIDS+=("$child_pid")
    done < <(ps --ppid "$RECORDED_PID" -o pid= 2>/dev/null || true)
    pstree -p "$RECORDED_PID" 2>/dev/null || ps --ppid "$RECORDED_PID" -o pid,ppid,cmd 2>/dev/null || true
  else
    echo "[DEAD]  PID=$RECORDED_PID (来自 $(basename "$pid_file")) — 进程已不存在"
  fi
done

# ---------- 方法 2: 通过 /proc/PID/cwd 或命令行中的项目路径精确匹配 ----------
echo ""
echo "--- 通过项目路径精确匹配进程 ---"

# 候选关键词（宽松匹配，后续会用路径二次过滤）
CANDIDATE_PATTERNS=(
  "run_lisa_llama2_7b_oneclick"
  "_LISA_DAEMONIZED"
  "prepare_datasets.sh"
  "train.py"
  "pred.py"
  "eval_sentiment.py"
  "pred_eval.py"
  "nvidia-smi.*-l"
)

COMBINED_PATTERN="$(IFS='|'; echo "${CANDIDATE_PATTERNS[*]}")"

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  # 排除 grep 自身和当前脚本
  echo "$line" | grep -qE "grep|kill_lisa\.sh" && continue

  pid=$(echo "$line" | awk '{print $2}')

  # ===== 核心：用项目路径做二次精确过滤 =====
  matched=false

  # 检查 1: 命令行参数中是否包含当前项目路径
  cmdline=""
  if [[ -f "/proc/$pid/cmdline" ]]; then
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  fi
  if [[ -z "$cmdline" ]]; then
    # fallback: 从 ps 输出取
    cmdline="$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i}')"
  fi

  if echo "$cmdline" | grep -qF "$ROOT_DIR"; then
    matched=true
  fi

  # 检查 2: 进程的工作目录 (cwd) 是否在项目路径下
  if [[ "$matched" == "false" ]] && [[ -d "/proc/$pid" ]]; then
    proc_cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    if [[ "$proc_cwd" == "$ROOT_DIR"* ]]; then
      matched=true
    fi
  fi

  if [[ "$matched" == "true" ]]; then
    FOUND_PIDS+=("$pid")
    echo "  [PID=$pid] $cmdline"
  fi
done < <(ps aux 2>/dev/null | grep -E "$COMBINED_PATTERN" || true)

# ---------- 去重 ----------
if [[ ${#FOUND_PIDS[@]} -eq 0 ]]; then
  echo ""
  echo "[INFO] 未找到任何属于当前项目的 Lisa 相关进程。"
  echo ""
  echo "提示: 也可手动检查 GPU 占用:"
  echo "  nvidia-smi"
  echo "  fuser -v /dev/nvidia*"
  exit 0
fi

UNIQUE_PIDS=($(printf '%s\n' "${FOUND_PIDS[@]}" | sort -un))

echo ""
echo "共找到 ${#UNIQUE_PIDS[@]} 个属于当前项目的相关进程: ${UNIQUE_PIDS[*]}"

# ---------- 执行杀进程 ----------
if [[ "$ACTION" == "--kill" ]]; then
  echo ""
  echo "[ACTION] 发送 SIGTERM (优雅终止)..."
  for pid in "${UNIQUE_PIDS[@]}"; do
    echo "  kill $pid"
    kill "$pid" 2>/dev/null || echo "  (PID $pid 已不存在)"
  done
  sleep 2
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
