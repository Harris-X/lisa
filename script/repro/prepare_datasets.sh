#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

echo "[1/3] 预拉取 HuggingFace 数据集缓存..."
# 注意：sst2 必须用 "stanfordnlp/sst2" 而不是 "sst2"，
#       因为项目根目录下有同名 sst2/ 文件夹，datasets 库会优先匹配本地目录导致报错。
#       ag_news 同理用 "fancyzhx/ag_news" 避免潜在冲突。
python - <<'PY'
from datasets import load_dataset

targets = [
    ("PKU-Alignment/BeaverTails", None),
    ("stanfordnlp/sst2", None),
    ("fancyzhx/ag_news", None),
    ("openai/gsm8k", "main"),
]

for name, config in targets:
    if config is None:
        ds = load_dataset(name)
        print(f"downloaded: {name}, splits={list(ds.keys())}")
    else:
        ds = load_dataset(name, config)
        print(f"downloaded: {name}/{config}, splits={list(ds.keys())}")
PY

echo "[2/3] 构建下游任务 json 数据..."
mkdir -p data

pushd sst2 >/dev/null
python build_dataset.py
popd >/dev/null

pushd agnews >/dev/null
python build_dataset.py
popd >/dev/null

pushd gsm8k >/dev/null
python build_dataset.py
popd >/dev/null

echo "[3/3] 数据准备完成。"
echo "- 生成文件: data/sst2.json, data/agnews.json, data/gsm8k.json"
echo "- HF_ENDPOINT: $HF_ENDPOINT"
echo "- 数据缓存: $HF_HOME"
