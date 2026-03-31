#!/usr/bin/env bash
set -euo pipefail

# 依次跑所有 model_type（固定 3 层），每个训练完成后立刻跑目录评估，
# 抽取宏平均 tau/spearman/r2/mape 写入 CSV。
#
# 用法（在 python/ 目录下）：
#   bash run_all_model_types_layers3.sh
#
# 你给的示例（其中一个配置）：
#   python train.py --data_dir ../processed_dataset/ --save tpn_mh.pt --layers 3 --model_type mpnn_mh
#
# 期望记录的某次结果（示例值）：
#   tau=0.9019 spearman=0.9851 r2=0.9715 mape=16.75%

DATA_DIR="${DATA_DIR:-../processed_dataset/}"
LAYERS="${LAYERS:-3}"
SEED="${SEED:-42}"
HIDDEN="${HIDDEN:-128}"
BATCH="${BATCH:-2}"
EPOCHS="${EPOCHS:-500}"
MIN_EPOCHS="${MIN_EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
LR="${LR:-1e-3}"

# train.py / evaluate.py 中硬编码的 choices
MODEL_TYPES=(mpnn mpnn_mh gcn gat sage gin)

RUN_DIR="${RUN_DIR:-runs_layers${LAYERS}}"
mkdir -p "$RUN_DIR"

CSV_PATH="${CSV_PATH:-${RUN_DIR}/results.csv}"
if [[ ! -f "$CSV_PATH" ]]; then
  printf "model_type,layers,seed,hidden,batch,lr,graph_loss_weight,ckpt_path,tau,spearman,r2,mape,eval_log\n" > "$CSV_PATH"
fi

parse_metrics_from_eval_log() {
  local log_path="$1"
  python - "$log_path" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
txt = log_path.read_text(encoding="utf-8", errors="replace")

# 目标行形如：
# 全图监督节点（每图 y_valid） — 宏平均:  tau=0.9019  spearman=0.9851  r2=0.9715  mape=16.75%  mae=...  rmse=...
pat = re.compile(
    r"全图监督节点（每图 y_valid）\s*—\s*宏平均:\s*"
    r"tau=(?P<tau>[-+0-9.]+)\s*"
    r"spearman=(?P<sp>[-+0-9.]+)\s*"
    r"r2=(?P<r2>[-+0-9.]+)\s*"
    r"mape=(?P<mape>[-+0-9.]+)%"
)
m = pat.search(txt)
if not m:
    # 兜底：如果输出语言/空格略有差异，尝试更宽松的匹配
    pat2 = re.compile(
        r"宏平均:.*?tau=(?P<tau>[-+0-9.]+).*?"
        r"spearman=(?P<sp>[-+0-9.]+).*?"
        r"r2=(?P<r2>[-+0-9.]+).*?"
        r"mape=(?P<mape>[-+0-9.]+)%"
    )
    m = pat2.search(txt)
if not m:
    print("nan,nan,nan,nan", end="")
    sys.exit(0)

print(f"{m.group('tau')},{m.group('sp')},{m.group('r2')},{m.group('mape')}", end="")
PY
}

for model_type in "${MODEL_TYPES[@]}"; do
  ckpt_path="${RUN_DIR}/${model_type}_layers${LAYERS}.pt"
  train_log="${RUN_DIR}/train_${model_type}_layers${LAYERS}.log"
  eval_log="${RUN_DIR}/eval_${model_type}_layers${LAYERS}.log"

  echo "============================================================"
  echo "[TRAIN] model_type=${model_type} layers=${LAYERS} -> ${ckpt_path}"
  echo "log: ${train_log}"

  python train.py \
    --data_dir "$DATA_DIR" \
    --save "$ckpt_path" \
    --layers "$LAYERS" \
    --model_type "$model_type" \
    --seed "$SEED" \
    --hidden "$HIDDEN" \
    --batch "$BATCH" \
    --epochs "$EPOCHS" \
    --min_epochs "$MIN_EPOCHS" \
    --patience "$PATIENCE" \
    --lr "$LR" \
    2>&1 | tee "$train_log"

  echo "------------------------------------------------------------"
  echo "[EVAL] model_type=${model_type} -> ${eval_log}"

  python evaluate.py \
    --model "$ckpt_path" \
    --model_type auto \
    --npz_dir "$DATA_DIR" \
    --quiet \
    2>&1 | tee "$eval_log"

  metrics_csv="$(parse_metrics_from_eval_log "$eval_log")"
  IFS=',' read -r tau spearman r2 mape <<<"$metrics_csv"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$model_type" "$LAYERS" "$SEED" "$HIDDEN" "$BATCH" "$LR" "$GRAPH_LOSS_WEIGHT" \
    "$ckpt_path" "$tau" "$spearman" "$r2" "$mape" "$eval_log" >> "$CSV_PATH"

  echo "[CSV] appended: ${model_type} tau=${tau} spearman=${spearman} r2=${r2} mape=${mape}%"
done

echo "============================================================"
echo "DONE. CSV saved to: ${CSV_PATH}"
