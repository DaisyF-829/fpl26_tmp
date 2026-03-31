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
MODEL_TYPES=(mpnn mpnn_mh mpnn_delayprop gcn gat sage gin)

RUN_DIR="${RUN_DIR:-runs_layers${LAYERS}}"
mkdir -p "$RUN_DIR"

CSV_PATH="${CSV_PATH:-${RUN_DIR}/results.csv}"
if [[ ! -f "$CSV_PATH" ]]; then
  printf "model_type,layers,seed,hidden,batch,lr,ckpt_path,eval_log,success,files,cov_mean,prec_mean,graph_mae_log_mean,graph_rmse_log_mean,graph_cpd_rel_pct_mean,all_tau,all_spearman,all_r2,all_mape,all_mae,all_rmse,leaf_graphs_ok,leaf_tau,leaf_spearman,leaf_r2,leaf_mape,leaf_mae,leaf_rmse\n" > "$CSV_PATH"
fi

parse_summary_from_eval_log() {
  local log_path="$1"
  python - "$log_path" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
txt = log_path.read_text(encoding="utf-8", errors="replace")

def g(pat: str):
    m = re.search(pat, txt, flags=re.M)
    return m.group(1) if m else "nan"

# 汇总头：成功数/总数
success = g(r"汇总（宏平均，按文件）\s*成功\s*(\d+)\s*/\s*(\d+)\s*个")
files = g(r"汇总（宏平均，按文件）\s*成功\s*\d+\s*/\s*(\d+)\s*个")
if success == "nan":
    # 兜底：允许空格差异
    m = re.search(r"成功\s*(\d+)\s*/\s*(\d+)", txt)
    if m:
        success, files = m.group(1), m.group(2)

# Coverage / Precision 均值
cov_mean = g(r"Coverage\s*均值:\s*([-+0-9.]+)")
prec_mean = g(r"Precision\s*均值:\s*([-+0-9.]+)")

# 图头(log(CPD)) 均值
graph_mae_log_mean = g(r"MAE\(log\)\s*均值=([-+0-9.]+)")
graph_rmse_log_mean = g(r"RMSE\(log\)\s*均值=([-+0-9.]+)")
graph_cpd_rel_pct_mean = g(r"CPD\s*相对误差%\s*均值=([-+0-9.]+)")

# 全图监督节点（m_all）宏平均
all_tau = g(r"全图监督节点（每图 y_valid）.*?tau=([-+0-9.]+)")
all_spearman = g(r"全图监督节点（每图 y_valid）.*?spearman=([-+0-9.]+)")
all_r2 = g(r"全图监督节点（每图 y_valid）.*?r2=([-+0-9.]+)")
all_mape = g(r"全图监督节点（每图 y_valid）.*?mape=([-+0-9.]+)%")
all_mae = g(r"全图监督节点（每图 y_valid）.*?mae=([-+0-9.]+)")
all_rmse = g(r"全图监督节点（每图 y_valid）.*?rmse=([-+0-9.]+)")

# 末节点宏平均（如果参与平均的图数=... 会出现）
leaf_graphs_ok = g(r"参与平均的图数=(\d+)\s*/\s*\d+")
leaf_tau = g(r"末节点.*?tau=([-+0-9.]+)")
leaf_spearman = g(r"末节点.*?spearman=([-+0-9.]+)")
leaf_r2 = g(r"末节点.*?r2=([-+0-9.]+)")
leaf_mape = g(r"末节点.*?mape=([-+0-9.]+)%")
leaf_mae = g(r"末节点.*?mae=([-+0-9.]+)")
leaf_rmse = g(r"末节点.*?rmse=([-+0-9.]+)")

print(
    ",".join(
        [
            success,
            files,
            cov_mean,
            prec_mean,
            graph_mae_log_mean,
            graph_rmse_log_mean,
            graph_cpd_rel_pct_mean,
            all_tau,
            all_spearman,
            all_r2,
            all_mape,
            all_mae,
            all_rmse,
            leaf_graphs_ok,
            leaf_tau,
            leaf_spearman,
            leaf_r2,
            leaf_mape,
            leaf_mae,
            leaf_rmse,
        ]
    ),
    end="",
)
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

  summary_csv="$(parse_summary_from_eval_log "$eval_log")"
  IFS=',' read -r \
    success files cov_mean prec_mean \
    graph_mae_log_mean graph_rmse_log_mean graph_cpd_rel_pct_mean \
    all_tau all_spearman all_r2 all_mape all_mae all_rmse \
    leaf_graphs_ok leaf_tau leaf_spearman leaf_r2 leaf_mape leaf_mae leaf_rmse \
    <<<"$summary_csv"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$model_type" "$LAYERS" "$SEED" "$HIDDEN" "$BATCH" "$LR" \
    "$ckpt_path" "$eval_log" \
    "$success" "$files" "$cov_mean" "$prec_mean" \
    "$graph_mae_log_mean" "$graph_rmse_log_mean" "$graph_cpd_rel_pct_mean" \
    "$all_tau" "$all_spearman" "$all_r2" "$all_mape" "$all_mae" "$all_rmse" \
    "$leaf_graphs_ok" "$leaf_tau" "$leaf_spearman" "$leaf_r2" "$leaf_mape" "$leaf_mae" "$leaf_rmse" \
    >> "$CSV_PATH"

  echo "[CSV] appended: ${model_type} all_tau=${all_tau} all_spearman=${all_spearman} all_r2=${all_r2} all_mape=${all_mape}%"
done

echo "============================================================"
echo "DONE. CSV saved to: ${CSV_PATH}"
