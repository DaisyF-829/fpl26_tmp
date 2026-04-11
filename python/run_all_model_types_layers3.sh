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
#
# PL 基线（用 tnode_pl_arrival，无则 tnode_pl_time；图级 CPD = 全图最大 PL = pl_max）：
#   脚本末尾会跑：python evaluate.py --pl_baseline --npz_dir "$DATA_DIR2" --quiet
#   仅跑 PL、跳过各 model 评估：  SKIP_MODEL_EVAL=1 bash run_all_model_types_layers3.sh
#   不跑 PL 段：                  RUN_PL_BASELINE=0 bash run_all_model_types_layers3.sh
#   单条命令： python evaluate.py --pl_baseline --npz_dir /path/to/npz_dir --quiet

export CUDA_VISIBLE_DEVICES=3

SKIP_MODEL_EVAL="${SKIP_MODEL_EVAL:-0}"
RUN_PL_BASELINE="${RUN_PL_BASELINE:-1}"

DATA_DIR="${DATA_DIR:-/home/yfdai/fpl_retiming/processed_dataset/}"
DATA_DIR2="${DATA_DIR2:-/home/yfdai/fpl_retiming/dataset_new/processed_dataset/}"
LAYERS="${LAYERS:-3}"
SEED="${SEED:-42}"
HIDDEN="${HIDDEN:-128}"
BATCH="${BATCH:-2}"
EPOCHS="${EPOCHS:-500}"
MIN_EPOCHS="${MIN_EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
LR="${LR:-1e-3}"

# train.py / evaluate.py 中硬编码的 choices
MODEL_TYPES=(mpnn_delayprop_s1 mpnn_delayprop_s2)

RUN_DIR="${RUN_DIR:-runs_new}"
mkdir -p "$RUN_DIR"

CSV_PATH="${CSV_PATH:-${RUN_DIR}/results.csv}"
if [[ ! -f "$CSV_PATH" ]]; then
  printf "model_type,layers,seed,hidden,batch,lr,ckpt_path,eval_log,success,files,cov_mean,prec_mean,graph_mae_log_mean,graph_rmse_log_mean,graph_cpd_rel_pct_mean,all_tau,all_spearman,all_r2,all_mape,all_mae,all_rmse,leaf_graphs_ok,leaf_tau,leaf_spearman,leaf_r2,leaf_mape,leaf_mae,leaf_rmse\n" > "$CSV_PATH"
fi

# 分组均值 CSV（每个 model_type 会输出多行：每个 group 一行）
GROUP_CSV_PATH="${GROUP_CSV_PATH:-${RUN_DIR}/group_results.csv}"
if [[ ! -f "$GROUP_CSV_PATH" ]]; then
  printf "model_type,layers,seed,hidden,batch,lr,ckpt_path,eval_log,group,n,cov_mean,prec_mean,topk,topk_found,topk_total,topk_ratio,leaf_tau_mean,leaf_sp_mean,leaf_r2_mean,leaf_mape_mean_pct,cpd_rel_pct_mean\n" > "$GROUP_CSV_PATH"
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

parse_group_means_from_eval_log() {
  local log_path="$1"
  python - "$log_path" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
txt = log_path.read_text(encoding="utf-8", errors="replace")

lines = txt.splitlines()

# 兼容 top10/top20 的字段名；优先匹配 top20_path，其次 top10_path
pat_full = re.compile(
    r"^\[(?P<group>[^\]]+)\]:\s*"
    r"n=(?P<n>\d+)\s+"
    r"cov_mean=(?P<cov>[-+0-9.]+)\s+"
    r"prec_mean=(?P<prec>[-+0-9.]+)\s+"
    r"(?:top(?P<topk1>\d+)_path|top(?P<topk2>\d+)_path)=(?P<found>\d+)/(?P<total>\d+)\((?P<ratio>[-+0-9.]+)\)\s+"
    r"(?:(?:leaf_tau_mean|tau_all_mean)=?(?P<tau>[-+0-9.]+)\s+)?"
    r"leaf_sp_mean=(?P<leaf_sp>[-+0-9.]+)\s+"
    r"leaf_r2_mean=(?P<leaf_r2>[-+0-9.]+)\s+"
    r"leaf_mape_mean=(?P<leaf_mape>[-+0-9.]+)%\s+"
    r"cpd_rel%_mean=(?P<cpd>[-+0-9.]+)\s*$"
)

pat_zero = re.compile(r"^\[(?P<group>[^\]]+)\]:\s*\(0 files\)\s*$")

out = []
for ln in lines:
    m0 = pat_zero.match(ln.strip())
    if m0:
        g = m0.group("group")
        out.append(",".join([g, "0", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan"]))
        continue
    m = pat_full.match(ln.strip())
    if not m:
        continue
    gd = m.groupdict()
    topk = gd.get("topk1") or gd.get("topk2") or "nan"
    # tau：新版本输出 leaf_tau_mean；旧版本可能是 tau_all_mean（如果你没改 evaluate.py），这里也兜底读取
    tau = gd.get("tau") or "nan"
    out.append(
        ",".join(
            [
                gd["group"],
                gd["n"],
                gd["cov"],
                gd["prec"],
                topk,
                gd["found"],
                gd["total"],
                gd["ratio"],
                tau,
                gd["leaf_sp"],
                gd["leaf_r2"],
                gd["leaf_mape"],
                gd["cpd"],
            ]
        )
    )

print("\n".join(out), end="")
PY
}

if [[ "$SKIP_MODEL_EVAL" != "1" ]]; then
for model_type in "${MODEL_TYPES[@]}"; do
  ckpt_path="${RUN_DIR}/${model_type}_layers${LAYERS}.pt"
  train_log="${RUN_DIR}/train_${model_type}_layers${LAYERS}.log"
  eval_log="${RUN_DIR}/eval_${model_type}_layers${LAYERS}_fpl.log"

  # echo "============================================================"
  # echo "[TRAIN] model_type=${model_type} layers=${LAYERS} -> ${ckpt_path}"
  # echo "log: ${train_log}"

  # python train.py \
  #   --data_dir "$DATA_DIR" \
  #   --save "$ckpt_path" \
  #   --layers "$LAYERS" \
  #   --model_type "$model_type" \
  #   --seed "$SEED" \
  #   --hidden "$HIDDEN" \
  #   --batch "$BATCH" \
  #   --epochs "$EPOCHS" \
  #   --min_epochs "$MIN_EPOCHS" \
  #   --patience "$PATIENCE" \
  #   --lr "$LR" \
  #   2>&1 | tee "$train_log"

  echo "------------------------------------------------------------"
  echo "[EVAL] model_type=${model_type} -> ${eval_log}"

  python evaluate.py \
    --model "$ckpt_path" \
    --model_type auto \
    --npz_dir "$DATA_DIR2" \
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

  # 分组均值解析并追加到 group_results.csv
  group_rows="$(parse_group_means_from_eval_log "$eval_log")"
  if [[ -n "${group_rows}" ]]; then
    # group_rows 每行：group,n,cov,prec,topk,found,total,ratio,leaf_tau,leaf_sp,leaf_r2,leaf_mape,cpd_rel
    while IFS= read -r gr; do
      [[ -z "$gr" ]] && continue
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$model_type" "$LAYERS" "$SEED" "$HIDDEN" "$BATCH" "$LR" \
        "$ckpt_path" "$eval_log" \
        "$gr" \
        >> "$GROUP_CSV_PATH"
    done <<< "$group_rows"
    echo "[GROUP_CSV] appended: ${model_type} -> ${GROUP_CSV_PATH}"
  else
    echo "[GROUP_CSV] no group lines found in: ${eval_log}"
  fi
done
else
  echo "SKIP_MODEL_EVAL=1：跳过各 model_type 的 evaluate。"
fi

# PL 基线：--pl_baseline（节点=PL 到达 tnode_pl_arrival；图级 CPD=pl_max=全图最大 PL）
if [[ "$RUN_PL_BASELINE" == "1" ]]; then
eval_log_pl="${RUN_DIR}/eval_pl_baseline_layers${LAYERS}_fpl.log"
ckpt_pl="-"
echo "------------------------------------------------------------"
echo "[EVAL PL baseline] -> ${eval_log_pl}"

python evaluate.py \
  --pl_baseline \
  --npz_dir "$DATA_DIR2" \
  --quiet \
  2>&1 | tee "$eval_log_pl"

summary_csv="$(parse_summary_from_eval_log "$eval_log_pl")"
IFS=',' read -r \
  success files cov_mean prec_mean \
  graph_mae_log_mean graph_rmse_log_mean graph_cpd_rel_pct_mean \
  all_tau all_spearman all_r2 all_mape all_mae all_rmse \
  leaf_graphs_ok leaf_tau leaf_spearman leaf_r2 leaf_mape leaf_mae leaf_rmse \
  <<<"$summary_csv"

printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
  "pl_baseline" "$LAYERS" "$SEED" "$HIDDEN" "$BATCH" "$LR" \
  "$ckpt_pl" "$eval_log_pl" \
  "$success" "$files" "$cov_mean" "$prec_mean" \
  "$graph_mae_log_mean" "$graph_rmse_log_mean" "$graph_cpd_rel_pct_mean" \
  "$all_tau" "$all_spearman" "$all_r2" "$all_mape" "$all_mae" "$all_rmse" \
  "$leaf_graphs_ok" "$leaf_tau" "$leaf_spearman" "$leaf_r2" "$leaf_mape" "$leaf_mae" "$leaf_rmse" \
  >> "$CSV_PATH"

echo "[CSV] appended: pl_baseline all_tau=${all_tau} all_spearman=${all_spearman} all_r2=${all_r2} all_mape=${all_mape}%"

group_rows="$(parse_group_means_from_eval_log "$eval_log_pl")"
if [[ -n "${group_rows}" ]]; then
  while IFS= read -r gr; do
    [[ -z "$gr" ]] && continue
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "pl_baseline" "$LAYERS" "$SEED" "$HIDDEN" "$BATCH" "$LR" \
      "$ckpt_pl" "$eval_log_pl" \
      "$gr" \
      >> "$GROUP_CSV_PATH"
  done <<< "$group_rows"
  echo "[GROUP_CSV] appended: pl_baseline -> ${GROUP_CSV_PATH}"
else
  echo "[GROUP_CSV] no group lines found in: ${eval_log_pl}"
fi
else
  echo "RUN_PL_BASELINE=0：跳过 PL 基线评估。"
fi

echo "============================================================"
echo "DONE. CSV saved to: ${CSV_PATH}"
echo "DONE. GROUP CSV saved to: ${GROUP_CSV_PATH}"
