#!/usr/bin/env bash
# 从当前目录递归查找所有 .npz，复制到 processed_dataset/，
# 目标文件名：相对路径中的 / 换成 _，例如 ./a/b/c.npz -> processed_dataset/a_b_c.npz
set -euo pipefail

DEST="${1:-processed_dataset}"
mkdir -p "$DEST"

# 不扫描输出目录，避免二次运行把已扁平化的 .npz 再拷一层
prune_path="./${DEST}"

count=0
while IFS= read -r -d '' f; do
  rel="${f#./}"
  name="${rel//\//_}"
  dest="$DEST/$name"
  cp -- "$f" "$dest"
  echo "$f -> $dest"
  count=$((count + 1))
done < <(find . \( -path "$prune_path" -prune \) -o \( -type f -name '*.npz' -print0 \))

echo "完成：共复制 $count 个文件到 $DEST/"
