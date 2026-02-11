#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

find third_party/mlx-c/mlx/c -maxdepth 1 -name '*.h' -print0 \
  | xargs -0 sed -nE 's/^[[:space:]]*([_a-zA-Z0-9*[:space:]]+[[:space:]]+)?(mlx_[a-zA-Z0-9_]+)[[:space:]]*\(.*/\2/p' \
  | sort -u > "$tmp_dir/c_functions.txt"

grep -oE "'mlx_[a-zA-Z0-9_]+'" lib/src/generated/mlx_bindings.dart \
  | tr -d "'" \
  | sort -u > "$tmp_dir/generated_symbols.txt"

comm -23 "$tmp_dir/c_functions.txt" "$tmp_dir/generated_symbols.txt" > "$tmp_dir/missing.txt"

c_count="$(wc -l < "$tmp_dir/c_functions.txt" | tr -d ' ')"
gen_count="$(wc -l < "$tmp_dir/generated_symbols.txt" | tr -d ' ')"
missing_count="$(wc -l < "$tmp_dir/missing.txt" | tr -d ' ')"

echo "C functions:        $c_count"
echo "Generated symbols:  $gen_count"
echo "Missing functions:  $missing_count"

if [[ "$missing_count" -gt 0 ]]; then
  echo
  echo "Missing:"
  cat "$tmp_dir/missing.txt"
  exit 1
fi
