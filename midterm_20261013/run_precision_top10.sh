#!/bin/bash
# run_precision_top10.sh -- top10 case 精度批跑(框架原生链路)
# yaml: dy^dy2stcinn_eval_inputspec.yml  dy_eval vs dy2st_eval_cinn_inputspec
#       precision: delta=0.01 rtol=0.01 (中期验收要求)
# 每 case 独立子进程跑 single_case_run_runner.py(调 SingleTest._case_run),
# 失败时进程 assert False(rc!=0), 汇总退出码 + 日志中的 precision 结果
# 产物: /work/precision_results_top10.csv + /work/case_logs_top10_prec/
set -u

BASE=/work/PaddleTest/framework/e2e/PaddleLT_new
CSV="$(cd "$(dirname "$0")" && pwd)/top10_cases.csv"
OUT=/work/precision_results_top10.csv
LOGS=/work/case_logs_top10_prec
YAML="yaml/dy^dy2stcinn_eval_inputspec.yml"

# 与批跑/动态shape验收一致的环境(CINN 编译所需)
export MIN_GRAPH_SIZE=0 FLAGS_cinn_debug=1 FLAGS_prim_forward_blacklist=pd_op.dropout
export PLT_DEVICE_ID=0 PLT_GET_NV_MEMORY=False PLT_BM_DB=non-db
export FRAMEWORK=paddle PLT_SET_DEVICE=gpu
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$BASE
cd "$BASE"
mkdir -p "$LOGS"

echo "case,precision,rc,max_abs_diff,atol,rtol,tol_at_max_diff,diff/tol" > "$OUT"

# 从 top10_cases.csv 逐行取 case 列(带 [i/N] 进度计数与时间戳, 格式对齐 run.py csv-batch)
i=0
total=$(tail -n +2 "$CSV" | wc -l)
tail -n +2 "$CSV" | cut -d, -f1 | while IFS= read -r case_id; do
  [ -z "$case_id" ] && continue
  i=$((i+1))
  rel_path="${case_id//^//}.py"
  tag="${case_id##*^}"
  echo "==== [$(date +%Y-%m-%d_%H:%M:%S)] [$i/$total] RUN $tag 精度测试 ===="
  timeout 600 python single_case_run_runner.py "$rel_path" "${tag}_prec" "$YAML" > "$LOGS/${tag}_prec.log" 2>&1
  rc=$?
  prec=$(grep -o "'precision': '[a-z]*'" "$LOGS/${tag}_prec.log" | tail -1 | sed "s/.*: '//;s/'//")
  [ -z "$prec" ] && prec="NO_RESULT"
  # 从框架 PRECISION_DETAIL 行提取实际误差与 atol/rtol 容差明细
  detail=$(grep "PRECISION_DETAIL" "$LOGS/${tag}_prec.log" | tail -1 | sed 's/.*PRECISION_DETAIL res=[^ ]* PASS //')
  max_diff=$(echo "$detail" | grep -oE "max_abs_diff=[^ ]+" | cut -d= -f2)
  atol_v=$(echo "$detail" | grep -oE "atol=[^ ]+" | cut -d= -f2)
  rtol_v=$(echo "$detail" | grep -oE "rtol=[^ ]+" | cut -d= -f2)
  tol_v=$(echo "$detail" | grep -oE "tol_at_max_diff=[^ ]+" | cut -d= -f2)
  ratio_v=$(echo "$detail" | grep -oE "diff/tol=[^ ]+" | cut -d= -f2)
  echo "$case_id,$prec,$rc,${max_diff:-NA},${atol_v:-NA},${rtol_v:-NA},${tol_v:-NA},${ratio_v:-NA}" >> "$OUT"
  echo "==== [$(date +%Y-%m-%d_%H:%M:%S)] [$i/$total] $tag precision=$prec rc=$rc ${detail:+详情: $detail} ===="
done
echo "==== [$(date +%Y-%m-%d_%H:%M:%S)] ALL DONE: $OUT ===="
cat "$OUT"
ok=$(tail -n +2 "$OUT" | awk -F, '$2=="passed"' | wc -l)
echo "    通过 $ok / $total"
