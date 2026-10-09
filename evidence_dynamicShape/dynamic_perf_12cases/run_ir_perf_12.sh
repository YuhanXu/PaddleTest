#!/bin/bash
# run_ir_perf_12.sh -- 对 12 个动态编译 case 跑 ir_perf_probe.py 三方性能对比
# 三列: dy / cinn动态input_spec编译 / cinn静态tracing编译(无input_spec, 与批跑同口径)
# 产物: /work/all_perf_results_100_dy_shape_12.csv + ir_perf_logs/{tag}_irperf.log
# 用法: bash run_ir_perf_12.sh   (脚本内部自行 cd 到 PaddleLT_new)
set -u

BASE=/work/PaddleTest/framework/e2e/PaddleLT_new
HERE=$(cd "$(dirname "$0")" && pwd)
LOGDIR=$HERE/ir_perf_logs
CSV=/work/all_perf_results_100_dy_shape_12.csv
PROBE=$HERE/ir_perf_probe.py
mkdir -p "$LOGDIR"

export MIN_GRAPH_SIZE=0 FLAGS_cinn_debug=1 FLAGS_prim_forward_blacklist=pd_op.dropout
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$BASE
cd "$BASE"

# tag|case相对路径 —— 12 个动态编译 case（kernel 签名法判定, 与 dynamic12_rerun/run.sh 同源）
CASES=(
  "segformer_b0_1024x1024_SIR_14|layercase/sublayer1000/Seg_cases/segformer_segformer_b0_cityscapes_1024x1024_160k/SIR_14.py"
  "segformer_b0_1024x512_SIR_26|layercase/sublayer1000/Seg_cases/segformer_segformer_b0_cityscapes_1024x512_160k/SIR_26.py"
  "segformer_b0_1024x512_SIR_14|layercase/sublayer1000/Seg_cases/segformer_segformer_b0_cityscapes_1024x512_160k/SIR_14.py"
  "segformer_b3_1024x512_SIR_18|layercase/sublayer1000/Seg_cases/segformer_segformer_b3_cityscapes_1024x512_160k/SIR_18.py"
  "segformer_b0_1024x1024_SIR_26|layercase/sublayer1000/Seg_cases/segformer_segformer_b0_cityscapes_1024x1024_160k/SIR_26.py"
  "segformer_b3_1024x1024_SIR_38|layercase/sublayer1000/Seg_cases/segformer_segformer_b3_cityscapes_1024x1024_160k/SIR_38.py"
  "segformer_b3_1024x512_SIR_38|layercase/sublayer1000/Seg_cases/segformer_segformer_b3_cityscapes_1024x512_160k/SIR_38.py"
  "segformer_b3_1024x1024_SIR_5|layercase/sublayer1000/Seg_cases/segformer_segformer_b3_cityscapes_1024x1024_160k/SIR_5.py"
  "ppyoloe_voc_l_SIR_182|layercase/sublayer1000/Det_cases/ppyoloe_voc_ppyoloe_plus_crn_l_30e_voc/SIR_182.py"
  "segformer_b0_1024x1024_SIR_5|layercase/sublayer1000/Seg_cases/segformer_segformer_b0_cityscapes_1024x1024_160k/SIR_5.py"
  "segformer_b3_1024x1024_SIR_18|layercase/sublayer1000/Seg_cases/segformer_segformer_b3_cityscapes_1024x1024_160k/SIR_18.py"
  "smalldet_sod_SIR_174|layercase/sublayer1000/Det_cases/smalldet_ppyoloe_plus_sod_crn_l_80e_coco/SIR_174.py"
)

echo "case,dy_eval_perf,dy2st_eval_cinn_dynspec_perf,dy2st_eval_cinn_nospec_perf,speedup_dyn,speedup_static,dyn_vs_static" > "$CSV"

for entry in "${CASES[@]}"; do
  tag="${entry%%|*}"
  path="${entry#*|}"
  echo "==== [$(date +%H:%M:%S)] $tag ====" >> "$LOGDIR/driver.log"
  timeout 900 python "$PROBE" "$path" > "$LOGDIR/${tag}_irperf.log" 2>&1
  rc=$?
  log="$LOGDIR/${tag}_irperf.log"
  dy=$(grep "^RESULT dy_eval_perf" "$log" 2>/dev/null | awk '{print $3}')
  dyn=$(grep "^RESULT dy2st_eval_cinn_dynspec_perf" "$log" 2>/dev/null | awk '{print $3}')
  st=$(grep "^RESULT dy2st_eval_cinn_nospec_perf" "$log" 2>/dev/null | awk '{print $3}')
  if [ -n "$dy" ] && [ -n "$dyn" ] && [ -n "$st" ]; then
    sp_dyn=$(awk -v a="$dy" -v b="$dyn" 'BEGIN{printf "%.3f", a/b}')
    sp_st=$(awk -v a="$dy" -v b="$st" 'BEGIN{printf "%.3f", a/b}')
    dvss=$(awk -v a="$st" -v b="$dyn" 'BEGIN{printf "%.3f", a/b}')
    echo "$tag,$dy,$dyn,$st,$sp_dyn,$sp_st,$dvss" >> "$CSV"
  else
    echo "$tag,ERROR_rc${rc},,," >> "$CSV"
  fi
  echo "$tag rc=$rc done [$(date +%H:%M:%S)]" >> "$LOGDIR/driver.log"
done
echo "ALL DONE csv=$CSV rows=$(wc -l < "$CSV")" >> "$LOGDIR/driver.log"
