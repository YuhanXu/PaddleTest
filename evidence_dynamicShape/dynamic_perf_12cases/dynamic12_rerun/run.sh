#!/bin/bash
# 重跑 12 个动态编译 case（kernel 签名法判定: 落盘 cu 含符号形参/COND__F 谓词桶）
# 每 case 跑两遍: dy_eval_perf 与 dy2st_eval_cinn_perf 各一个独立子进程(与批跑同链路, 无 input_spec);
# cinn 侧通过 FLAGS_cinn_source_code_save_path 落盘该 case 实际生成的 CINN kernel 源码
# 产物: logs/{tag}_{dy|cinn}.log, kernels/{tag}.cu, results.csv
set -u

BASE=/work/PaddleTest/framework/e2e/PaddleLT_new
OUT=$(cd "$(dirname "$0")" && pwd)
LOGS=$OUT/logs
KERNELS=$OUT/kernel
mkdir -p "$LOGS" "$KERNELS"

export MIN_GRAPH_SIZE=0 FLAGS_cinn_debug=1 FLAGS_prim_forward_blacklist=pd_op.dropout
export PLT_DEVICE_ID=0 PLT_GET_NV_MEMORY=False PLT_BM_DB=non-db
export CASE_TYPE=layercase CASE_DIR=sublayer1000
export TESTING_MODE=performance MULTI_WORKER=0
export FRAMEWORK=paddle PLT_SET_DEVICE=gpu
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$BASE
cd "$BASE"

# tag|case 相对路径 —— 12 个动态编译 case（来自 /work/all_perf_results_100_kernelmode.csv）
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

echo "case,dy_eval_perf,dy2st_eval_cinn_perf,speedup,n_symbol_params,n_cond_f_buckets,n_cond_true,n_groups,n_kernels" > "$OUT/results.csv"

for entry in "${CASES[@]}"; do
  tag="${entry%%|*}"
  path="${entry#*|}"
  echo "==== [$(date +%H:%M:%S)] $tag dy ====" >> "$OUT/driver.log"

  # 1) dy 性能
  timeout 600 python single_engine_runner.py "$path" "${tag}_dy" \
    yaml/dy_eval_perf_only_benchmark.yml > "$LOGS/${tag}_dy.log" 2>&1
  dy_rc=$?

  # 2) cinn 性能 + kernel 落盘
  FLAGS_cinn_source_code_save_path="$KERNELS/${tag}.cu" \
  timeout 900 python single_engine_runner.py "$path" "${tag}_cinn" \
    yaml/dy2st_eval_cinn_perf_only_benchmark.yml > "$LOGS/${tag}_cinn.log" 2>&1
  cinn_rc=$?

  python3 - "$OUT" "$tag" "$dy_rc" "$cinn_rc" << 'PYEOF'
import csv, os, re, sys
out, tag, dy_rc, cinn_rc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def get_perf(path, key):
    try:
        txt = open(path, errors="ignore").read()
        m = re.search(rf"'{key}': ([\d.]+)", txt)
        return float(m.group(1)) if m else None
    except Exception:
        return None
dy = get_perf(f"{out}/logs/{tag}_dy.log", "dy_eval_perf")
cinn = get_perf(f"{out}/logs/{tag}_cinn.log", "dy2st_eval_cinn_perf")
sp = f"{dy/cinn:.3f}" if dy and cinn else ""
cu = f"{out}/kernel/{tag}.cu"
sym = cond_f = cond_t = grp = krn = -1
if os.path.exists(cu):
    txt = open(cu, errors="ignore").read()
    sym = len(re.findall(r"\bint(?:32|64)_t S\d+\b", txt))
    cond_f = txt.count("COND__F")
    cond_t = txt.count("COND_true")
    grp = txt.count('extern "C" {')
    krn = txt.count("__global__")
row = [tag, dy or "", cinn or "", sp, sym, cond_f, cond_t, grp, krn]
with open(f"{out}/results.csv", "a", newline="") as f:
    csv.writer(f).writerow(row)
print(f"  -> {tag}: dy={dy} cinn={cinn} sp={sp} sym={sym} condF={cond_f}")
PYEOF
done
echo "ALL DONE $(date)" >> "$OUT/driver.log"
