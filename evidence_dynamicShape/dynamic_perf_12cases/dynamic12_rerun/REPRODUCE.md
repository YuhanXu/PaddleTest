# 动态编译 12 case 重跑复现说明

日期: 2026-10-08 | 机器: yq01-inf-hic-k8s-a100-aa24-0582 (A100)

## 背景

`/work/PaddleTest/evidence_dynamicShape/dynamic_perf_12cases/all_perf_results_100_kernelmode.csv` 用 kernel 签名法（落盘 cu 中
`int(32|64)_t S\d+` 符号形参与 `COND__F` 谓词分桶）判定出 100 个 case 里
12 个为 **dynamic-compiled**（内生动态：批跑链路 to_static 未传 input_spec，
由网络内部数据依赖算子自动符号化）。本目录重跑这 12 个 case 的
dy_eval_perf vs dy2st_eval_cinn_perf 性能，并保留 CINN 生成的 kernel 源码。

## 12 个 case 清单

| # | tag | case | 符号形参 | 谓词桶 |
|---|---|---|---|---|
| 1 | segformer_b0_1024x1024_SIR_14 | Seg/segformer_b0_cityscapes_1024x1024_160k/SIR_14 | 20 | 16 |
| 2 | segformer_b0_1024x512_SIR_26 | Seg/segformer_b0_cityscapes_1024x512_160k/SIR_26 | 20 | 16 |
| 3 | segformer_b0_1024x512_SIR_14 | Seg/segformer_b0_cityscapes_1024x512_160k/SIR_14 | 20 | 16 |
| 4 | segformer_b3_1024x512_SIR_18 | Seg/segformer_b3_cityscapes_1024x512_160k/SIR_18 | 20 | 16 |
| 5 | segformer_b0_1024x1024_SIR_26 | Seg/segformer_b0_cityscapes_1024x1024_160k/SIR_26 | 20 | 16 |
| 6 | segformer_b3_1024x1024_SIR_38 | Seg/segformer_b3_cityscapes_1024x1024_160k/SIR_38 | 20 | 16 |
| 7 | segformer_b3_1024x512_SIR_38 | Seg/segformer_b3_cityscapes_1024x512_160k/SIR_38 | 20 | 16 |
| 8 | segformer_b3_1024x1024_SIR_5 | Seg/segformer_b3_cityscapes_1024x1024_160k/SIR_5 | 20 | 16 |
| 9 | ppyoloe_voc_l_SIR_182 | Det/ppyoloe_voc_ppyoloe_plus_crn_l_30e_voc/SIR_182 | 2 | 0 |
| 10 | segformer_b0_1024x1024_SIR_5 | Seg/segformer_b0_cityscapes_1024x1024_160k/SIR_5 | 20 | 16 |
| 11 | segformer_b3_1024x1024_SIR_18 | Seg/segformer_b3_cityscapes_1024x1024_160k/SIR_18 | 20 | 16 |
| 12 | smalldet_sod_SIR_174 | Det/smalldet_ppyoloe_plus_sod_crn_l_80e_coco/SIR_174 | 3 | 0 |

## 一键复现

```bash
bash /work/PaddleTest/evidence_dynamicShape/dynamic_perf_12cases/dynamic12_rerun/run.sh
```

## 手工单跑某个 case（dy / cinn 各一条）

```bash
cd /work/PaddleTest/framework/e2e/PaddleLT_new
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH \
       MIN_GRAPH_SIZE=0 FLAGS_cinn_debug=1 FLAGS_prim_forward_blacklist=pd_op.dropout \
       PLT_DEVICE_ID=0 PLT_GET_NV_MEMORY=False PLT_BM_DB=non-db \
       CASE_TYPE=layercase CASE_DIR=sublayer1000 \
       TESTING_MODE=performance MULTI_WORKER=0 FRAMEWORK=paddle PLT_SET_DEVICE=gpu \
       PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PWD

# dy 性能
python single_engine_runner.py \
  layercase/sublayer1000/Seg_cases/segformer_segformer_b0_cityscapes_1024x512_160k/SIR_26.py \
  segformer_b0_1024x512_SIR_26_dy \
  yaml/dy_eval_perf_only_benchmark.yml

# cinn 性能 + kernel 源码落盘（关键: FLAGS_cinn_source_code_save_path）
FLAGS_cinn_source_code_save_path=/work/PaddleTest/evidence_dynamicShape/dynamic_perf_12cases/dynamic12_rerun/kernel/segformer_b0_1024x512_SIR_26.cu \
python single_engine_runner.py \
  layercase/sublayer1000/Seg_cases/segformer_segformer_b0_cityscapes_1024x512_160k/SIR_26.py \
  segformer_b0_1024x512_SIR_26_cinn \
  yaml/dy2st_eval_cinn_perf_only_benchmark.yml
```

注: 该链路 to_static 未传 input_spec（与批跑 `paddle_eval_bm.py:154` 一致），
动态符号化由网络内部数据依赖算子（segformer 的 gather/scatter 链）内生触发。

## kernel 签名判定方法（复用）

```bash
# 符号形参（动态编译标志）
grep -cE '\bint(32|64)_t S[0-9]' <case>.cu
# 谓词分桶数（按形状区间分桶的 schedule）
grep -c 'COND__F' <case>.cu
# 静态退化谓词
grep -c 'COND_true' <case>.cu
```

## 产物结构

```
/work/PaddleTest/evidence_dynamicShape/dynamic_perf_12cases/dynamic12_rerun/
├── run.sh                 # 一键复现脚本（12 case 循环: dy + cinn + cu 落盘）
├── REPRODUCE.md           # 本文件
├── results.csv            # 12 case 性能与 kernel 统计汇总
├── driver.log             # 运行时间线
├── scan.log               # 汇总打印
├── logs/{tag}_dy.log      # 每 case dy 运行日志
├── logs/{tag}_cinn.log    # 每 case cinn 运行日志（含性能结果行）
└── kernel/{tag}.cu        # 每 case CINN 生成的 CUDA kernel 源码（含符号形参/谓词分桶）
```

相关上游产物：
- `/work/PaddleTest/evidence_dynamicShape/dynamic_perf_12cases/all_perf_results_100_kernelmode.csv`（100 case 编译模式判定）
- `/work/kernel_sig_probe/`（判定的原始 cu 与日志，运行期产物不入库）
- `../kernel_sig_scan.py`、`../kernel_sig_one.py`、`../probe_dynamic_cases.py`（判定/扫描脚本，已随本目录归档）
- `/work/PaddleTest/framework/e2e/PaddleLT_new/single_engine_runner.py`（单执行器入口）
- `/work/PaddleTest/framework/e2e/PaddleLT_new/yaml/dy_eval_perf_only_benchmark.yml` / `dy2st_eval_cinn_perf_only_benchmark.yml`

## 本次结果（2026-10-08）

- 11/12 cinn 胜出，speedup 范围 1.13~1.61，中位数 ~1.32
- smalldet_sod/SIR_174 本次 0.82x（上批 1.69x）——临界 case 波动翻转，与
  100 case 稳定性分析中"speedup 接近 1 的边缘 case 会跨线"的结论一致
- segformer 系 10 case 全部 sym=20 / condF=16 / 9 group / 20 kernel
  （谓词分桶编译稳定复现）
