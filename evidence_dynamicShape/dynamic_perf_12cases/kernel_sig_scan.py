#!/usr/bin/env python3
# kernel 签名法批量判定: 对 all_perf_results_100 的 100 个 case, 逐个独立子进程跑无 input_spec 的
# CINN 前向(复现批跑链路 paddle_eval_bm.py:154), 落盘 cinn_source.cu, 统计符号形参/谓词分桶,
# 判定每个 case 的实际编译模式(dynamic/static), 与 inputspec 声明交叉比对
import csv
import os
import re
import subprocess
import sys

BASE = "/work/PaddleTest/framework/e2e/PaddleLT_new"
OUT = "/work/kernel_sig_probe"  # 判定原始 cu 落盘目录(运行期产物)
LOGS = os.path.join(OUT, "logs")
os.makedirs(LOGS, exist_ok=True)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_IN = os.path.join(HERE, "all_perf_results_100_dynamic.csv")  # 已含 inputspec 声明列
CSV_OUT = os.path.join(HERE, "all_perf_results_100_kernelmode.csv")

env_base = dict(os.environ)
env_base.update({
    "LD_LIBRARY_PATH": "/usr/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
    "PYTHONPATH": "/usr/local/lib/python3.10/dist-packages:" + BASE,
    "MIN_GRAPH_SIZE": "0",
    "FLAGS_cinn_debug": "1",
    "FLAGS_prim_forward_blacklist": "pd_op.dropout",
})

rows = list(csv.reader(open(CSV_IN)))[1:]
results = []
for i, row in enumerate(rows):
    title = row[0]
    rel_path = title.replace("^", "/") + ".py"
    cu_path = os.path.join(OUT, f"case_{i:03d}.cu")
    log_path = os.path.join(LOGS, f"case_{i:03d}.log")

    env = dict(env_base)
    env["FLAGS_cinn_source_code_save_path"] = cu_path
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "kernel_sig_one.py"), rel_path],
            cwd=BASE, env=env, capture_output=True, timeout=300, text=True, errors="ignore",
        )
        with open(log_path, "w") as f:
            f.write(proc.stdout + "\n---stderr---\n" + proc.stderr)
        ok = "FWD_OK" in proc.stdout
    except subprocess.TimeoutExpired:
        with open(log_path, "w") as f:
            f.write("TIMEOUT 300s\n")
        ok = False

    # 统计落盘 cu
    sym_params, cond_f, cond_true, n_group, n_kernel = -1, -1, -1, -1, -1
    if os.path.exists(cu_path):
        txt = open(cu_path, errors="ignore").read()
        sym_params = len(re.findall(r"\bint(?:32|64)_t S\d+\b", txt))
        cond_f = txt.count("COND__F")
        cond_true = txt.count("COND_true")
        n_group = txt.count('extern "C" {')
        n_kernel = txt.count("__global__")

    mode = "fwd-error" if not ok else ("dynamic-compiled" if (sym_params > 0 or cond_f > 0) else "static-compiled")
    results.append(row + [mode, sym_params, cond_f, cond_true, n_group, n_kernel])
    print(f"[{i+1}/{len(rows)}] {mode:<17} sym={sym_params:<3} condF={cond_f:<3} condT={cond_true:<3} {title.split(chr(94))[3][:40]}", flush=True)

with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "spec_declared", "n_dyn_dims", "inputspec",
                "dy_eval_perf", "dy2st_eval_cinn_perf", "speedup", "status",
                "kernel_compile_mode", "n_symbol_params", "n_cond_f_buckets", "n_cond_true", "n_groups", "n_kernels"])
    w.writerows(results)

# 汇总
from collections import Counter
modes = Counter(r[8] for r in results)
print("\n编译模式分布:", dict(modes))
# 交叉表: inputspec 声明 vs 实际编译模式
cross = Counter((r[1], r[8]) for r in results)
print("声明(x) 实际编译(y) 交叉表:")
for (decl, mode), c in sorted(cross.items()):
    print(f"  {decl:<10} x {mode:<17} : {c}")
# 动态编译 case 的性能
dyn_c = [r for r in results if r[8] == "dynamic-compiled"]
if dyn_c:
    sp = [float(r[6]) for r in dyn_c if r[7] == "OK"]
    if sp:
        import statistics
        print(f"\n动态编译 case: {len(dyn_c)} 个, speedup>1: {sum(1 for x in sp if x>1)}/{len(sp)}, 中位数 {statistics.median(sp):.2f}")
print(f"CSV: {CSV_OUT}")
