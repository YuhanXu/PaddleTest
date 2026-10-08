#!/usr/bin/env python3
# 扫描 all_perf_results_100.csv 的 100 个 case, 判定哪些是动态 shape (create_inputspec 含 -1)
# 方法与 PaddleTest_dynamicShape T2/ir_probe.py 一致: 加载 case 模块, 读 create_inputspec()
import os
import sys
import csv
import importlib.util

import paddle

CSV_IN = os.environ.get("PLT_CSV_IN", "/work/all_perf_results_100.csv")
CSV_OUT = os.environ.get("PLT_CSV_OUT", "/work/all_perf_results_100_dynamic.csv")

rows = list(csv.reader(open(CSV_IN)))[1:]
out_rows = []
n_dyn = 0
for i, row in enumerate(rows):
    title = row[0]
    rel = title.replace("^", "/") + ".py"
    tag = "dynamic" if "?" in title else "static"
    try:
        spec = importlib.util.spec_from_file_location(f"case_mod_{i}", rel)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        shapes = [list(s.shape) for s in m.create_inputspec()]
        dyn = any(-1 in sh for sh in shapes)
        # 记录动态维个数
        n_dyn_dim = sum(1 for sh in shapes for d in sh if d == -1)
        info = ";".join(str(sh) for sh in shapes)
        if dyn:
            n_dyn += 1
    except Exception as e:
        dyn = None
        n_dyn_dim = -1
        info = f"ERR: {str(e)[:80]}"
    out_rows.append([title, "dynamic" if dyn else ("static" if dyn is False else "unknown"), n_dyn_dim, info] + row[1:])

with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "shape_mode", "n_dyn_dims", "inputspec", "dy_eval_perf", "dy2st_eval_cinn_perf", "speedup", "status"])
    w.writerows(out_rows)

dyn_rows = [r for r in out_rows if r[1] == "dynamic"]
static_rows = [r for r in out_rows if r[1] == "static"]
err_rows = [r for r in out_rows if r[1] == "unknown"]
print(f"总计 {len(out_rows)}: 动态 {len(dyn_rows)}, 静态 {len(static_rows)}, 加载异常 {len(err_rows)}")

# 动态 case 的性能分布
if dyn_rows:
    sp = sorted(float(r[6]) for r in dyn_rows if r[7] == "OK")
    if sp:
        import statistics
        gt1 = sum(1 for x in sp if x > 1)
        print(f"动态 case speedup>1(cinn更快): {gt1}/{len(sp)}, 中位数 {statistics.median(sp):.2f}, 算术平均 {statistics.mean(sp):.2f}")
if static_rows:
    sp2 = sorted(float(r[6]) for r in static_rows if r[7] == "OK")
    if sp2:
        import statistics
        gt1 = sum(1 for x in sp2 if x > 1)
        print(f"静态 case speedup>1(cinn更快): {gt1}/{len(sp2)}, 中位数 {statistics.median(sp2):.2f}, 算术平均 {statistics.mean(sp2):.2f}")

print("动态 case 清例:")
for r in dyn_rows:
    print(f"  {r[0].split(chr(94))[2]:<12} {r[0].split(chr(94))[3]:<48} dyn_dims={r[2]:<2} sp={r[6]}")
