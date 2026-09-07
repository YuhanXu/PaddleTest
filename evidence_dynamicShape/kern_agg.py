#!/usr/bin/env python3
"""聚合 nsys cuda_gpu_trace csv：按 kernel 名聚合 n/mean/min/max + grid/block。"""
import csv, sys, re
from collections import defaultdict

rows = []
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        name = r.get("Name", "")
        if "COND__" not in name:
            continue
        rows.append((name, int(r["Duration (ns)"]), r["GrdX"], r["GrdY"], r["GrdZ"],
                     r["BlkX"], r["BlkY"], r["BlkZ"]))

agg = defaultdict(list)
for name, dur, gx, gy, gz, bx, by, bz in rows:
    # G1/G2 靠函数名区分：G1 = gs_bc_add...max（18 元素）；G2 = mul_gs_bc_add（557568 元素）
    g = "G1" if ("gs_bc_max" in name and "mul" not in name) else "G2"
    agg[g].append((dur, gx, gy, gz, bx, by, bz))

for g in ("G1", "G2"):
    if g not in agg:
        continue
    durs = [d for d, *_ in agg[g]]
    dur, gx, gy, gz, bx, by, bz = agg[g][0]
    print(f"{g} grid={gx}x{gy}x{gz} block={bx}x{by}x{bz} n={len(durs)} "
          f"mean={sum(durs)//len(durs)}ns min={min(durs)} max={max(durs)}")
