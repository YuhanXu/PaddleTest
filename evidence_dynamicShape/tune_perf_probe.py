#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4 性能对比：同一个 case、同一个运行 shape，对比两种 schedule 的 kernel 实测耗时。

    mode=dyn    input_spec 含 -1 → CINN 走分桶编译，运行期按谓词选桶
    mode=static input_spec 全静态 → CINN 按确切尺寸生成单一 kernel

用法（配 nsys 采 GPU trace）：

    nsys profile -t cuda -o rep_dyn    python tune_perf_probe.py <case.py> dyn
    nsys profile -t cuda -o rep_static python tune_perf_probe.py <case.py> static

前 WARMUP 次不计入（nsys 全采，统计时按 kernel 名聚合取均值即可，
warmup 与正式循环用的是同一个 kernel，故直接取全部 n 的均值）。
"""
import sys
import importlib.util

import paddle

CASE = sys.argv[1]
MODE = sys.argv[2]
SHAPE = [1, 72, 88, 88]
WARMUP = 5
ITERS = 50

spec = importlib.util.spec_from_file_location("case_mod", CASE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

paddle.set_device("gpu")
paddle.seed(123)
net = m.LayerCase()
net.eval()

if MODE == "dyn":
    input_spec = list(m.create_inputspec())
elif MODE == "static":
    input_spec = [paddle.static.InputSpec(shape=SHAPE, dtype="float32", name="x")]
else:
    raise SystemExit("mode must be dyn|static")

print("MODE:", MODE, "INPUTSPEC:", [s.shape for s in input_spec], flush=True)
paddle.base.core._set_prim_all_enabled(True)
net = paddle.jit.to_static(net, backend="CINN", full_graph=True, input_spec=input_spec)

paddle.seed(2024)
x = paddle.rand(shape=SHAPE, dtype="float32")

for _ in range(WARMUP + ITERS):
    out = net(x)
paddle.device.synchronize()
print("RAN:", SHAPE, "iters:", WARMUP + ITERS, flush=True)
print("DONE", flush=True)
