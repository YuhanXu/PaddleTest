#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编译次数探针：关闭编译缓存后，逐 shape 计时并统计 codegen 事件数。

SourceCodePrint 的 ofstream 在单例构造时以 trunc 打开一次，之后每次 write() 追加，
因此 cinn_source.cu 中 `extern "C" {` 块的个数 == 进程内 codegen 事件数。
若多个 shape 只产生 = group 数的块数，即证明新 shape 未触发重新 codegen。
"""
import importlib.util
import sys
import time

import paddle

CASE = sys.argv[1]
SHAPES = [
    [1, 72, 88, 88],    # S0 基准
    [2, 72, 88, 88],    # S1 batch x2
    [1, 72, 44, 44],    # S2 spatial /2
    [1, 72, 132, 132],  # S3 spatial x1.5
    [2, 72, 44, 44],    # S4
    [4, 72, 88, 88],    # S5 batch x4
    [2, 72, 88, 88],    # S6 重复 S1
]

spec = importlib.util.spec_from_file_location("case_mod", CASE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

paddle.set_device("gpu")
paddle.seed(123)
net = m.LayerCase()
net.eval()
paddle.base.core._set_prim_all_enabled(True)
net = paddle.jit.to_static(
    net, backend="CINN", full_graph=True, input_spec=list(m.create_inputspec())
)

for i, sh in enumerate(SHAPES):
    paddle.seed(2024)
    x = paddle.rand(shape=sh, dtype="float32")
    t0 = time.time()
    net(x)
    paddle.device.synchronize()
    print(f"SHAPE[{i}] {sh}  {(time.time() - t0) * 1000:9.2f} ms", flush=True)
print("DONE", flush=True)
