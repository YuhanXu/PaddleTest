#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2 负对照：与 ir_probe.py 同一个 case、同一条 CINN 编译路径，唯一差别是
input_spec 用**全静态** shape（不带 -1）。

用途：证明「多桶 + COND__ 谓词后缀 + int32_t Sx 形参」是**动态 shape 特有**的产物，
而不是 CINN 对任何图都会生成的东西。

    FLAGS_cinn_source_code_save_path=/tmp/e2/src.cu \
    python static_shape_probe.py <case.py>

预期：产物里 __global__ 数 == extern "C" 数（每 group 只 1 个 kernel）、
无 COND__ 后缀、kernel 形参里没有 S0/S1/S2。
"""
import sys
import importlib.util

import paddle

CASE = sys.argv[1]
SHAPE = [1, 72, 88, 88]

spec = importlib.util.spec_from_file_location("case_mod", CASE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

paddle.set_device("gpu")
paddle.seed(123)
net = m.LayerCase()
net.eval()

# 与 ir_probe.py 的唯一差别：这里不调用 m.create_inputspec()（其 shape 含 -1），
# 而是给一个具体尺寸的 InputSpec。
input_spec = [paddle.static.InputSpec(shape=SHAPE, dtype="float32", name="x")]
print("INPUTSPEC:", [s.shape for s in input_spec], flush=True)
paddle.base.core._set_prim_all_enabled(True)
net = paddle.jit.to_static(net, backend="CINN", full_graph=True, input_spec=input_spec)

paddle.seed(2024)
x = paddle.rand(shape=SHAPE, dtype="float32")
out = net(x)
paddle.device.synchronize()
outs = paddle.utils.flatten(out)
print("RAN:", SHAPE, "-> out shapes:", [list(o.shape) for o in outs], flush=True)
print("DONE", flush=True)
