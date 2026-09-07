#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1 运行期分桶命中探针：单进程只跑**一个** shape，配合 nsys GPU kernel trace
证明"不同张量形状实际启动了不同的 bucket kernel"。

用法：
    python bucket_dispatch_probe.py <case.py> <N,C,H,W>

为什么一个进程只跑一个 shape：nsys 的 trace 是进程级的，若同进程跑多个 shape，
报告里会同时出现多个 bucket kernel，无法把 kernel 归因到具体 shape。
拆成多进程后每份报告里只剩该 shape 命中的那一个桶，归因是无歧义的。

bucket kernel 的函数名自带谓词（_FPA_=( _BPA_=) MUL=* GE=>= LE=<=），
所以只看 kernel 名即可读出命中的是哪个分桶条件，不需要额外插桩。
"""
import sys
import importlib.util

import paddle

CASE = sys.argv[1]
SHAPE = [int(v) for v in sys.argv[2].split(",")]

spec = importlib.util.spec_from_file_location("case_mod", CASE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

paddle.set_device("gpu")
paddle.seed(123)
net = m.LayerCase()
net.eval()
input_spec = list(m.create_inputspec())
print("INPUTSPEC:", [s.shape for s in input_spec], flush=True)
paddle.base.core._set_prim_all_enabled(True)
net = paddle.jit.to_static(net, backend="CINN", full_graph=True, input_spec=input_spec)

paddle.seed(2024)
x = paddle.rand(shape=SHAPE, dtype="float32")
# 预热一次，确保编译与 cudnn 算法选择不落在计时窗口里；两次都会被 trace 到，
# 但同一 shape 命中的桶相同，不影响归因。
out = net(x)
paddle.device.synchronize()
out = net(x)
paddle.device.synchronize()
outs = paddle.utils.flatten(out)
print("RAN:", SHAPE, "-> out shapes:", [list(o.shape) for o in outs], flush=True)
print("DONE", flush=True)
