#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""边界 shape 数值精度探针：T8.1 的 [1,72,1,1] / [64,72,88,88] 只验证了分桶派发
（bucket_dispatch_probe.py，无 eager 基准、无数值断言）。本探针补 eager vs CINN
数值对比，口径与主测试 test_dynamic_shape_cinn.py 一致（atol=rtol=1e-5）。

用法：python boundary_shape_acc_probe.py <case.py> <N,C,H,W>
"""
import sys
import importlib.util

import numpy as np
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

paddle.seed(2024)
x = paddle.rand(shape=SHAPE, dtype="float32")

# eager 基准（不走 CINN）
out_eager = net(x)
paddle.device.synchronize()

# CINN：动态 inputspec（与主测试同入口 -1,72,-1,-1）
input_spec = list(m.create_inputspec())
paddle.base.core._set_prim_all_enabled(True)
cinn_net = paddle.jit.to_static(
    net, backend="CINN", full_graph=True, input_spec=input_spec
)
out_cinn = cinn_net(x)
paddle.device.synchronize()

e = paddle.utils.flatten(out_eager)[0].numpy()
c = paddle.utils.flatten(out_cinn)[0].numpy()
ok = np.allclose(c, e, atol=1e-5, rtol=1e-5)
diff = np.abs(c - e)
print(
    f"SHAPE={SHAPE} eager vs CINN: allclose(atol=rtol=1e-5)={ok} "
    f"max_abs_diff={diff.max():.3e} mean_abs_diff={diff.mean():.3e}",
    flush=True,
)
print("PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
