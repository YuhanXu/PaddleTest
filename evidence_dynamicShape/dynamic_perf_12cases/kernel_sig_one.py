#!/usr/bin/env python3
# 单 case runner: 无 input_spec 的 CINN to_static 一次前向(复现批跑链路), cu 落盘由环境变量控制
import sys
import importlib.util

import paddle

case_path = sys.argv[1]

paddle.set_device("gpu")
spec = importlib.util.spec_from_file_location("case_mod", case_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

net = m.LayerCase()
net.eval()
inputs = list(m.create_tensor_inputs()) if hasattr(m, "create_tensor_inputs") else []

net = paddle.jit.to_static(net, backend="CINN", full_graph=True)
out = net(*inputs)
paddle.device.synchronize()
print("FWD_OK", case_path)
