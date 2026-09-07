#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三通道数独立对照探针：同一进程内先后编译三个拓扑相同、仅通道数 C 不同
的 picodet SE 子图（C=72/56/44），正面验证"不同 C 不复用编译"。

结论判据（对 src.cu 计数）：
    grep -c 'extern "C" {' <src.cu>   == 6   # 3 case × 2 group，全部真编译
    grep -c '__global__'   <src.cu>   == 24  # 每 group 4 个分桶 kernel

它与 7.4 实验 2 构成一对对照：实验 2 证明"拓扑相同且 C 相同 ⇒ FusionInfo
缓存命中、codegen 不增"；本探针给出另一侧——C 不同 ⇒ 缓存必不命中，
每个 case 各自完整编译（三个 case 的首次前向各 ~5s，与完整编译耗时一致）。

用法（src.cu 路径须在 import paddle 之前由环境变量传入）：
    export FLAGS_cinn_source_code_save_path=$PWD/three_c_src.cu
    python three_c_probe.py
"""
import importlib.util
import os
import time

import paddle

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(
    HERE, os.pardir, "framework", "e2e", "PaddleLT_new",
    "layercase", "sublayer1000", "Det_cases"))
CASES = [
    "picodet_legacy_model_picodet_l_640_coco/SIR_17.py",  # C=72
    "picodet_legacy_model_picodet_m_320_coco/SIR_17.py",  # C=56
    "picodet_legacy_model_picodet_s_320_coco/SIR_17.py",  # C=44
]

paddle.set_device("gpu")
for rel in CASES:
    path = os.path.join(BASE, rel)
    spec = importlib.util.spec_from_file_location("case_mod", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    paddle.seed(123)
    net = m.LayerCase()
    net.eval()
    input_spec = list(m.create_inputspec())
    paddle.base.core._set_prim_all_enabled(True)
    net = paddle.jit.to_static(
        net, backend="CINN", full_graph=True, input_spec=input_spec
    )

    shape = list(m.create_tensor_inputs()[0].shape)
    paddle.seed(2024)
    x = paddle.rand(shape=shape, dtype="float32")
    t0 = time.perf_counter()
    out = net(x)
    paddle.device.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"CASE {rel} shape={shape}   {dt:.2f} ms", flush=True)
print("DONE", flush=True)
