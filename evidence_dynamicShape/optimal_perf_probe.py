#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 搜索式调优读侧探针：FLAGS_tile_config_policy=optimal 时，
从自定义 tile_config 数据库读取 bucket→tileConfig，替代 default 规则分桶。

    用法（三个环境变量须在 import paddle 前设好，DB_DIR 作为 argv[2] 传入）：
    export FLAGS_tile_config_policy=optimal
    nsys profile -t cuda -o rep  python optimal_perf_probe.py <case.py> <db_dir>

关键点：paddle/__init__.py 在 import 时把 CINN_CONFIG_PATH 指向内置库
paddle/cinn_config（仅含 A100-40GB / V100 等机型的预搜索 JSON，本机
A100-80GB 无条目）。因此自定义数据库必须在 **import 之后** 用 os.environ
覆盖才生效（os.environ 赋值会调 putenv，后续 C++ 侧 getenv 可见）。

数据库目录布局（与内置库一致，由 file_database.cc 的 IterSpaceTypeToDir 推导）：
    <db_dir>/tile_config/NVGPU_NVIDIA_A100_SXM4_80GB/S_EREBE/Sdynamic.json
每行一个 JSON 条目（proto3 JSON，int64 字段为字符串，与内置库格式一致）：
    {"bucketInfo":{"dimension":[{"lowerBound":1,"upperBound":2147483647,
     "iterType":"S","isDynamic":true}]},"tileConfig":{"warpNum":"8",
     "treeReduceNum":"1","spatialInnerNum":"1"}}

跑 [1,72,88,88] 固定 shape、动态 input_spec、5 warmup + 50 iters，
与 tune_perf_probe.py 的 dyn 模式完全同构，便于逐 kernel 对比。
"""
import os
import sys
import importlib.util

import paddle

CASE = sys.argv[1]
DB = sys.argv[2]
SHAPE = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 72, 88, 88]
WARMUP = 5
ITERS = 50

os.environ["CINN_CONFIG_PATH"] = DB

spec = importlib.util.spec_from_file_location("case_mod", CASE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

paddle.set_device("gpu")
paddle.seed(123)
net = m.LayerCase()
net.eval()
input_spec = list(m.create_inputspec())

print("DB:", DB, "INPUTSPEC:", [s.shape for s in input_spec], flush=True)
paddle.base.core._set_prim_all_enabled(True)
net = paddle.jit.to_static(net, backend="CINN", full_graph=True, input_spec=input_spec)

paddle.seed(2024)
x = paddle.rand(shape=SHAPE, dtype="float32")

for _ in range(WARMUP + ITERS):
    out = net(x)
paddle.device.synchronize()
print("RAN:", SHAPE, "iters:", WARMUP + ITERS, flush=True)
print("DONE", flush=True)
