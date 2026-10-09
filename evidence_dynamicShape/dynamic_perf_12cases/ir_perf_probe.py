#!/usr/bin/env python3
# ir_perf_probe.py -- 动态 input_spec 的 CINN 性能探针
# 用法: python ir_perf_probe.py <case相对路径>   (须在 PaddleLT_new 目录下运行)
# 输出: dy / cinn动态spec编译 / cinn静态tracing 三列稳态性能(每100 iter 平均秒数, 与批跑 csv 同量纲)
# 编译入口对齐 evidence_dynamicShape/ir_probe.py:
#   动态列 = to_static(backend="CINN", full_graph=True, input_spec=create_inputspec())
#   静态列 = to_static(backend="CINN", full_graph=True)  (与批跑 paddle_eval_bm.py:154 相同)
# 计时口径对齐 engine/paddle_eval_bm.py: warmup 10 + 10 轮 x 100 iter, 轮内含 cuda sync, 取轮均值
# fallback: case 无 create_inputspec()/返回空时, 用 paddle_base_layer_test 的
#   create_inputspec(inputs, stop_gradient=None) 把所有输入 shape 全标 -1, 保证 dyn 列可编译
import sys
import time
import timeit
import importlib.util

import paddle

CASE = sys.argv[1]
N_WARMUP = 10
N_ROUND = 10
N_ITER = 100  # 每轮迭代数, 输出量纲 = 每轮平均秒数(与 all_perf_results csv 一致)


def load_case(path):
    spec = importlib.util.spec_from_file_location("case_mod", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_dyn_spec(m, inputs):
    """case 自带 create_inputspec 且非空则用之; 否则按实际输入就地构造全 -1 动态 spec"""
    if hasattr(m, "create_inputspec"):
        sp = list(m.create_inputspec())
        if sp:
            return sp
    spec = []
    for t in inputs:
        shape = [-1] * len(t.shape) if len(t.shape) > 0 else []
        spec.append(paddle.static.InputSpec(shape=shape, dtype=t.dtype))
    return spec


def bench(fn):
    timeit.timeit(fn, number=N_WARMUP)
    times = []
    for _ in range(N_ROUND):
        t0 = time.time()
        for _ in range(N_ITER):
            fn()
        paddle.core._cuda_synchronize(paddle.CUDAPlace(0))
        times.append(time.time() - t0)
    times.sort()
    return sum(times) / len(times)


def main():
    m = load_case(CASE)
    paddle.set_device("gpu")
    paddle.seed(123)

    inputs = list(m.create_tensor_inputs())
    input_spec = make_dyn_spec(m, inputs)
    print(f"INPUTSPEC: {[tuple(s.shape) for s in input_spec]}", flush=True)

    # 1) eager dy 基线
    net = m.LayerCase()
    net.eval()
    dy_t = bench(lambda: net(*inputs))
    print(f"RESULT dy_eval_perf {dy_t:.6f}", flush=True)

    # 2) CINN 动态 input_spec 编译
    with paddle.decomposition.decomp.prim_guard():
        dyn_net = paddle.jit.to_static(m.LayerCase(), backend="CINN", full_graph=True, input_spec=input_spec)
        dyn_net.eval()
        dyn_t = bench(lambda: dyn_net(*inputs))
    print(f"RESULT dy2st_eval_cinn_dynspec_perf {dyn_t:.6f}", flush=True)

    # 3) CINN 静态 tracing 编译 (批跑同口径)
    with paddle.decomposition.decomp.prim_guard():
        st_net = paddle.jit.to_static(m.LayerCase(), backend="CINN", full_graph=True)
        st_net.eval()
        st_t = bench(lambda: st_net(*inputs))
    print(f"RESULT dy2st_eval_cinn_nospec_perf {st_t:.6f}", flush=True)

    print(f"SUMMARY dy={dy_t:.6f} cinn_dyn={dyn_t:.6f} cinn_static={st_t:.6f} "
          f"sp_dyn={dy_t / dyn_t:.3f} sp_static={dy_t / st_t:.3f} dyn_vs_static={st_t / dyn_t:.3f}", flush=True)


if __name__ == "__main__":
    main()
