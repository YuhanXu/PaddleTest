"""定位 S1/S5 那几十毫秒花在哪个 op、以及是否伴随显存增长。

逐 op 之间插 synchronize 分别计时，同时打印 allocator 的 reserved/allocated。
"""
import importlib.util
import sys
import time

import paddle
import paddle.nn.functional as F


def load(p):
    s = importlib.util.spec_from_file_location("m", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def t(fn, *a, **kw):
    paddle.device.synchronize()
    t0 = time.time()
    out = fn(*a, **kw)
    paddle.device.synchronize()
    return (time.time() - t0) * 1000.0, out


paddle.set_device("gpu")
m = load(sys.argv[1])
paddle.seed(123)
net = m.LayerCase()
net.eval()
base = list(m.create_tensor_inputs())[0].shape
C = base[1]

VARIANTS = [
    (1.0, 1.0, "S0"), (2.0, 1.0, "S1-bx2"), (1.0, 0.5, "S2-sp/2"),
    (1.0, 1.5, "S3-spx1.5"), (2.0, 0.5, "S4-bx2sp/2"), (4.0, 1.0, "S5-bx4"),
    (2.0, 1.0, "S6-bx2rep"),
]

MB = 1024.0 * 1024.0
hdr = f"{'variant':<12}{'shape':<20}{'pool':>8}{'conv1':>8}{'relu':>7}" \
      f"{'conv2':>8}{'hsig':>7}{'mul':>8}{'resvMB':>9}{'+resv':>8}"
print(hdr)
prev_resv = paddle.device.cuda.memory_reserved() / MB
for bf, sf, tag in VARIANTS:
    shape = [max(1, int(round(base[0] * bf))), C,
             max(1, int(round(base[2] * sf))), max(1, int(round(base[3] * sf)))]
    paddle.seed(2024)
    x = paddle.rand(shape=shape, dtype="float32")

    ms_pool, v1 = t(F.adaptive_avg_pool2d, x, 1)
    ms_c1, v2 = t(F.conv2d, v1, net.parameter_2, net.parameter_3)
    ms_relu, v3 = t(F.relu, v2)
    ms_c2, v4 = t(F.conv2d, v3, net.parameter_0, net.parameter_1)
    ms_hs, v5 = t(F.hardsigmoid, v4, slope=0.2, offset=0.5)
    ms_mul, _ = t(paddle.multiply, x, v5)

    resv = paddle.device.cuda.memory_reserved() / MB
    print(f"{tag:<12}{str(shape):<20}{ms_pool:>8.2f}{ms_c1:>8.2f}{ms_relu:>7.2f}"
          f"{ms_c2:>8.2f}{ms_hs:>7.2f}{ms_mul:>8.2f}{resv:>9.1f}{resv - prev_resv:>8.1f}")
    prev_resv = resv
