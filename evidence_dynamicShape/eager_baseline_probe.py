"""为什么 S1(batch=2) / S5(batch=4) 首次前向要几十毫秒？—— eager 基线对照

判别做法：拿掉 CINN，纯 eager（不 to_static）跑同样的 7 组 shape。
若 eager 也在首次 batch=2/4 上慢，则该开销与 CINN、与符号化编译均无关。

配套：perop_probe.py（定位到哪个 op + 显存是否增长）、
      conv_shape_cost_probe.py（干净进程里单独证明是 cuDNN 按 conv shape 选算法）。
"""
import importlib.util
import sys
import time

import paddle


def load(p):
    s = importlib.util.spec_from_file_location("m", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def timed(fn, *a):
    t0 = time.time()
    out = fn(*a)
    paddle.device.synchronize()
    return (time.time() - t0) * 1000.0, out


paddle.set_device("gpu")
case = sys.argv[1]
m = load(case)

paddle.seed(123)
net = m.LayerCase()
net.eval()
base = list(m.create_tensor_inputs())[0].shape
C = base[1]

VARIANTS = [
    (1.0, 1.0, "S0-baseline"),
    (2.0, 1.0, "S1-batchx2"),
    (1.0, 0.5, "S2-spatial/2"),
    (1.0, 1.5, "S3-spatialx1.5"),
    (2.0, 0.5, "S4-batchx2-spatial/2"),
    (4.0, 1.0, "S5-batchx4"),
    (2.0, 1.0, "S6-batchx2(repeat)"),
]

print(f"=== 纯 eager（无 CINN、无 to_static），{case.split('/')[-2]} 基准 {base} ===")
print(f"{'variant':<24}{'shape':<22}{'eager_ms':>10}")
for bf, sf, tag in VARIANTS:
    shape = [
        max(1, int(round(base[0] * bf))),
        C,
        max(1, int(round(base[2] * sf))),
        max(1, int(round(base[3] * sf))),
    ]
    paddle.seed(2024)
    x = paddle.rand(shape=shape, dtype="float32")
    ms, _ = timed(net, x)
    print(f"{tag:<24}{str(shape):<22}{ms:>10.2f}")
