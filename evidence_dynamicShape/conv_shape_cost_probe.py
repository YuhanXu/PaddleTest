"""干净进程里单独验证：cuDNN conv 的开销是否按 N 缓存。

前一版探针把裸 conv 放在整网循环之后跑，N=1/2/4 早已被前面的 conv 走过一遍、
cuDNN 缓存已命中，所以测不出东西（属无效对照）。这里在**全新进程**里只跑裸 conv。
"""
import time

import paddle

paddle.set_device("gpu")
C = 960
conv = paddle.nn.Conv2D(C, C // 4, 1)


def t(x):
    paddle.device.synchronize()
    t0 = time.time()
    conv(x)
    paddle.device.synchronize()
    return (time.time() - t0) * 1000.0


print(f"裸 conv2d([N,{C},1,1] -> {C // 4})，干净进程")
print(f"{'N':<5}{'ms':>10}   说明")
for n, note in [(1, "首次 N=1（含 cuDNN 首次初始化）"), (2, "首次 N=2"),
                (1, "N=1 重复"), (4, "首次 N=4"), (2, "N=2 重复"),
                (8, "首次 N=8"), (4, "N=4 重复")]:
    x = paddle.rand(shape=[n, C, 1, 1], dtype="float32")
    print(f"{n:<5}{t(x):>10.2f}   {note}")
