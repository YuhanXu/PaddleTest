#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小 CINN 动态 shape 驱动：只做一次编译 + 若干 shape 前向，
不做 fd 重定向，让 glog/VLOG 输出直接落到 stdout/stderr 重定向的文件里。

采 IR 时请用**短 glob 形式**的 vmodule：

    GLOG_logtostderr=1 GLOG_vmodule='shape_o*=3' python ir_probe.py <case.py>

不要写 GLOG_vmodule=shape_optimization_pass=3。原因（2026-09-01 已定位到根因）：

任何**长度 >= 16 字节**的 GLOG_* 字符串型环境变量（vmodule / log_dir /
log_backtrace_at）都会让进程在**打完 DONE 之后的退出期**触发 glibc 堆校验并挂死
（报错文本漂移：corrupted double-linked list / double free or corruption (!prev) /
malloc_consolidate(): unaligned fastbin chunk detected）。这是 Paddle 的打包缺陷，
与 CINN、符号化 shape pass、IR dump 全都无关（`import paddle` 一行即可复现）：

  libglog.a 被**静态链进 4 个 .so**（libpaddle.so / libphi_core.so / libphi_gpu.so /
  libcinnapi.so），且 glog 的全局符号都是导出的。ELF 符号插入使 4 份
  `fLS::FLAGS_vmodule_buf` 数据坍缩成 1 个实例（实测只有 libpaddle.so 那份被写入，
  另 3 份 .bss 全 0），但 4 个 `_GLOBAL__sub_I_vlog_is_on.cc` 静态初始化器**各跑一次**，
  于是同一个 std::string 被构造 4 次、并注册 4 次 __cxa_atexit 析构。
  值 <= 15 字节走 SSO（无堆缓冲）⇒ 重复构造析构无害；>= 16 字节每次构造都 malloc，
  前 3 个泄漏、最后一个被 free 4 次 ⇒ 退出期 double free。
  实测分界线正好在 15/16 字节；GLOG_v 是 int32 不涉及堆，故 GLOG_v=3 不复现。

挂死（而非直接 abort 退出）是次级原因：Paddle 装的 glog failure signal handler
（init.cc:437 InstallFailureSignalHandler + SignalHandle）在 SIGABRT 里做
ostringstream / backtrace 符号化，都要 malloc，而 abort 正是从持锁的 malloc 校验路径
发出来的 —— 自死锁。用 /proc/<pid>/status 可证实：SigBlk=0x20（SIGABRT 被屏蔽 ⇒
正处在自己的 SIGABRT handler 里）、SigCgt 含 4/6/7/8/11/15。

glog 的 vmodule 支持 * 与 ? 通配（SafeFNMatch_），所以把 25 字节的
`shape_optimization_pass=3` 缩成 10 字节的 `shape_o*=3` 即可绕开，
匹配到的文件与产物完全一致（实测除时间戳外逐字节相同）、EXIT=0。
"""
import os
import sys
import importlib.util

import paddle

CASE = sys.argv[1]
SHAPES = [[1, 72, 88, 88], [2, 72, 88, 88], [1, 72, 44, 44]]

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

for sh in SHAPES:
    paddle.seed(2024)
    x = paddle.rand(shape=sh, dtype="float32")
    out = net(x)
    paddle.device.synchronize()
    outs = paddle.utils.flatten(out)
    print("RAN:", sh, "-> out shapes:", [list(o.shape) for o in outs], flush=True)
print("DONE", flush=True)
