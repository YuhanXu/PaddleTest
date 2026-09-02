#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CINN 可变形状输入张量自动调优验证脚本
=====================================

技术指标 3：支持可变形状(dynamic shape)输入张量，支持根据不同张量形状自动调优。

验证思路（"同一模型对不同输入 shape 自动调优"）：
  1. 选取 layercase/sublayer1000 下的子图，其 create_inputspec() 含动态维(-1)。
  2. 固定同一份权重：eager 与 CINN 两个实例共享同一 state_dict。
  3. 用带 -1 的动态 InputSpec 对 CINN 实例只 to_static 编译"一次"。
  4. 依次喂入多组匹配动态维的具体 shape，逐一：
       - 与 eager(动态图)输出做数值一致性校验；
       - 记录该 shape 触发的 CINN codegen 事件数(codegen)。
    codegen 的计数方式：让 CINN 把每次生成的源码追加落盘
    (FLAGS_cinn_source_code_save_path)，统计文件中 `extern "C" {` 块的新增个数。
    SourceCodePrint 的 ofstream 在单例构造时以 trunc 打开一次、之后每次 write() 追加
    (paddle/cinn/backends/compiler.cc:220-253)，故块数单调递增，可按 shape 做差分。
    这是可比的计数：同一 case 内多 shape 只在首个 shape 产生 = group 数的事件；
    不同 case(通道数 C 不同即为不同子图)各自产生自己的事件。

    历史坑：曾用 glog 标记 "Compiling subgraph with CINN backend" 计数，该语句是
      LOG_FIRST_N(INFO, 1) << ...   (add_cinn_pass.cc:334)
    静态计数器使其整个进程最多打印一次，导致第 2 个 case 起恒为 0、
    看不到不同 C 的子图各自发生的编译。该计数已废弃，不再出现在报告里。

    判定 shape 是否被符号化处理，另可打印中间 IR：
      GLOG_vmodule='shape_o*=3' 观察 [ShapeDialect] 段动态维标注为 S0/S1/...；
      （必须用这种 <=15 字节的短 glob 写法：本 build 中长度 >= 16 字节的 GLOG_* 字符串型
        环境变量会让进程在退出析构期 double free 并挂死，详见 evidence_dynamicShape/
        ir_probe.py 的模块注释与 glog_multilink_evidence.log）
      FLAGS_cinn_source_code_save_path 导出的源码里动态维应为运行期 int32_t 形参；
      FLAGS_enable_cinn_compile_cache=false 关缓存后比较 codegen 事件增量。

用法：
    python test_dynamic_shape_cinn.py                 # 跑内置 10 个 case
    python test_dynamic_shape_cinn.py <case.py>       # 跑指定单个 layercase 文件
    python test_dynamic_shape_cinn.py --static-tuning # 额外做逐 shape 静态编译探针
    python test_dynamic_shape_cinn.py --static-only <case.py>  # 干净进程只跑静态探针
"""
import os
import sys
import time
import tempfile
import importlib.util

# 让 CINN 编译日志(INFO)即时刷新到 stderr。必须在 import paddle 之前设置。
os.environ.setdefault("GLOG_logbufsecs", "0")

# 可靠的编译次数计数：让 CINN 把每次 codegen 的源码追加落盘，统计 `extern "C" {` 块数。
# 必须在 import paddle 之前设置；允许外部预先 export 同名 FLAGS 以自定义落盘位置。
CODEGEN_SRC = os.environ.setdefault(
    "FLAGS_cinn_source_code_save_path",
    os.path.join(tempfile.mkdtemp(prefix="cinn_codegen_"), "cinn_source.cu"),
)

import numpy as np
import paddle

# ------------------------------------------------------------------ #
# 内置 10 个 case：单输入、float32、InputSpec=(-1, C, -1, -1)          #
# 均为 SE(Squeeze-Excitation)风格子图，对任意 H/W>=1 稳健，来自 8 个模型 #
# ------------------------------------------------------------------ #
SUBLAYER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "layercase", "sublayer1000"
)

CASES = [
    "Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py",   # C=72
    "Det_cases/picodet_legacy_model_picodet_m_320_coco/SIR_17.py",   # C=56
    "Det_cases/picodet_legacy_model_picodet_s_320_coco/SIR_17.py",   # C=44
    "Det_cases/ttfnet_pafnet_lite_mobilenet_v3_20x_coco/SIR_22.py",  # C=72
    "Det_cases/ssd_ssdlite_mobilenet_v3_large_320_coco/SIR_31.py",   # C=120
    "Det_cases/yolov3_yolov3_mobilenet_v3_large_ssld_270e_voc/SIR_64.py",  # C=480
    "Det_cases/centernet_centernet_mbv3_large_140e_coco/SIR_22.py",  # C=72
    "Det_cases/ppyolo_ppyolo_tiny_650e_coco/SIR_31.py",              # C=64
    "Det_cases/ppyolo_ppyolo_mbv3_small_coco/SIR_49.py",             # C=240
    "Det_cases/ppyolo_ppyolo_mbv3_large_coco/SIR_77.py",             # C=960
]

# (batch_factor, spatial_factor, tag)。含一次重复(S1==S6)用于验证缓存复用。
VARIANTS = [
    (1.0, 1.0, "S0-baseline"),
    (2.0, 1.0, "S1-batchx2"),
    (1.0, 0.5, "S2-spatial/2"),
    (1.0, 1.5, "S3-spatialx1.5"),
    (2.0, 0.5, "S4-batchx2-spatial/2"),
    (4.0, 1.0, "S5-batchx4"),
    (2.0, 1.0, "S6-batchx2(repeat->缓存)"),
]

ATOL = 1e-5
RTOL = 1e-5

# CINN 每完成一次 codegen 就往 CODEGEN_SRC 追加一个 `extern "C" { ... }` 块。
CODEGEN_BLOCK = 'extern "C" {'
_CODEGEN_SEEN = 0


def _codegen_total():
    """累计 codegen 事件数；文件尚未生成(计数不可用)时返回 -1。"""
    try:
        with open(CODEGEN_SRC, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().count(CODEGEN_BLOCK)
    except FileNotFoundError:
        return -1


def _codegen_delta():
    """自上次调用以来新增的 codegen 事件数；计数不可用时返回 -1。"""
    global _CODEGEN_SEEN
    cur = _codegen_total()
    if cur < 0:
        return -1
    delta = cur - _CODEGEN_SEEN
    _CODEGEN_SEEN = cur
    return delta


def _load_module(case_path):
    """按文件路径动态导入 layercase 模块（用唯一模块名避免缓存冲突）。"""
    mod_name = "layercase_" + os.path.relpath(case_path, SUBLAYER_DIR).replace(
        os.sep, "_"
    ).replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, case_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dynamic_dims(spec_shape):
    """返回 InputSpec 中值为 -1 的维度下标集合。"""
    return {i for i, d in enumerate(spec_shape) if d == -1}


def _make_shape(base_shape, dyn_dims, batch_f, spatial_f):
    """在动态维上按 batch/spatial 因子缩放，固定维保持不变。"""
    new = list(base_shape)
    for i in range(len(new)):
        if i not in dyn_dims:
            continue
        f = batch_f if i == 0 else spatial_f
        new[i] = max(1, int(round(base_shape[i] * f)))
    return new


def _fmt_shapes(shapes):
    """多输入时打印列表，单输入时直接打印该 shape。"""
    return str(shapes[0]) if len(shapes) == 1 else str(shapes)


def _build_nets(module):
    """构造共享同一份权重的 eager 网络与 CINN 编译网络。"""
    paddle.seed(123)
    net_eager = module.LayerCase()
    net_eager.eval()

    net_cinn = module.LayerCase()
    net_cinn.set_state_dict(net_eager.state_dict())  # 关键：同一份权重
    net_cinn.eval()

    input_spec = list(module.create_inputspec())
    paddle.base.core._set_prim_all_enabled(True)  # with_cinn 需要 with_prim
    net_cinn = paddle.jit.to_static(
        net_cinn, backend="CINN", full_graph=True, input_spec=input_spec
    )
    return net_eager, net_cinn, input_spec


def run_case(case_path):
    """对单个 layercase 执行多 shape 验证。

    每个 variant 返回 dict：
      status ∈ {PASS, FAIL:numeric, FAIL:cinn, SKIP:shape}
      - SKIP:shape  —— 构造输入或 eager 前向即失败，说明该 shape 对本子图非法，可跳过
      - FAIL:cinn   —— CINN 前向自身抛异常
      - FAIL:numeric—— CINN 与 eager 数值不一致（真 bug，必须判失败）
    并记录 cinn 前向耗时 dt_ms 与该 shape 触发的 CINN codegen 事件数 codegen。

    返回 (rows, spec_shapes)，spec_shapes 为动态 InputSpec 的 shape（含 -1），
    打进报告以体现"这一次编译的入口是动态的"。
    """
    module = _load_module(case_path)
    base_inputs = list(module.create_tensor_inputs())
    input_spec = list(module.create_inputspec())
    base_shapes = [list(t.shape) for t in base_inputs]
    dtypes = [t.dtype for t in base_inputs]
    dyn = [_dynamic_dims(s.shape) for s in input_spec]

    net_eager, net_cinn, _ = _build_nets(module)

    rows = []
    for bf, sf, tag in VARIANTS:
        shapes = [_make_shape(bs, d, bf, sf) for bs, d in zip(base_shapes, dyn)]
        row = {"tag": tag, "shapes": shapes, "dt_ms": -1.0, "codegen": -1}

        # 阶段 1：构造输入 + eager 前向。此处失败 => 该 shape 非法 => SKIP（不算错误）
        try:
            paddle.seed(2024)
            inputs = [paddle.rand(shape=sp, dtype=dt) for sp, dt in zip(shapes, dtypes)]
            eager_out = net_eager(*inputs)
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"SKIP:shape({type(exc).__name__})"
            rows.append(row)
            continue

        # 阶段 2：CINN 前向。首个 shape 会触发符号化编译。抛异常 => FAIL:cinn
        try:
            t0 = time.time()
            cinn_out = net_cinn(*inputs)
            paddle.device.synchronize()
            row["dt_ms"] = (time.time() - t0) * 1000.0
            row["codegen"] = _codegen_delta()
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"FAIL:cinn({type(exc).__name__})"
            rows.append(row)
            continue

        # 阶段 3：数值一致性。不一致 => FAIL:numeric（绝不降级为 SKIP）
        try:
            for e, c in zip(
                paddle.utils.flatten(eager_out), paddle.utils.flatten(cinn_out)
            ):
                np.testing.assert_allclose(e.numpy(), c.numpy(), atol=ATOL, rtol=RTOL)
            row["status"] = "PASS"
        except AssertionError:
            row["status"] = "FAIL:numeric"
        rows.append(row)
    return rows, [list(s.shape) for s in input_spec]



def probe_static_tuning(case_path, factors=((1.0, 1.0), (1.0, 0.5), (2.0, 1.0))):
    """按 shape 自动调优的直接证据（静态 spec 路径）。

    对每个不同的具体 shape，用【静态 InputSpec（无 -1）】各自 to_static(CINN) 编译一次，
    统计每次的 codegen 事件数并做数值校验。预期：每个不同 shape 都 codegen>=1，
    即"CINN 根据不同张量形状分别触发编译/自动调度(auto-schedule)调优"。
    """
    module = _load_module(case_path)
    base_inputs = list(module.create_tensor_inputs())
    input_spec = list(module.create_inputspec())
    base_shapes = [list(t.shape) for t in base_inputs]
    dtypes = [t.dtype for t in base_inputs]
    stops = [getattr(s, "stop_gradient", True) for s in input_spec]
    dyn = [_dynamic_dims(s.shape) for s in input_spec]

    paddle.seed(123)
    net_eager = module.LayerCase()
    net_eager.eval()

    rows = []
    for bf, sf in factors:
        shapes = [_make_shape(bs, d, bf, sf) for bs, d in zip(base_shapes, dyn)]
        row = {"shapes": shapes, "codegen": -1, "status": "?"}
        try:
            paddle.seed(123)
            net = module.LayerCase()
            net.set_state_dict(net_eager.state_dict())
            net.eval()
            static_spec = [
                paddle.static.InputSpec(shape=sp, dtype=dt, stop_gradient=sg)
                for sp, dt, sg in zip(shapes, dtypes, stops)
            ]
            paddle.base.core._set_prim_all_enabled(True)
            net = paddle.jit.to_static(
                net, backend="CINN", full_graph=True, input_spec=static_spec
            )
            paddle.seed(2024)
            inputs = [paddle.rand(shape=sp, dtype=dt) for sp, dt in zip(shapes, dtypes)]
            eager_out = net_eager(*inputs)
            cinn_out = net(*inputs)
            paddle.device.synchronize()
            row["codegen"] = _codegen_delta()
            for e, c in zip(
                paddle.utils.flatten(eager_out), paddle.utils.flatten(cinn_out)
            ):
                np.testing.assert_allclose(e.numpy(), c.numpy(), atol=ATOL, rtol=RTOL)
            row["status"] = "PASS"
        except AssertionError:
            row["status"] = "FAIL:numeric"
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"FAIL:{type(exc).__name__}"
        rows.append(row)
    return rows


def _print_static_report(case_rel, rows):
    print("\n" + "-" * 84)
    print(f"[静态spec/按shape编译] {case_rel}")
    print(f"{'shape':<26}{'codegen':>9}{'status':>16}")
    for r in rows:
        s = _fmt_shapes(r["shapes"])
        c = str(r["codegen"]) if r["codegen"] >= 0 else "-"
        print(f"{s:<26}{c:>9}{r['status']:>16}")
    npass = sum(1 for r in rows if r["status"] == "PASS")
    vals = [r["codegen"] for r in rows if r["status"] == "PASS"]
    if vals and all(v >= 0 for v in vals):
        each_compiled = all(v >= 1 for v in vals)
        verdict = "是" if each_compiled else "否"
    else:
        each_compiled = None  # codegen 计数不可用，不参与判定
        verdict = "计数不可用"
    print(f"每个不同 shape 均独立触发编译: {verdict}  (PASS={npass}/{len(rows)})")
    return npass, len(rows), each_compiled


def _print_case_report(case_rel, rows, spec_shapes=None):
    """打印单 case 报告，返回 (n_pass, n_fail, n_skip, n_total)。"""
    print("\n" + "=" * 84)
    print(f"CASE: {case_rel}")
    if spec_shapes:
        print(f"inspec (动态编译入口): {_fmt_shapes(spec_shapes)}")
    print("-" * 84)
    print(f"{'variant':<26}{'shape':<26}{'status':<16}{'codegen':>9}{'cinn_ms':>9}")
    for r in rows:
        shape_s = _fmt_shapes(r["shapes"])
        ms = f"{r['dt_ms']:7.2f}" if r["dt_ms"] >= 0 else "   -   "
        comp = str(r["codegen"]) if r["codegen"] >= 0 else "-"
        print(f"{r['tag']:<26}{shape_s:<26}{r['status']:<16}{comp:>9}{ms:>9}")

    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_fail = sum(1 for r in rows if r["status"].startswith("FAIL"))
    n_skip = sum(1 for r in rows if r["status"].startswith("SKIP"))
    print("-" * 84)
    print(f"PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}  (共 {len(rows)})")

    # 符号化编译的直接证据：codegen 事件按 shape 的分布
    events = [
        (r["tag"], _fmt_shapes(r["shapes"]), r["codegen"])
        for r in rows
        if r["status"] == "PASS" and r["codegen"] >= 0
    ]
    if events:
        triggered = [f"{t} {s}" for t, s, c in events if c > 0]
        reused = [f"{t} {s}" for t, s, c in events if c == 0]
        total = sum(c for _, _, c in events)
        # total 是本 case 新增的 codegen 次数。未命中编译缓存时等于该子图的 fusion group 数；
        # 若同拓扑同通道数的子图已被本进程内别的 case 编译过，则部分/全部 group 命中缓存而计少。
        note = "未命中缓存的 group 数" if total > 0 else "全部命中编译缓存"
        print(f"本 case 新增 codegen 事件: {total} ({note})")
        print(f"触发 codegen 的 shape:    {triggered}")
        print("未触发(复用已编译)的 shape:")
        for item in reused:
            print(f"    {item}")
    else:
        print(f"codegen 计数不可用(读不到 {CODEGEN_SRC})")
    return n_pass, n_fail, n_skip, len(rows)


def main(argv):
    """用法:
      python test_dynamic_shape_cinn.py [--static-tuning] [<case.py>]

    默认(动态 spec)：同一模型单次符号化编译 + 多 shape 正确性与编译次数观测。
    --static-tuning：额外跑"每个不同 shape 各自静态编译"探针，检查是否按 shape 独立编译。
    """
    if not paddle.is_compiled_with_cinn():
        print("[SKIP] 当前 Paddle 未编译 CINN (is_compiled_with_cinn()=False)，无法实跑验证。")
        return 2
    if not paddle.device.is_compiled_with_cuda():
        print("[SKIP] 当前 Paddle 未编译 CUDA，CINN 需要 GPU。")
        return 2
    paddle.set_device("gpu")
    print(f"[codegen 计数] CINN 生成源码落盘于: {CODEGEN_SRC}")

    static_mode = "--static-tuning" in argv
    static_only = "--static-only" in argv
    positional = [a for a in argv[1:] if not a.startswith("--")]
    if positional:
        case_list = [os.path.abspath(positional[0])]
    else:
        case_list = [os.path.join(SUBLAYER_DIR, c) for c in CASES]

    # --static-only：不跑动态编译，仅做"每个 shape 各自静态编译"，用于在干净进程里
    # 观察 CINN 是否按 shape 分别编译（避免被同进程的结构级编译缓存掩盖）。
    if static_only:
        verdicts = []
        for path in case_list:
            rel = os.path.relpath(path, SUBLAYER_DIR)
            srows = probe_static_tuning(path)
            _sp, _st, each_ok = _print_static_report(rel, srows)
            verdicts.append(each_ok)
        print("\n[static-only] 结论: ", end="")
        if any(v is None for v in verdicts):
            print("codegen 计数不可用，无法判定")
        elif all(verdicts):
            print("每个不同 shape 均独立触发编译 -> 存在按 shape 的编译/调优")
        else:
            print("并非每个 shape 都触发编译 -> CINN 对同一子图按结构缓存，多 shape 复用同一次编译")
        return 0

    case_summ = []  # (rel, n_pass, n_fail, n_skip, n_total, static_ok)
    for path in case_list:
        rel = os.path.relpath(path, SUBLAYER_DIR)
        try:
            rows, spec_shapes = run_case(path)
            p, f, s, t = _print_case_report(rel, rows, spec_shapes)
        except Exception as exc:  # noqa: BLE001  case 级构建失败 => 视为该 case 失败
            print(f"\n[CASE ERROR] {rel}: {type(exc).__name__}: {exc}")
            p, f, s, t = 0, len(VARIANTS), 0, len(VARIANTS)

        static_ok = None
        if static_mode:
            try:
                srows = probe_static_tuning(path)
                sp, st, static_ok = _print_static_report(rel, srows)
                # 静态模式下若有数值 FAIL 也并入判定
                if sp != st:
                    f += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[STATIC ERROR] {rel}: {type(exc).__name__}: {exc}")
                static_ok = False
                f += 1
        case_summ.append((rel, p, f, s, t, static_ok))

    print("\n" + "#" * 84)
    print("汇总：同一模型 / 多 shape / CINN 动态形状 + 自动调优")
    print("#" * 84)
    # 验收标准(§6)：每个 case 的全部 variant 必须 PASS（无 FAIL、无 SKIP）才算 OK；
    # 若开启 --static-tuning，还要求每个不同 shape 都独立触发编译(按 shape 调优)。
    all_ok = True
    for rel, p, f, s, t, static_ok in case_summ:
        ok = (f == 0 and s == 0 and p == t) and (static_ok in (None, True))
        all_ok = all_ok and ok
        flag = "OK  " if ok else ("FAIL" if f else "PART")
        extra = "" if static_ok is None else f" static_tuning={'是' if static_ok else '否'}"
        print(f"[{flag}] PASS={p} FAIL={f} SKIP={s} /{t}{extra}  {rel}")
    print("-" * 84)
    tot_pass = sum(x[1] for x in case_summ)
    tot = sum(x[4] for x in case_summ)
    print(f"总计 shape 通过: {tot_pass}/{tot}，覆盖 {len(case_summ)} 个模型子图")
    total_codegen = _codegen_total()
    if total_codegen >= 0:
        print(
            f"全进程 codegen 事件合计: {total_codegen}"
            f"（每个 case 各自编译，与 shape 个数无关；源码见 {CODEGEN_SRC}）"
        )
    print("最终判定:", "全部通过 (exit 0)" if all_ok else "存在 FAIL/SKIP/未按shape调优 (exit 1)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))




