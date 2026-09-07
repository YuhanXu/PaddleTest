# 9.3 技术指标 3：支持可变形状输入张量，支持根据不同张量形状自动调优

> 配套脚本：`framework/e2e/PaddleLT_new/test_dynamic_shape_cinn.py`
> 配套测试计划：`PaddleTest_dynamicShape_schedule.md`（§8 用例清单、§9 实跑结果、§10 机制说明）

## 9.3.3 测试步骤

### 步骤 1：环境准备与前置自检

本测试需要一份已编译 CINN 且 CUDA 可用的 Paddle。在 A100(SM80) 开发机上按下列环境变量激活
（关键是把 `/usr/lib64` 加入 `LD_LIBRARY_PATH`，否则 `import paddle` 会因找不到 `libcuda.so.1` 而失败）：

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
export PYTHONPATH=/work/Paddle/build/python
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source /work/env3.10/bin/activate
```

自检命令与预期输出：

```bash
python -c "import paddle; print(paddle.__version__, \
paddle.is_compiled_with_cinn(), paddle.device.is_compiled_with_cuda())"
# 3.5.0.dev20260526 True True
```

脚本入口 `main()` 也内置了同样的前置检查，二者任一为 False 即以 `exit 2` 标记 SKIP，
不会给出假通过结论。

> **图 9.3.3-1**：环境自检与 `nvidia-smi` 输出截图
> （需体现 driver 535.230.02、GPU=A100、`is_compiled_with_cinn()=True`）。

### 步骤 2：确认被测对象具备动态维声明

被测对象取 `layercase/sublayer1000` 下 10 个 SE(Squeeze-Excitation) 风格子图，来自 8 个检测模型，
均为单输入 float32，且 `create_inputspec()` 的 batch 与 H/W 维为 `-1`、通道维 C 为固定值。
清单与基准 shape 见测试计划 §8 表格。抽验其中一个：

```bash
python -c "
import importlib.util, os
p='layercase/sublayer1000/Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py'
s=importlib.util.spec_from_file_location('m', p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print([t.shape for t in m.create_tensor_inputs()])   # [[1, 72, 88, 88]]
print([sp.shape for sp in m.create_inputspec()])     # [[-1, 72, -1, -1]]
"
```

`-1` 的存在是本测试成立的前提：它使得同一次编译入口可以接受不同的具体 shape。

### 步骤 3：构造共享同一份权重的 eager 与 CINN 两个实例

为排除"权重不同导致输出不同"这一干扰，两个实例必须共享 `state_dict`。
这一步对应脚本 `_build_nets()`（第 130-145 行）：

```python
paddle.seed(123)
net_eager = module.LayerCase()
net_eager.eval()

net_cinn = module.LayerCase()
net_cinn.set_state_dict(net_eager.state_dict())   # 关键：同一份权重
net_cinn.eval()

input_spec = list(module.create_inputspec())      # 含 -1 的动态 spec
paddle.base.core._set_prim_all_enabled(True)      # with_cinn 需要 with_prim
net_cinn = paddle.jit.to_static(
    net_cinn, backend="CINN", full_graph=True, input_spec=input_spec
)
```

注意 `to_static` 在整个 case 生命周期内**只调用一次**，之后不再重新编译，
这样"多 shape 复用同一编译入口"才是被测行为本身。

### 步骤 4：遍历 shape 变体矩阵

在动态维 N/H/W 上按 (batch 因子, spatial 因子) 缩放，固定维 C 不变，共 9 组
（脚本 `VARIANTS`；S0~S6 为原始 7 组，S7/S8 为 2026-09-07 追加的跨桶边界变体）：

| 组别 | 因子 (batch, spatial) | 以 picodet_l_640 为例 | 覆盖点 |
|------|----------------------|----------------------|--------|
| S0 | (1, 1) | [1, 72, 88, 88] | 基准，与源文件一致 |
| S1 | (2, 1) | [2, 72, 88, 88] | batch 维变化 |
| S2 | (1, 0.5) | [1, 72, 44, 44] | 空间维缩小 |
| S3 | (1, 1.5) | [1, 72, 132, 132] | 空间维放大 |
| S4 | (2, 0.5) | [2, 72, 44, 44] | 多维同时变化 |
| S5 | (4, 1) | [4, 72, 88, 88] | batch 进一步放大 |
| S6 | (2, 1) | [2, 72, 88, 88] | 对照组，重复 S1 |
| S7 | (1, 0.01) | [1, 72, 1, 1] | 空间维归一，翻入 LE1023 小桶 |
| S8 | (64, 1) | [64, 72, 88, 88] | batch×64，翻入 GE1024 大桶 |

缩放后向下取整并保底为 1（`max(1, int(round(...)))`），避免小分辨率子图被缩成 0 维。

### 步骤 5：逐 shape 三阶段判定

每个 shape 按三个阶段独立记录状态，三类失败原因严格区分、不互相掩盖
（脚本 `run_case()`，第 172-204 行）：

- 阶段 1 —— 构造输入并跑 eager 前向。此处异常说明该 shape 对本子图非法，记 `SKIP:shape`。
- 阶段 2 —— 跑 CINN 前向并 `paddle.device.synchronize()` 后计时。异常记 `FAIL:cinn`。
- 阶段 3 —— 与 eager 输出逐张量比对，不一致记 `FAIL:numeric`，一致记 `PASS`。

```python
for e, c in zip(paddle.utils.flatten(eager_out), paddle.utils.flatten(cinn_out)):
    np.testing.assert_allclose(e.numpy(), c.numpy(), atol=1e-5, rtol=1e-5)
```

`atol=rtol=1e-5` 相对源用例的 1e-8 有所放宽，原因是子图含 conv，CINN 融合后累加次序与 cuDNN 不同，
属可接受的浮点差异；该量级仍足以捕获真实计算错误。

### 步骤 6：逐 shape 统计 CINN codegen 事件数

脚本需要回答"某个新 shape 是否触发了重新编译/重新调优"。这里采用 **codegen 事件计数**：

```python
os.environ.setdefault("GLOG_logbufsecs", "0")     # 必须在 import paddle 之前
CODEGEN_SRC = os.environ.setdefault(              # 同样必须在 import paddle 之前
    "FLAGS_cinn_source_code_save_path",
    os.path.join(tempfile.mkdtemp(prefix="cinn_codegen_"), "cinn_source.cu"),
)
CODEGEN_BLOCK = 'extern "C" {'
```

`SourceCodePrint` 的 ofstream 在单例构造时以 trunc 打开一次、之后每次 `write()` 追加
（`paddle/cinn/backends/compiler.cc:220-253`），因此该文件中 `extern "C" {` 块的个数
**单调递增**且等于进程内已发生的 codegen 次数。脚本在每个 shape 前向后读一次块数并做差分，
填入报告的 `codegen` 列；进程结束时打印总数。

预期形态：同一 case 内只有首个 shape 的 `codegen` 为正（等于该子图的 fusion group 数），
其余 shape 恒为 0 ⇒ 一次符号化编译服务全部 shape。单个 case 的合计**允许为 0 或小于
group 数**——这是同拓扑同通道数的子图命中 FusionInfo 编译缓存（本轮 ttfnet=1、centernet=0），
不是失败。定量对照见 7.4。

> **已废弃的计数法**：早期版本用 fd 级重定向捕获 glog、统计
> `"Compiling subgraph with CINN backend"` 出现次数作为 `compiles` 列。该语句是
> `paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.cc:334` 的
> `LOG_FIRST_N(INFO, 1)`，静态计数器使其**整个进程最多打印一次**，导致第 2 个 case 起恒为 0，
> 与真实编译次数无关。该列及配套的 stderr 重定向已从脚本移除；历史日志中的 `compiles=0`
> 不代表未编译。

### 步骤 7：打印中间 IR，直接证明 shape 在 CINN 中是符号化的

这是本指标最直接的证据链：CINN 内部对 shape 的表示本身就是**符号（symbolic）**的，
只要把中间 IR 打出来，就能看到动态维以符号变量参与运算，而不是被固化成常量。
无需依赖日志计数或耗时推断。

本节所有命令与产物均已在 A100 上**实跑验证**，产物归档于
`/work/PaddleTest/evidence_dynamicShape/`。采集 IR 时使用同目录下的最小驱动
`ir_probe.py`（一次动态 spec 编译 + 依次跑 `[1,72,88,88]`/`[2,72,88,88]`/`[1,72,44,44]`），
以免主脚本的多 case 循环干扰 glog 输出的对应关系。

**7.1 打印带符号维标注的 PIR（已实测）**

PIR 的动态维由 `InferSymbolicShapeContext::GetNextSymName()` 生成，命名为 `S0`、`S1`、`S2`…
（见 `paddle/pir/src/dialect/shape/utils/shape_analysis.cc:113-114`）。
`ShapeOptimizationPass` 会用 `shape_analysis.PrintHook()` 把每个 Value 的
`ShapeOrDataDimExprs` 一并打印，打印开关是 `VLOG` 级别 3
（`paddle/pir/src/dialect/shape/transforms/shape_optimization_pass.cc:31,54-64`）。

不要用全局 `GLOG_v=3`（日志量巨大且拖慢一个数量级），改用 `GLOG_vmodule` 只放开该文件；
写法必须用通配短写法（原因见下方"已知缺陷"）：

```bash
export GLOG_logtostderr=1
export GLOG_vmodule='shape_o*=3'    # shape_optimization_pass=3 的通配短写法（10 字节）
python ir_probe.py <case.py> > ir_symbolic.log 2>&1

grep -n "\[ShapeDialect\]" ir_symbolic.log
# 5:===================== [ShapeDialect]Origin Program =====================
# 129:===================== [ShapeDialect]ShapeOptimizationPass Program =====================
```

> **已知缺陷（与本指标无关，但影响本步执行体验）**：本 build 中长度 >= 16 字节的
> `GLOG_*` **字符串型**环境变量（`vmodule` / `log_dir` / `log_backtrace_at`）会让进程
> 在打完全部输出、进入退出析构阶段时触发 glibc 堆校验并挂死（报错文本漂移：
> `corrupted double-linked list` / `double free or corruption (!prev)` 等，需 Ctrl+C）。
> 根因是 `libglog.a` 被静态链进 4 个 .so（`base/libpaddle.so`、`libs/libphi_core.so`、
> `libs/libphi_gpu.so`、`libs/libcinnapi.so`）且 glog 全局符号导出：ELF 符号插入把
> 4 份 `fLS::FLAGS_vmodule_buf` 坍缩成 1 个实例，但 4 个静态初始化器各构造一次、
> 各注册一次 `__cxa_atexit` ⇒ 退出期同一堆指针被 free 4 次；<= 15 字节走 SSO
> （无堆缓冲）故无害，分界线实测正好在 15/16 字节。`import paddle` 一行加长
> vmodule 即可复现（无 GPU、无 tensor、无 to_static），与 CINN/符号化/IR dump 无关。
> **规避即上面的短 glob 写法**（glog vmodule 支持 `*`/`?`），exit=0、产物与长写法
> 逐字节相同（除时间戳）。完整取证链（SSO 分界线矩阵、ctypes 读 4 份 buf、gdb 4 次
> 初始化 hit、ASAN double-free）见测试步骤文档 T4 末尾及 `glog_multilink_evidence.log`。
> 不要用 `os._exit(0)` 跳过析构——那会把 abort 压成 `EXIT=0`，损害验收产物完整性。

实测产物 `shape_dialect_after_pass.txt` 中，输入 `pd_op.data` 那一行是最关键的一行：

```
(%4) = "pd_op.data" () {... shape:[-1,72,-1,-1] ...} : () -> tensor<-1x72x-1x-1xf32>
        { (shape[S0, 72, S1, S2], data[NULL]) }
```

即 `-1` 的三个动态维被标注为符号 `S0`（batch）、`S1`（H）、`S2`（W），
固定通道维仍是常量 `72`。符号沿数据流传播且被正确化简：

| Value | 算子 | 符号 shape |
|-------|------|-----------|
| %4 | `pd_op.data` | `[S0, 72, S1, S2]` |
| %6 | `pd_op.pool2d`(adaptive→1×1) | `[S0, 72, 1, 1]` |
| %7 | `pd_op.conv2d` | `[S0, 18, 1, 1]` |
| %13 | `pd_op.conv2d` | `[S0, 72, 1, 1]` |
| %25 | `pd_op.multiply`(SE 相乘) | `[S0, 72, S1, S2]` |

注意 `pool2d` 之后 H/W 由 `S1`/`S2` 收敛为常量 `1`（adaptive_avg_pool 到 1×1 的语义在符号层面
被推导出来了），而输出又重新携带 `S1`/`S2`。这说明符号推导不是简单的占位符替换。

**7.2 导出 PIR 程序为可读文件（本 build 有缺陷，仅作辅助）**

`ApplyCinnPass` 入口处会通过 `PirToPyCodeConverter` 把 program 落盘（`add_cinn_pass.cc:338-341`），
FLAGS 定义见 `paddle/common/flags.cc:1834,1848`：

```bash
mkdir -p $PWD/ir_dump
export FLAGS_logging_pir_py_code_dir=$PWD/ir_dump
python ir_probe.py <case.py>
ls $PWD/ir_dump
# exec_programs.py  fusion_op_programs.py  group_op_programs.py
# original_programs.py  programs_example_input_tensor_meta.py
```

> **实测发现的两个问题，采集时须注意：**
>
> 1. **`FLAGS_logging_pir_py_code_dump_symbolic_dims=1` 会导致进程崩溃**：
>    `SystemError: (Fatal) The input data pointer is null. (at paddle/pir/src/core/storage_manager.cc:85)`，
>    抛在 `pir_partial_program.py:879` 的 `apply_cinn_pass`，且 `original_programs.py` 落盘为 0 字节。
>    已隔离确认：只设 `FLAGS_logging_pir_py_code_dir` 正常（EXIT=0，5 个文件），
>    加上 `dump_symbolic_dims=1` 必崩。
> 2. 因此不开该开关时，落盘文件里的 `__results_symbols_signature__` 全为 `s_null()`，
>    **不含符号维信息**，只能作为拓扑归档，不能充当符号化证据。
>
> 结论：符号化证据以 7.1（`GLOG_vmodule`）与 7.3（kernel 源码）为准，7.2 仅备查。

**7.3 导出 CINN 生成的 CUDA 源码（最关键一步，已实测）**

这一步能看到符号最终如何进入 kernel。

> **注意：`FLAGS_cinn_dump_group_lowered_func` / `FLAGS_cinn_dump_group_source_code` /
> `FLAGS_cinn_dump_group_ptx` 在本 build 中是死代码，设了也不会产出任何文件。**
> 这三个 flag 定义在 `paddle/cinn/runtime/flags.cc:202-227`，消费函数
> `CompilationInfoDumper::DumpLoweredFuncByGroupIndex` / `DumpSourceCodeByGroupIndex` /
> `DumpPtxCodeByGroupIndex` 定义在 `paddle/cinn/backends/compiler.cc:104-190`，
> 但全 `paddle/cinn` 目录下**只有声明（compiler.h:53-67）和定义，没有任何调用点**。
> 实测：设置这三个 flag 跑完 EXIT=0，dump 目录 0 个文件。

可用的替代路径是 `SourceCodePrint`（`paddle/cinn/backends/compiler.cc:220-253`），
由 `FLAGS_cinn_source_code_save_path` 驱动；若不设该路径则退化为
`LOG(INFO) << "[CUDA] source code:"`。另有 `compiler.cc:699` 的 `VLOG(3) << "[CUDA] C:"`。
实测采用前者：

```bash
export FLAGS_cinn_source_code_save_path=$PWD/cinn_source.cu
python ir_probe.py <case.py>
```

产物即归档的 `evidence_dynamicShape/cinn_source.cu`（8150 字节，2 个 group、共 8 个 kernel）。
审阅要点全部对上，且比预期更强：

**(a) 符号维以运行期整型形参进入 kernel。** SE 子图第二个 group 的 kernel 签名：

```cuda
__global__ void __launch_bounds__(1)
fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__...
  (const float* __restrict__ var, const float* __restrict__ var_1,
   const float* __restrict__ var_9, float* __restrict__ var_24,
   int32_t S0, int32_t S1, int32_t S2)
```

`S0`/`S1`/`S2` 就是 7.1 里 PIR 标注的那三个符号，作为 `int32_t` 形参传入；
通道维 72 则是字面常量。

**(b) 循环边界与索引是符号表达式，未固化任何具体尺寸。** 同一 kernel 体内：

```cuda
__builtin_assume(((int)blockIdx.x < (((S0 * S1) * S2) * 72)));
var_24[(int)blockIdx.x] = (var_9[(int)blockIdx.x] * max(min((((var_1[
    ((((int)blockIdx.x % ((S1 * S2) * 72)) / (S1 * S2))
     + (((int)blockIdx.x / ((S1 * S2) * 72)) * 72))]
  + var[(((int)blockIdx.x % ((S1 * S2) * 72)) / (S1 * S2))])
  * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
```

全篇搜不到 88 / 44 / 132 这类具体尺寸——只有 `S0`、`S1`、`S2` 和固定维 72。
这直接排除了"每个 shape 各编一份固化 kernel"的可能。

**(c) 同一 group 生成多个按形状分桶（bucket）的 kernel —— 这就是"按形状自动调优"的实体产物。**
每个 group 都不是一份 kernel，而是若干份，函数名里 `COND__` 之后编码了**该 kernel 的生效条件**。
命名编码规则见 `PredicatePrinter`（`paddle/cinn/backends/codegen_device_util.cc:117-174`）：
`_FPA_`=`(`、`_BPA_`=`)`、`MUL`=`*`，`GE/LE/GT/AND` 为比较与逻辑运算。
以第一个 group 为例，4 个 kernel 解码后是：

| kernel | 生效条件（解码后） | `__launch_bounds__` | 调度策略 |
|--------|-------------------|--------------------|---------|
| #1 | `1 <= S0*18 <= 1023`，且 `S0*18 <= INT32_MAX` | 1 | 一元素一 block，无循环 |
| #2 | `1 <= S0*18 <= 1023`，且 `S0*18 > INT32_MAX` | 1 | 同上，索引改 `int64_t` |
| #3 | `S0*18 >= 1024`，且 `S0*18 <= INT32_MAX` | 1024 | 1024 线程 × 4 次循环 tile |
| #4 | `S0*18 >= 1024`，且 `S0*18 > INT32_MAX` | 1024 | 同上，索引改 `int64_t` |

即：**编译期就按符号表达式的取值区间切了桶，小工作量走 `launch_bounds(1)` 的朴素 kernel，
大工作量走 1024 线程 + 4 倍循环展开的 tile 版本；同时按是否超 `INT32_MAX` 切了索引位宽。
运行期只需求值谓词、选中对应 kernel，不重新编译。**

这条链路在源码中可完整对上：
`ScheduleConfigManager::ExtractConfigs(target_, group_info_)` 产出
`map<BucketInfo, ScheduleConfig>` → `DynamicShapeGroupScheduler::InitBuckets()` 为每个 bucket
用 `MakeBucketPredicate()` 造符号谓词（`paddle/cinn/ir/group_schedule/dy_shape_group_scheduler.cc:48-83`,
`198-226`）→ 各 bucket 独立套 tactics 做 tile（同文件 `Schedule()`, 85-95）→
`GetIRs()` 返回 `vector<pair<SymbolicPredicate, Expr>>`
（`paddle/cinn/hlir/framework/pir/op_lowering_impl.cc:167,193`）→
`BucketLoweredFuncsWrapper::predicate2funcs`（同文件 242-245）→
codegen 时把谓词写进函数名，并生成 host 侧 switch 函数
（`codegen_device_util.cc:100-115` 的 `CreateSwitchFunction`）。

**7.4 可靠地数编译次数：codegen 事件计数 + 关缓存对照（已实测）**

`compiles` 日志计数不可用（步骤 6），但存在一个真正可比的计数：
`SourceCodePrint` 的 ofstream 在单例构造时以 `std::ios_base::out`（trunc）**打开一次**，
之后每次 `write()` 都是追加（`paddle/cinn/backends/compiler.cc:220-253`）。
因此 `FLAGS_cinn_source_code_save_path` 指向文件中 `extern "C" {` 块的个数
**等于该进程内发生的 codegen 事件数**：

```bash
grep -c 'extern "C" {' src.cu     # codegen 事件数
grep -c '__global__'   src.cu     # 分桶 kernel 总数
```

同时按需关闭编译缓存以排除 FusionInfo 复用
（`paddle/common/flags.cc:1266-1273`，命中判定见 `pir_compiler.cc:451-484`）：

```bash
export FLAGS_enable_cinn_compile_cache=false
```

**实验 1 —— 同一 case、7 个 shape、缓存开/关对照**（驱动 `cache_probe.py`，
被测 `picodet_l_640/SIR_17.py`）：

| shape | cache=on | cache=off |
|-------|---------:|----------:|
| S0 [1,72,88,88] | 5365.76 ms | 5862.54 ms |
| S1 [2,72,88,88] | 81.22 ms | 80.54 ms |
| S2 [1,72,44,44] | 0.79 ms | 0.62 ms |
| S3 [1,72,132,132] | 0.71 ms | 0.56 ms |
| S4 [2,72,44,44] | 0.49 ms | 0.37 ms |
| S5 [4,72,88,88] | 61.80 ms | 59.57 ms |
| S6 [2,72,88,88]（重复 S1） | 0.56 ms | 0.43 ms |
| **codegen 事件数** | **2** | **2** |
| `__global__` kernel 数 | 8 | 8 |

**codegen 事件恒为 2（= 该子图的 group 数），与 7 个不同 shape 无关、与缓存开关无关。
这是"一次编译服务全部 shape"的直接测量证明**，不依赖任何日志宏。
S1(81ms) / S5(62ms) 的耗时因此可以判定**不是**重编译或重调优；
这笔开销的正向归因见下方"补充：S1/S5 残余耗时的归因"。

**实验 2 —— 缓存开关本身是否生效**（驱动 `two_case_probe.py`，同一进程内先后跑
`ttfnet_pafnet_lite/SIR_22` 与 `centernet_mbv3_large/SIR_22`，两者拓扑相同、C 同为 72，
基准 shape 分别为 44×44 与 64×64）：

| | cache=on | cache=off |
|---|---:|---:|
| ttfnet/SIR_22 首次前向 | 5444.85 ms | 5963.00 ms |
| centernet/SIR_22 首次前向 | **141.77 ms** | **5698.96 ms** |
| codegen 事件数 | **2** | **4** |
| `__global__` kernel 数 | 8 | 16 |

codegen 事件 2→4 说明 `FLAGS_enable_cinn_compile_cache=false` **确实生效**，
这反过来确认实验 1 中"关缓存后仍只有 2 次 codegen"是有效证据而非 flag 未起作用。
同时解释了步骤 8 汇总表里 centernet 那个异常偏低的首次耗时（约 141~192ms）：
两个子图符号化后是同一张图，FusionInfo 一致，**命中了编译缓存**，
这本身也是符号化的一条旁证（基准 shape 不同却能复用同一份编译产物）。

> 注意：跨 case 复用只发生在"拓扑相同 **且** 通道数相同"时。不同 C 对应不同权重与不同算子形状，
> 属结构不同的子图，不复用。

> **图 9.3.3-4**：`[ShapeDialect]ShapeOptimizationPass Program` 段截图
> （需体现 `pd_op.data` 的 `(shape[S0, 72, S1, S2], data[NULL])`）。
>
> **图 9.3.3-5**：`cinn_source.cu` 中 `__global__` kernel 签名截图
> （需体现 `int32_t S0, int32_t S1, int32_t S2` 形参、索引为符号表达式，
> 以及同一 group 下多个带 `COND__` 分桶后缀的 kernel）。

#### 补充：S1/S5 残余耗时的归因（cuDNN 按 conv 问题 shape 选算法）

上面的 codegen 计数只证明了"S1/S5 的几十毫秒**不是**重编译"，没有说明它是什么。
下面三个探针给出正向归因，被测 `ppyolo_mbv3_large/SIR_77.py`（C=960，基准 `[1,960,10,10]`）。
现象特征是：**只有 batch 首次变大时慢，空间维怎么变都不慢，重复 shape 不慢。**

**探针 1 —— 拿掉 CINN，纯 eager 是否复现**（`eager_baseline_probe.py`，不 `to_static`）：

| variant | shape | eager_ms |
|---------|-------|---------:|
| S0 | [1,960,10,10] | 1924.06 |
| **S1** | [2,960,10,10] | **90.93** |
| S2 | [1,960,5,5] | 0.60 |
| S3 | [1,960,15,15] | 0.58 |
| S4 | [2,960,5,5] | 0.43 |
| **S5** | [4,960,10,10] | **60.83** |
| S6 | [2,960,10,10]（重复 S1） | 0.57 |

纯 eager 完整复现同一模式 ⇒ 与 CINN、与符号化编译、与分桶调优全部无关。

**探针 2 —— 逐 op 计时定位 + 显存对照**（`perop_probe.py`，op 间插 `synchronize()`）：

| variant | shape | pool | conv1 | relu | conv2 | hsig | mul | +resvMB |
|---------|-------|-----:|------:|-----:|------:|-----:|----:|--------:|
| S0 | [1,960,10,10] | 1.21 | 1593.35 | 107.22 | 37.64 | 0.17 | 0.24 | 0.4 |
| S1 | [2,960,10,10] | 0.20 | **48.55** | 0.05 | **34.23** | 0.05 | 0.46 | 1.5 |
| S2 | [1,960,5,5] | 0.11 | 0.30 | 0.04 | 0.17 | 0.03 | 0.04 | 0.0 |
| S3 | [1,960,15,15] | 0.07 | 0.16 | 0.03 | 0.14 | 0.03 | 0.22 | **1.6** |
| S4 | [2,960,5,5] | 0.06 | 0.17 | 0.03 | 0.15 | 0.03 | 0.03 | 0.0 |
| S5 | [4,960,10,10] | 0.07 | **30.93** | 0.03 | **29.90** | 0.04 | 0.29 | 2.9 |
| S6 | [2,960,10,10] | 0.09 | 0.20 | 0.03 | 0.16 | 0.03 | 0.04 | 0.0 |

开销 100% 落在两个 conv2d 上。**同时否证"显存分配增长"假说**：S3 的 reserved 增长
(+1.6MB) 比 S1 (+1.5MB) 还多，却只花 0.16ms。

**探针 3 —— 干净进程裸跑 conv2d**（`conv_shape_cost_probe.py`，只跑
`paddle.nn.Conv2D(960, 240, 1)` 喂 `[N,960,1,1]`）：

| N | ms | 说明 |
|--:|---:|------|
| 1 | 514.18 | 首次 N=1（含 cuDNN 首次初始化） |
| 2 | **51.49** | 首次 N=2 |
| 1 | 0.28 | N=1 重复 |
| 4 | **34.69** | 首次 N=4 |
| 2 | 0.24 | N=2 重复 |
| 8 | **43.95** | 首次 N=8 |
| 4 | 0.27 | N=4 重复 |

每个首次见到的 N 付一次几十毫秒，之后同 N 恒为 ~0.26ms —— cuDNN 对每个新 conv 问题 shape
做一次算法选择并缓存的典型行为。

**为什么只有 batch 敏感？** 结构原因：两个 conv 都接在 `adaptive_avg_pool2d(output_size=1)`
之后，输入已被 reduce 成 `[N, C, 1, 1]`，**conv 的问题 shape 只随 N 变**。S2/S3 只改 H/W，
对 conv 而言是同一 shape、直接命中缓存；S1/S5 改 batch 才是新 shape。

> **无效对照提醒（已自查纠正）**：若把裸 conv 循环放在整网循环之后、同一进程里跑，
> N=1/2/4 早被前面的整网 conv 走过、cuDNN 缓存已热，会测出平坦的 ~0.15ms，
> 从而被误读为"否证 cuDNN 假说"。**必须在全新进程里测。**

结论：该开销是 **conv2d 的 cuDNN 按 conv 问题 shape 选算法**的一次性成本，与 CINN 无关
（CINN 不 lower conv2d，走 cuDNN kernel）。这反而加强指标 3 的结论：动态 shape 下的残余
per-batch 开销可归因到 CINN 之外的算子库行为。


### 步骤 7 续：自动调优的触发方式、候选集、选择结果与代价

步骤 7.3(c) 只证明了"编译产物里存在多份按形状分桶的 schedule"。要完整支撑"根据不同张量
形状自动调优"，还需说明**调优由什么触发、候选集是什么、运行期实际选了哪个、调优带来多少
性能差异**。本节四项均已在 A100 上实测，产物为 `bucket_dispatch.log`、
`static_shape_control.log`、`tune_perf.log`。操作步骤见测试步骤文档 T8。

**(1) 触发方式（源码链路）**

调优策略由 `FLAGS_tile_config_policy`（`paddle/cinn/runtime/flags.cc:53`，默认 `"default"`）
选择，`InitScheduleConfig()` 据此装配 `ScheduleConfigManager`
（`paddle/cinn/ir/group_schedule/config/schedule_config_manager.cc:34-72`）：

| policy | 行为 |
|---|---|
| `default`（默认） | 走 `BuildScheduleConfig(group_info, target)` **规则生成**候选，不做实测 |
| `optimal` / `hybrid` | 从 `FileTileConfigDatabase` 读**已调优落盘**的配置，`hybrid` 以规则结果兜底 |
| `search` | 触发实测搜索（`ir/group_schedule/search/config_searcher.cc`）；运行期在 `cinn_jit_instruction.cc:121-152` 切到 CUDA Graph 重放 25 次计时打分 |

**必须如实声明的口径问题**：`FLAGS_enable_auto_tuner`（`flags.cc:243`）在整个 `paddle/`
中**只有定义、没有任何调用点**，是死 flag。因此默认配置下的"自动调优"指的是
**按形状区间分桶的规则化 schedule 特化**，而不是 profiling 搜索式 autotuner；后者需显式
设 `FLAGS_tile_config_policy=optimal|hybrid|search`。本轮验收按前者取证并声明该口径。

**(2) 候选集（编译期枚举，实测 4 桶/group）**

以第一个 group（谓词变量 `S0*18`）为例，`cinn_source.cu` 中 4 个 kernel 构成
`{numel 区间} × {索引位宽}` 的叉积：

| # | 生效谓词 | `launch_bounds` | 形参位宽 | 调度结构 | 源码行 |
|---|---|---|---|---|---|
| 1 | `1 <= S0*18 <= 1023` 且 `<= INT32_MAX` | 1 | `int32_t` | 一元素一 block，无循环 | `cinn_source.cu:5-8` |
| 2 | 同上但 `> INT32_MAX` | 1 | `int64_t` | 同上，索引转 64 位 | `:10-13` |
| 3 | `S0*18 >= 1024` 且 `<= INT32_MAX` | 1024 | `int32_t` | 1024 线程 × 4 次循环 tile | `:15-25` |
| 4 | 同上但 `> INT32_MAX` | 1024 | `int64_t` | 同上，索引转 64 位 | `:27-37` |

第二个 group（谓词变量 `S0*S1*S2*72`）结构相同，位于 `:45/:50/:55/:68`。
注意 #2 在逻辑上不可达（`<=1023` 与 `> INT32_MAX` 互斥），说明候选由规则机械生成、
未做可达性剪枝 —— 这是"规则化分桶"而非"搜索式调优"的又一处旁证。

**(3) 选择结果（运行期实测命中）**

用三个**跨过谓词分界线**的 shape 各跑一次，nsys `cuda_gpu_trace` 直接读出实际启动的
kernel 名（名字里就带谓词）与启动配置：

| 运行 shape | group1 `S0*18` | 命中桶 | grid / block / reg | group2 `S0*S1*S2*72` | 命中桶 | grid / block / reg |
|---|---|---|---|---|---|---|
| `[1,72,1,1]` | 18 | `LE1023` | 18×1×1 / 1 / 16 | 72 | `LE1023` | 72×1×1 / 1 / 16 |
| `[1,72,88,88]` | 18 | `LE1023` | 18×1×1 / 1 / 16 | 557568 | **`GE1024`** | 137×1×1 / 1024 / 22 |
| `[64,72,88,88]` | 1152 | **`GE1024`** | 1×1×1 / 1024 / 19 | 35684352 | `GE1024` | 8713×1×1 / 1024 / 22 |

两个 group 各按**自己的**谓词独立翻转：group1 在第 2→3 行翻转（`S0` 1→64），group2 在
第 1→2 行翻转（空间维 1×1→88×88）。不同桶的 grid、block、寄存器数实测均不同，说明是
**不同的已编译 schedule**，而不是同一 kernel 换启动参数。

三次运行的 `FLAGS_cinn_source_code_save_path` 产物 md5 **完全一致**
（`09c5ab1c800755b6f644f5074f67b1c4`）⇒ 一次编译即产出全部 4 桶，shape 只决定选哪个。

**补测（2026-09-07）：边界 shape 的数值精度。** 上表三个 shape 此前只验证了"派发到
不同桶 kernel"，未做数值对比（当时主测试的 70/70 PASS 不覆盖这两个边界 shape；
同日已将其追加为主用例 S7/S8 变体，见步骤 4 与下文）。
`boundary_shape_acc_probe.py` 以主测试同口径（eager 基准、同 seed、同动态
inputspec、atol=rtol=1e-5）补测：

| shape | eager vs CINN | max_abs_diff | 判定 |
|---|---|---|---|
| `[1,72,1,1]`（group2 翻小桶） | allclose=True | 2.98e-08 | PASS |
| `[64,72,88,88]`（group1 翻大桶） | allclose=True | 5.96e-08 | PASS |

max_abs_diff 在 float32 机器精度（2^-24 ≈ 6e-8）量级，与主测试其余 shape 变体的精度
水平一致。至此**每个出现过的 shape 同时具有派发证据与精度证据**。

另需如实说明 shape 的来源与演进：这三个 shape 最初是**测试装置按谓词反推注入的**
（`bucket_dispatch_probe.py` 的命令行参数），并非 case 文件自带——case 自带张量仅有
基准 `[1,72,88,88]`，原始 7 个 shape 变体（S0~S6）经逐谓词求值**全部不跨 1023/1024
桶边界**（group1 恒 `LE1023`、group2 恒 `GE1024`，即每 group 始终命中同一个 kernel）。
**2026-09-07 已将两个边界 shape 追加为主用例的 S7/S8 变体**并全量重跑：
10 case × 9 shape = **90/90 PASS**，且每个 case 的 S7/S8 `codegen` 列为 **0**、
全进程 codegen 合计仍为 17 ⇒ **桶翻转发生在运行期谓词求值，不触发任何重编译**。
至此主用例自身即同时覆盖"数值正确 + 编译复用 + 跨桶派发"三重证据，本节单 shape
探针与 T8.4 精度补测保留为独立复现入口。

**(4) 分桶归因于动态 shape（静态负对照）**

同一个 case、同一条 CINN 路径，只把 `input_spec` 换成全静态 `[1,72,88,88]`
（`static_shape_probe.py`）：

| 指标 | 动态 spec | 静态 spec |
|---|---|---|
| `extern "C"` group 数 | 2 | 2 |
| `__global__` kernel 数 | 8 | **2** |
| `COND__` 谓词桶数 | 8 | **0**（退化为 `COND_true__`） |
| 符号形参 `S0/S1/S2` | 8 | **0** |
| group1 实测 launch | grid 18 / block 1 | grid 1 / block **32** |
| group2 实测 launch | grid 137 / block 1024 | grid 8×72 / block **256** |

静态编译选出的 block（32 / 256）**不等于**动态四桶中的任何一个（1 / 1024）。这证明 tile
配置是 numel 的函数：尺寸已知就按确切值定一份，尺寸未知就按区间枚举多份、运行期选。

**(5) 调优代价量化**

同 case、同运行 shape `[1,72,88,88]`、各 55 次前向（5 warmup + 50 iters），
`tune_perf_probe.py`：

| group | 元素数 | static（确切尺寸） | dyn（分桶命中） | dyn / static |
|---|---|---|---|---|
| group1 | 18 | block 32，**2143 ns** | block 1，**2019 ns** | 0.94×（同属启动开销量级） |
| group2 | 557568 | grid 8×72 / block 256，**3951 ns** | grid 137 / block 1024，**7259 ns** | **1.84×** |
| 合计 | — | 6094 ns | 9278 ns | 1.52× |

min/max 离散度在 ±5% 内，差异远大于噪声。解读：分桶 schedule 确实随形状变化，但默认
`policy=default` 的规则化分桶在大 numel group 上比确切尺寸调优慢 1.84× —— 这个差距正是
`optimal`/`hybrid` 策略要填的。此处 static 一侧只是调优上界的**近似**，不等于
实测搜索出的最优配置；真正的"调优前 vs 调优后"已由下方 (6) 闭环。

**(6) 搜索式调优的读侧验证与收益实测（闭环 (5) 的缺口）**

走 `FLAGS_tile_config_policy=optimal` 的读侧（从 tile_config 数据库读 bucket→tileConfig，
替代 default 规则分桶），搜索以手工网格驱动（搜索器 `ScheduleConfigSearcher` 无产线
调用方，等价复刻其穷举 `Search()`）。机制要点（详见测试步骤文档 T9）：

- **Paddle wheel 自带预搜索库** `paddle/cinn_config/tile_config/`（A100-40GB / V100
  等机型；`paddle/__init__.py:856` import 时把 `CINN_CONFIG_PATH` 指向它，会覆盖
  用户先设的值，自定义库须在 import 之后用 `os.environ` 再覆盖）；本机 A100-80GB
  无条目。
- **空库 + policy=optimal 是硬崩溃不是静默回退**：`ir_analyzer.cc:107` 抛
  "Didn't find blocks in expr" → SIGABRT（exit 134）；只有 policy 值不合法才静默
  退回 default。
- **控制实验证明读侧生效**：单桶 JSON（warpNum=8）使 kernel 8→4 个、谓词变为
  自定义区间、全部 `__launch_bounds__(256)`=warpNum×32（default 为 1/1024 两档）。
- **网格搜索（11 候选，G2@[1,72,88,88]，n=55）**：最优 = warpNum 8 +
  spatialInnerNum 2（block 256，grid 1090）。
- **收益**：同日同法复测 default G2 = 7252 ns，搜索最优 = 5651 ns ⇒ **加速
  1.28×（−22.2%）**；与 static 上界差距从 1.84× 收窄到 1.43×。即使 block 同为
  1024（warpNum=32，6647 ns）仍快于 default —— tileConfig 还改变了 grid/循环结构。
- **双桶最终配置**（小 numel 桶 [1,1023] warpNum=1 + 大 numel 桶 [1024,INT32_MAX]
  (8,2)）：一次编译 8 kernel，运行期选桶——[1,72,88,88] 命中大桶 5651 ns，
  [1,72,1,1] 命中小桶 2380 ns（default 同 shape 2560 ns）。

**(7) 三场景 kernel 产物对照（default / optimal / static）**

先厘清场景口径：三场景都以 `paddle.jit.to_static(net, backend="CINN")`（动转静
开 CINN）为共同前提，按 input_spec 与 policy 细分；纯"动态图"（eager，不开
to_static）**不属于三者之列**——它不经过 CINN 编译，是数值基线（90/90 PASS 的
对照方）与 `eager_baseline.log` 的性能参照。

| 场景 | input_spec | policy | 含义 | 产物文件 |
|---|---|---|---|---|
| default | 动态 `(-1,72,-1,-1)` | `default`（默认） | 生产默认路径：规则化分桶 | `default_src.cu` |
| optimal | 动态 `(-1,72,-1,-1)` | `optimal` | 读搜索落库的 bucket→tileConfig | `optimal_src.cu` |
| static | 静态 `(1,72,88,88)` | —（无关） | 对照组/性能上界：确切尺寸单 kernel | `static_src.cu` |

三份 G2（557568 元素 SE 乘法）kernel 的关键差异（完整源码见归档文件）：

| 维度 | default | optimal | static |
|---|---|---|---|
| kernel 总数 | 8（2 桶×2 位宽×2 group） | 8（同结构，桶边界来自 JSON） | **2**（每 group 1 个） |
| 谓词 | `GE1024/LE1023`（规则定死） | 同型，边界可自定义 | `COND_true`（无谓词） |
| 形参 | `int32_t S0,S1,S2` | 同左 | **无** |
| launch_bounds | 1 / 1024 | 32 / 256 | 32 / 256 |
| grid | 1-D，137 block | 1-D，1090 block | **2-D 8×72** |
| 每元素索引开销 | 符号 div/mod | 符号 div/mod | **零**（编译期折叠） |
| G2 实测 | 7252 ns | 5651 ns | 3951 ns |

default 大桶（节选，`__launch_bounds__(1024)`、4 轮循环、每元素 2 次以上运行期
除/模）：

```cuda
for (int32_t i_..._0 = 0; i_..._0 < 4; i_..._0 += 1) {
  if (((((blockIdx.x*4)+i_..._0)*1024)+threadIdx.x) < ((S0*S1)*S2)*72) {
    var_1_local = var_1[(idx % ((S1*S2)*72))/(S1*S2) + (idx/((S1*S2)*72))*72];
    var_24[idx] = var_9_local * max(min(((var_1_local+var_local)*0.1667f)+0.5f,1.f),0.f);
```

optimal 大桶（同一谓词区间与索引结构，仅切分形态不同：`__launch_bounds__(256)`、
spatialInnerNum=2 → 2 轮循环、grid 1090）：

```cuda
for (int32_t i_..._7 = 0; i_..._7 < 2; i_..._7 += 1) {
  if (((((blockIdx.x*2)+i_..._7)*256)+threadIdx.x) < ((S0*S1)*S2)*72) {
    // 索引计算与 default 完全相同（仍是符号 div/mod）
```

static（无任何 S 形参；C=72 升格为 `blockIdx.y`、88×88=7744 等全为字面常量，
广播权重直接 `var_1[blockIdx.y]`，除/模全部消失）：

```cuda
void __launch_bounds__(256) fn_..._COND_true__kernel(
    const float* var, const float* var_1, const float* var_12, float* var_13) {
  __builtin_assume(((int)blockIdx.x < 8));
  __builtin_assume(((int)blockIdx.y < 72));
  ...
  var_1_local = var_1[(int)blockIdx.y];
```

三档性能的归因：**(i) static 最快**是因为确切尺寸让 `%`/`/` 编译期折叠、广播维
变成 grid.y、无谓词——这是"确切尺寸调优"的全部红利；**(ii) optimal 比 default 快
1.28×** 时索引计算结构完全相同，差的纯粹是切分形态——default 的 1024 线程×4 轮
只有 137 个 block，对 A100 的 108 个 SM 而言尾部分布不均、并发块少；optimal 的
256 线程×2 轮有 1090 个 block（每 SM ~10 个），负载均衡与延迟隐藏更好。**搜索
搜到的不是新算法，是同一算法下更匹配硬件的 block/tile 形态**——这是规则
（`warp_num` 一刀切）给不出的。**(iii) optimal 与 static 剩余 1.43×** 是符号化
本身（运行期 div/mod + 谓词 + 无法用 grid.y 承载广播维）的代价，即"支持任意
shape"的泛化成本，与 (5) 的定性一致。另注：default 小桶 `__launch_bounds__(1)`、
每 block 单线程（18 元素开 18 个单线程 block），规则对小 numel 的切分同样保守。


### 步骤 8：执行并收集结果

```bash
cd framework/e2e/PaddleLT_new

# 主用例：10 个 case × 9 个 shape（含 S7/S8 跨桶边界变体），动态 spec 单次编译
python test_dynamic_shape_cinn.py 2>&1 | tee dynamic_shape_cinn.log
echo "exit=$?"
```

正式验收只执行上面这一条。脚本的 `--static-tuning` / `--static-only` 探针**不要用于验收**：
它们对每个 shape 各自 `to_static` 一次，测的是"分别编译"而非动态 spec 下的行为，属额外观测。

单 case 报告形如：

```
[codegen 计数] CINN 生成源码落盘于: /tmp/cinn_codegen_xxxxxxxx/cinn_source.cu

CASE: Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py
inspec (动态编译入口): [-1, 72, -1, -1]
variant                   shape                     status           codegen  cinn_ms
S0-baseline               [1, 72, 88, 88]           PASS                   2  xxxx.xx
S1-batchx2                [2, 72, 88, 88]           PASS                   0    xx.xx
...
S7-spatial_min(小桶)        [1, 72, 1, 1]             PASS                   0    xx.xx
S8-batchx64(大桶)           [64, 72, 88, 88]          PASS                   0    xx.xx
PASS=9  FAIL=0  SKIP=0  (共 9)
本 case 新增 codegen 事件: 2 (未命中缓存的 group 数)
触发 codegen 的 shape:    ['S0-baseline [1, 72, 88, 88]']
未触发(复用已编译)的 shape:
    S1-batchx2 [2, 72, 88, 88]
    S2-spatial/2 [1, 72, 44, 44]
    ...
```

`inspec (动态编译入口)` 与 `触发/未触发 codegen` 两段都带具体 shape，目的是让单份日志
自身就能读出"一个含 `-1` 的编译入口 + 9 组具体 shape + 只有第一组触发 codegen"这条因果，
不必再对照脚本源码去反查 tag 对应哪个 shape。

`codegen` 列的读法：只有首个 shape 为正（= 该子图 fusion group 数），其余恒为 0
⇒ 一次符号化编译服务全部 shape。若某个后续 shape 出现正数，说明发生了 per-shape 重编译。
**S7/S8 的 codegen 同样为 0** ⇒ 跨桶翻转（运行期谓词求值换 kernel）也不触发重编译。

本轮 10 个 case 的 codegen 分布（全进程合计 17）：

| case | 通道数 C | codegen | 说明 |
|------|---------:|--------:|------|
| picodet_l_640 / SIR_17 | 72 | 2 | 各自编译 |
| picodet_m_320 / SIR_17 | 56 | 2 | 各自编译 |
| picodet_s_320 / SIR_17 | 44 | 2 | 各自编译 |
| ttfnet_pafnet_lite / SIR_22 | 72 | **1** | 1 个 group 命中缓存 |
| ssd_ssdlite / SIR_31 | 120 | 2 | 各自编译 |
| yolov3_mbv3_large / SIR_64 | 480 | 2 | 各自编译 |
| centernet_mbv3_large / SIR_22 | 72 | **0** | 两个 group 全命中缓存 |
| ppyolo_tiny / SIR_31 | 64 | 2 | 各自编译 |
| ppyolo_mbv3_small / SIR_49 | 240 | 2 | 各自编译 |
| ppyolo_mbv3_large / SIR_77 | 960 | 2 | 各自编译 |

通道数互不相同的 8 个 case 各自产生 2 次 codegen ⇒ **不同 C 视为不同子图，不复用编译**
（此前基于 `compiles=0` 得出的"跨 C 复用"推断已由此证否）；C=72 的三个同构 case 只有第一个
真正编译 ⇒ 编译缓存按 FusionInfo 命中。两条现象在主用例日志内同时可见，
与 7.4 实验 2 的独立对照结论一致。

"不同 C 不复用编译"另有最小独立对照 `three_c_probe.py`
（产物 `three_c_run.log` / `three_c_src.cu`，已复现验证）：同进程先后编译三个
拓扑相同、仅 C 不同的 picodet 子图（C=72/56/44），codegen = 6（3 × 2 group，
无一命中缓存）、分桶 kernel = 24，三个首次前向各 ~5s（完整编译耗时）——
与实验 2 的"同 C 命中缓存"（codegen 2→4）构成同一枚硬币的两面。

> **图 9.3.3-2**：单 case 明细表截图（体现 9 组 shape 全 PASS）。
>
> **图 9.3.3-3**：汇总段截图
> （10 行 `[OK ]`、`总计 shape 通过: 90/90`、`最终判定: 全部通过 (exit 0)`）。

### 步骤 9：结果归档

本轮实测产物已归档于 `/work/PaddleTest/evidence_dynamicShape/`：

| 文件 | 内容 | 对应步骤 |
|------|------|---------|
| `dynamic_shape_cinn.log` | 10 case × 9 shape（含 S7/S8 跨桶边界变体）全量日志，`exit 0`、90/90 PASS、codegen 合计仍 17 | 步骤 8 |
| `shape_dialect_after_pass.txt` | `[ShapeDialect]ShapeOptimizationPass Program` 段，含 `shape[S0, 72, S1, S2]` | 步骤 7.1 |
| `shape_dialect_origin.txt` | `[ShapeDialect]Origin Program` 段（pass 前对照） | 步骤 7.1 |
| `cinn_source.cu` | CINN 生成的 CUDA 源码，2 group / 8 个分桶 kernel | 步骤 7.3 |
| `ir_probe.py` | 采集 IR 用的最小驱动（不做 fd 重定向） | 步骤 7 |
| `cache_probe.py` | 实验 1 驱动：单 case 7 shape 逐个计时 | 步骤 7.4 |
| `cache_on_run.log` / `cache_on_src.cu` | 实验 1 缓存开：耗时日志 + 源码（codegen=2, kernel=8） | 步骤 7.4 |
| `cache_off_run.log` / `cache_off_src.cu` | 实验 1 缓存关：同上，codegen 仍为 2 | 步骤 7.4 |
| `two_case_probe.py` | 实验 2 驱动：同进程跑两个同拓扑同 C 的 case | 步骤 7.4 |
| `two_case_cache_on_run.log` / `..._src.cu` | 实验 2 缓存开：centernet 141.77ms，codegen=2 | 步骤 7.4 |
| `two_case_cache_off_run.log` / `..._src.cu` | 实验 2 缓存关：centernet 5698.96ms，codegen=4 | 步骤 7.4 |
| `eager_baseline_probe.py` / `eager_baseline.log` | 纯 eager 复现 S1/S5 慢（无 CINN） | 步骤 7 补充 |
| `perop_probe.py` / `perop.log` | 逐 op 计时 + 显存对照，定位到两个 conv2d | 步骤 7 补充 |
| `conv_shape_cost_probe.py` / `conv_shape_cost.log` | 干净进程裸 conv：每个首见 N 付一次几十毫秒 | 步骤 7 补充 |
| `ir_symbolic.log` | T4 全量 glog 输出（含两段 `[ShapeDialect]`） | 步骤 7.1 |
| `glog_multilink_evidence.log` | glog 多重链接缺陷取证（与本指标无关，见步骤 7.1 附注） | 步骤 7.1 附 |
| `glog_multilink_probe.py` / `glog_sso_matrix.sh` | 上述缺陷的最小复现驱动（SSO 分界线矩阵） | 步骤 7.1 附 |
| `bucket_dispatch_probe.py` | T8.1 驱动：单进程单 shape，配合 nsys 采分桶命中 | 步骤 7 续 |
| `bucket_dispatch.log` | 三个 shape 的运行期分桶命中表（含 md5 一致性） | 步骤 7 续 (3) |
| `static_shape_probe.py` / `static_shape_control.log` | T8.2 静态 spec 负对照：8 kernel → 2、`COND__`=0 | 步骤 7 续 (4) |
| `tune_perf_probe.py` / `tune_perf.log` | T8.3 static vs dyn 的 kernel 实测耗时（n=55） | 步骤 7 续 (5) |
| `three_c_probe.py` / `three_c_run.log` / `three_c_src.cu` | 三个不同 C 的 picodet 同进程对照：codegen=6、kernel=24 | 步骤 8 |
| `optimal_search.log` | T9 汇总：机制发现、控制实验、11 候选网格、收益 1.28× | 步骤 7 续 (6) |
| `optimal_perf_probe.py` / `kern_agg.py` | T9 探针（import 后覆盖 CINN_CONFIG_PATH）+ nsys 聚合 | 步骤 7 续 (6) |
| `optimal_tile_config.json` / `optimal_src.cu` | T9 双桶最终配置 + 生成的 CUDA 源码（8 kernel） | 步骤 7 续 (6) |
| `default_src.cu` / `static_src.cu` | 三场景对照另两份源码（规则分桶 / 确切尺寸单 kernel） | 步骤 7 续 (7) |
| `boundary_shape_acc_probe.py` / `boundary_shape_acc.log` | 边界 shape（[1,72,1,1]/[64,72,88,88]）eager vs CINN 精度补测，双 PASS | 步骤 7 续 (3) |
| `nsys_raw/`（76 文件，25 MB） | T8.1/T8.3/T9 全部原始 nsys 档案（`.nsys-rep` + trace CSV + run.log + src.cu），本节所有性能数字可由其二进制重放复算（`nsys stats -r cuda_gpu_trace`） | 步骤 7 续 (3)(5)(6)(7) |

连同步骤 1 的环境自检输出与 5 张截图，共同构成本指标的证据链。
观测编译行为时用步骤 7.4 的 codegen 事件计数（`extern "C" {` 块数）配合
`FLAGS_enable_cinn_compile_cache=false`，而不是日志计数，也不是仅清理磁盘缓存目录。

## 9.3.4 合格判据

以下 5 条全部满足即判定技术指标 3 合格。

**判据 1（功能正确性，必过）**
同一份权重、同一次 CINN 动态编译入口下，每个 case 的 S0~S8 共 9 组 shape
（含 S7/S8 两个跨桶边界变体）全部为 `PASS`，
不允许出现 `FAIL:numeric`、`FAIL:cinn`、`SKIP:shape` 中的任何一种。10 个 case 合计 **90/90 PASS**。

**判据 2（数值精度，必过）**
CINN 输出与动态图 eager 输出满足 `assert_allclose(atol=1e-5, rtol=1e-5)`，逐输出张量比对。
数值不一致一律判失败，不得降级为跳过。

**判据 3（动态场景覆盖度，必过）**
shape 变体必须同时覆盖三类动态变化，并覆盖跨桶边界：

- batch 维单独变化（S1、S5、**S8**）
- 空间维单独变化，含缩小、放大与归一（S2、S3、**S7**）
- batch 与空间维同时变化（S4）
- **跨 1023/1024 桶边界**（S7 翻入小桶、S8 翻入大桶），且翻桶不触发重编译

**判据 4（shape 符号化可验证，必过）**
必须提供中间 IR 与编译产物层面的证据（步骤 7），而非仅凭日志计数或耗时推断。四条均已实测取得：

- PIR 打印中，输入张量的动态维标注为符号 `S0`/`S1`/`S2`（由 `GetNextSymName()` 生成），
  固定通道维仍为常量 —— 实测 `pd_op.data` 标注为 `(shape[S0, 72, S1, S2], data[NULL])`；
- CINN 生成的 CUDA 源码中，动态维以运行期整型形参（`int32_t S0, S1, S2`）进入 kernel，
  循环边界与索引为符号表达式，全篇无 88/44/132 这类具体尺寸字面常量；
- 同一 group 生成多个带 `COND__` 谓词后缀的 kernel，谓词是符号表达式的取值区间
  （如 `1 <= S0*18 <= 1023` 走 `launch_bounds(1)`、`S0*18 >= 1024` 走 1024 线程 ×4 循环 tile），
  运行期按谓词选桶；
- codegen 事件数（`extern "C" {` 块个数）在 7 个不同 shape 下恒为 2（= group 数），
  且 `FLAGS_enable_cinn_compile_cache=false` 关缓存后仍为 2（步骤 7.4 实验 1）；
  该 flag 的有效性由实验 2（codegen 2→4）单独确认。主用例日志的 `codegen` 列复现同一形态：
  每个 case 只有 S0 为正、S1~S6 恒为 0。

前两条证明"shape 在 CINN 中是符号化的"，即编译产物本身是形状泛化的，
这正是同一次编译入口能服务多种具体 shape 的机制解释。
第三条进一步证明形状自适应的 tile 决策在编译期已按形状区间分桶落地。
第四条以可比的计数直接证明"一次编译服务全部 shape"，并否证"新 shape 触发重编译/重调优"。

编译次数一律以 codegen 事件计数为准（步骤 6 的实现、步骤 7.4 的定量对照）。
旧版脚本的 `compiles` 列已移除，其 0 值由 `LOG_FIRST_N(INFO, 1)` 造成，不得用于任何编译次数结论。

**判据 5（自动化判定，必过）**
脚本退出码为 0。退出码逻辑要求每个 case `FAIL=0 且 SKIP=0 且 PASS=总数`，与判据 1 严格对齐，
不存在"部分通过即算通过"的宽松口径。

### 明确不作为合格判据的项

以下三项属观测记录，不参与合格判定，理由如下：

1. **单个 case 的 codegen 合计绝对值不作为判据。** 该值受编译缓存影响：同拓扑同通道数的
   后来 case 会命中 FusionInfo 缓存而计少（本轮 ttfnet=1、centernet=0，全进程合计 17）。
   参与判定的是**分布形态**（只在首个 shape 为正）与步骤 7.4 的开/关缓存对照。
   反之，通道数互不相同的 8 个 case 各自产生 2 次 codegen，可作为"不同 C 不复用编译"的正面记录。
2. **旧版 `compiles` 列不作为判据，也不再输出。**
   它统计 `"Compiling subgraph with CINN backend"` 出现次数，而该语句是
   `add_cinn_pass.cc:334` 的 `LOG_FIRST_N(INFO, 1)`，进程内最多打印一次，
   导致第 2 个 case 起恒为 0。历史日志中的 `compiles=0` 不代表未编译，
   据此得出的"跨 C 复用同一次编译"推断已被证否。运行期是否重编译由步骤 7.4 实验 1 判定：
   关缓存后 codegen 在 7 个 shape 下恒为 2，**没有 per-shape 重编译**；
   S1/S5 的 81ms/62ms 已归因为 **conv2d 的 cuDNN 按 conv 问题 shape 选算法**的一次性成本
   （步骤 7 补充：纯 eager 无 CINN 同样复现，逐 op 定位到两个 conv，干净进程裸 conv 可独立复现），
   与 CINN 无关，更不是重调优。
   符号化 kernel 在运行期不做 per-shape 重编译属正确行为，不是缺陷。
3. **`--static-tuning` / `--static-only` 探针结果不作为判据。**
   这两个探针对每个 shape 各自 `to_static` 一次，测的是"分别编译"而非动态 spec 下的行为，
   仅作机制说明（测试计划 §10.1）；**正式验收执行不带这两个参数**。

### 关于"根据不同张量形状自动调优"的判据说明

该分句的能力落点在**编译期**而非运行期，且本轮已取得实测产物证据（步骤 7.3(c)）：
group_schedule 通过 `ScheduleConfigManager::ExtractConfigs` 得到若干 `(BucketInfo, ScheduleConfig)`，
`DynamicShapeGroupScheduler::InitBuckets()` 为每个 bucket 生成一个**符号谓词**，
各 bucket 独立套 tactics 做 tile，最终一个 group 产出多份调度策略不同的 kernel，
谓词编码在函数名里，host 侧 switch 函数在运行期求值选桶。

因此"根据不同张量形状自动调优"的验收方式是：
判据 1（单次编译的符号 kernel 能正确服务全部动态 shape）
+ 判据 4 第三条（编译产物中确实存在按形状区间分桶的多份 tile 策略）
+ 判据 4 第四条（codegen 事件数与 shape 数无关，说明分桶 kernel 在首次编译时一并生成），
而**不是**要求运行期观测到 per-shape 重编译事件——后者在符号化编译下本就不该发生。
机制的完整代码链路见步骤 7.3(c) 与测试计划 §10。

需要说明的局限：SE 子图 reduce 到 1×1，形状无关性强，观察到的分桶维度是
"总元素数区间 + 索引位宽"，而非 reduce 维相关的 tile 差异。
若要展现更丰富的按形状调优差异，应换用含**动态 reduce 维**的子图（softmax / layernorm）。

### 未覆盖项声明

本轮仅做 CINN vs eager 两方比对，未纳入 dy2st(backend=None) 静态图作为第三方基线。
该项不影响上述判据成立，属可选补强，若需排除 dy2st 自身引入的差异可后续补充。

