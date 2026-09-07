# 技术指标 3 测试方案（可变形状输入张量 + 按形状自动调优）

> 指标原文：支持可变形状（dynamic shape）输入张量，支持根据不同张量形状自动调优。
> **本文件只写"怎么操作"**：执行顺序、命令、预期输出、通过与否的判定、需要留存的记录。
> 实现原理与源码依据见 `PaddleTest_dynamicShape_9.3.3_9.3.4.md`；
> 用例清单与背景见 `PaddleTest_dynamicShape_schedule.md`。本文不重复这些内容。

## 1 测试对象与范围

| 项 | 内容 |
|---|---|
| 被测软件 | PaddlePaddle 3.5.0.dev（编译时开启 CINN，CUDA 可用） |
| 被测能力 | CINN 后端在动态 shape 输入下的功能正确性、数值精度、编译行为 |
| 主测试脚本 | `framework/e2e/PaddleLT_new/test_dynamic_shape_cinn.py` |
| 辅助驱动 | `evidence_dynamicShape/ir_probe.py`、`cache_probe.py`、`two_case_probe.py`、`eager_baseline_probe.py`、`perop_probe.py`、`conv_shape_cost_probe.py`（各自独立、无跨文件 import；目录平铺，文件依赖关系见 T7） |
| 测试用例 | `layercase/sublayer1000` 下 10 个 SE 结构子图（清单见测试计划 §8） |
| 测试环境 | Linux + NVIDIA GPU（本轮：A100/SM80，driver 535.230.02） |
| 测试方法 | 以动态图 eager 为基线的对比测试 + 编译产物静态核查 |
| 执行量 | 主用例 1 次全量执行 + 4 次辅助采集 + 3 次归因探针（T6.3，选做） |

## 2 执行总览

按下表顺序执行。T1 不通过不得继续；T3 为功能主项，T4~T6、T8 为证据采集项。

| 编号 | 步骤 | 目的 | 对应判据 |
|------|------|------|---------|
| T1 | 环境自检 | 确认 CINN/CUDA 可用，避免假通过 | 前置 |
| T2 | 用例前置核查 | 确认输入 spec 含动态维 `-1` | 前置 |
| T3 | 主用例全量执行 | 10 case × 7 shape 功能与数值正确性 | 判据 1/2/3/5 |
| T4 | 采集符号化 PIR | 证明动态维在 IR 中是符号 | 判据 4-a |
| T5 | 采集 CINN 生成的 CUDA 源码 | 证明符号进入 kernel、存在形状分桶 | 判据 4-b/4-c |
| T6 | 编译次数计数与关缓存对照 | 证明一次编译服务全部 shape | 判据 4-d |
| T8 | 自动调优的触发/选择/代价 | 补齐运行期选桶、静态负对照、性能对比 | 判据 4-e/4-f |
| T7 | 归档与截图 | 形成可复核的证据链 | 交付 |

> T8 在文档中排在 T6 之后、T7（归档）之前执行；编号沿用新增顺序，未重排既有编号，
> 以免和已发出的报告、截图编号对不上。

---

## T1 环境自检

**操作**

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
export PYTHONPATH=/work/Paddle/build/python
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source /work/env3.10/bin/activate

nvidia-smi
python -c "import paddle; print(paddle.__version__, \
paddle.is_compiled_with_cinn(), paddle.device.is_compiled_with_cuda())"
```

**预期**

```
3.5.0.dev20260526 True True
```

**判定**：两个 `True` 必须同时出现。任一为 `False` 则终止测试并记为环境不满足
（主脚本遇此情况自身也会 `exit 2`，不会输出通过结论）。

**留存**：终端截图（**图 9.3.3-1**），需同框体现 driver 版本、GPU 型号、
`is_compiled_with_cinn()=True`。

**常见问题**：`import paddle` 报 `libcuda.so.1: cannot open shared object file`
不是缺驱动，是 `LD_LIBRARY_PATH` 未包含 `/usr/lib64`，按上面第一条 export 补上即可。

---

## T2 用例前置核查

**操作**（抽验 1 个用例即可，其余同构）

```bash
cd /work/PaddleTest/framework/e2e/PaddleLT_new
python -c "
import importlib.util
p='layercase/sublayer1000/Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py'
s=importlib.util.spec_from_file_location('m', p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print('tensor :', [t.shape for t in m.create_tensor_inputs()])
print('inspec :', [sp.shape for sp in m.create_inputspec()])
"
```

**预期**

```
tensor : [[1, 72, 88, 88]]
inspec : [[-1, 72, -1, -1]]
```

**判定**：`create_inputspec()` 的 N/H/W 必须为 `-1`，通道维为固定正整数。
若全为具体值，说明用例未声明动态维，本指标无从测起，需先修用例。

---

## T3 主用例全量执行

被测 shape 矩阵（脚本内置，无需手工构造；下表以 picodet_l_640 基准 [1,72,88,88] 为例）：

| 组别 | 缩放因子 (batch, spatial) | 实际 shape | 覆盖点 |
|------|--------------------------|-----------|--------|
| S0 | (1, 1) | [1, 72, 88, 88] | 基准 |
| S1 | (2, 1) | [2, 72, 88, 88] | batch 变化 |
| S2 | (1, 0.5) | [1, 72, 44, 44] | 空间维缩小 |
| S3 | (1, 1.5) | [1, 72, 132, 132] | 空间维放大 |
| S4 | (2, 0.5) | [2, 72, 44, 44] | 多维同时变化 |
| S5 | (4, 1) | [4, 72, 88, 88] | batch 进一步放大 |
| S6 | (2, 1) | [2, 72, 88, 88] | 重复 S1 的对照组 |

**操作**

```bash
cd /work/PaddleTest/framework/e2e/PaddleLT_new
python test_dynamic_shape_cinn.py 2>&1 | tee dynamic_shape_cinn.log
echo "exit=$?"
```

> 正式验收**只执行上面这一条命令**，不要加 `--static-tuning` / `--static-only`
> （这两个探针每个 shape 各自 `to_static` 一次，属额外观测，不参与验收）。
>
> 脚本自身即以 codegen 事件计数（`FLAGS_cinn_source_code_save_path` 落盘源码中
> `extern "C" {` 块数的逐 shape 差分），落盘路径在开头一行打印；若外部已 export
> 该 FLAGS，则沿用外部路径。

**预期**（单 case 报告与全局汇总）

```
[codegen 计数] CINN 生成源码落盘于: /tmp/cinn_codegen_xxxxxxxx/cinn_source.cu

CASE: Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py
inspec (动态编译入口): [-1, 72, -1, -1]
variant                   shape                     status           codegen  cinn_ms
S0-baseline               [1, 72, 88, 88]           PASS                   2  xxxx.xx
S1-batchx2                [2, 72, 88, 88]           PASS                   0    xx.xx
...
PASS=7  FAIL=0  SKIP=0  (共 7)
本 case 新增 codegen 事件: 2 (未命中缓存的 group 数)
触发 codegen 的 shape:    ['S0-baseline [1, 72, 88, 88]']
未触发(复用已编译)的 shape:
    S1-batchx2 [2, 72, 88, 88]
    S2-spatial/2 [1, 72, 44, 44]
    S3-spatialx1.5 [1, 72, 132, 132]
    S4-batchx2-spatial/2 [2, 72, 44, 44]
    S5-batchx4 [4, 72, 88, 88]
    S6-batchx2(repeat->缓存) [2, 72, 88, 88]
...
总计 shape 通过: 70/70
全进程 codegen 事件合计: 17（每个 case 各自编译，与 shape 个数无关；源码见 ...）
最终判定: 全部通过 (exit 0)
```

报告开头的 `inspec (动态编译入口)` 一行即 T2 核查的那个含 `-1` 的 spec，打在日志里是为了让
"这一次编译的入口是动态的"与后面 7 组具体 shape 在同一份产物内自证；`触发/未触发 codegen`
两段列出 tag 与其对应的具体 shape，便于直接核对是哪些具体 shape 复用了同一份编译产物。

**判定**

- 每个 case `PASS=7、FAIL=0、SKIP=0`，10 个 case 合计 `70/70`；
- 进程退出码为 `0`；
- `codegen` 列的读法：**只在首个 shape 上为正数（= 该子图 group 数），其余 shape 为 0**。
  这正是判据 4-d 想要的形状——一次编译服务全部 shape。若某个后续 shape 出现正数，
  说明发生了 per-shape 重编译，需回到 T6 复核。
- 单个 case 的 `本 case 新增 codegen 事件` 允许为 **0 或小于 group 数**：这是编译缓存命中
  （拓扑相同且通道数相同的子图跨 case 复用），不是失败。本轮实测分布见下表。

**本轮实测 codegen 分布**（全进程合计 17）

| # | case | codegen | 说明 |
|---|------|--------:|------|
| 1 | picodet_l_640 / SIR_17 (C=72) | 2 | 各自编译 |
| 2 | picodet_m_320 / SIR_17 (C=56) | 2 | 各自编译 |
| 3 | picodet_s_320 / SIR_17 (C=44) | 2 | 各自编译 |
| 4 | ttfnet_pafnet_lite / SIR_22 (C=72) | **1** | 与 case 1 部分同构，1 个 group 命中缓存 |
| 5 | ssd_ssdlite / SIR_31 (C=120) | 2 | 各自编译 |
| 6 | yolov3_mbv3_large / SIR_64 (C=480) | 2 | 各自编译 |
| 7 | centernet_mbv3_large / SIR_22 (C=72) | **0** | 与 case 1/4 同构同 C，两个 group 全命中缓存 |
| 8 | ppyolo_tiny / SIR_31 (C=64) | 2 | 各自编译 |
| 9 | ppyolo_mbv3_small / SIR_49 (C=240) | 2 | 各自编译 |
| 10 | ppyolo_mbv3_large / SIR_77 (C=960) | 2 | 各自编译 |

通道数各不相同的 8 个 case 各自产生 2 次 codegen ⇒ **不同 C 视为不同子图、不复用编译**；
C=72 的三个同构 case 只有第一个真正编译 ⇒ 缓存按 FusionInfo 命中。两条现象在主用例
日志里同时可见，与 T6.2 的独立对照实验结论一致。

**失败分类处理**

| status | 含义 | 处理 |
|--------|------|------|
| `FAIL:numeric` | CINN 与 eager 输出不一致 | 判不合格，保留日志定位算子 |
| `FAIL:cinn` | CINN 前向抛异常 | 判不合格 |
| `SKIP:shape` | eager 侧该 shape 即非法 | 判不合格（本轮用例不应出现） |

**留存**：`dynamic_shape_cinn.log`；单 case 明细表截图（**图 9.3.3-2**）、
汇总段截图（**图 9.3.3-3**，含 10 行 `[OK ]` 与 `70/70`、`exit 0`）。

---

## T4 采集符号化 PIR

采集动态维在 PIR 中被标注为符号的证据。必须用 `ir_probe.py`，
不能用主脚本（主脚本做 fd 级 stderr 重定向，会吞掉 C++ glog 输出）。

**操作**

```bash
cd /work/PaddleTest/framework/e2e/PaddleLT_new
CASE=layercase/sublayer1000/Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py

export GLOG_logtostderr=1
export GLOG_vmodule='shape_o*=3'                   # 不要用全局 GLOG_v=3
python ../../../evidence_dynamicShape/ir_probe.py "$CASE" > ir_symbolic.log 2>&1
echo "exit=$?"

grep -n "\[ShapeDialect\]" ir_symbolic.log
grep -n "pd_op.data" ir_symbolic.log | head -2
```

> `shape_o*=3` 是 `shape_optimization_pass=3` 的通配写法（glog vmodule 支持 `*`/`?`），
> 两者匹配到的文件与输出**逐字节相同**。这里必须用短写法：本 build 中长度 >= 16 字节的
> `GLOG_*` 字符串型环境变量会让进程在退出期 double free 并挂死（见本步末「已知缺陷」）。

**预期**

```
5:===================== [ShapeDialect]Origin Program =====================
129:===================== [ShapeDialect]ShapeOptimizationPass Program =====================
```

pass 后的段落中，输入 `pd_op.data` 一行应带符号标注：

```
(%4) = "pd_op.data" () {... shape:[-1,72,-1,-1] ...} : () -> tensor<-1x72x-1x-1xf32>
        { (shape[S0, 72, S1, S2], data[NULL]) }
```

**判定**：三个 `-1` 维分别标注为 `S0`/`S1`/`S2`，固定通道维仍为常量 `72`。
若动态维被标注为具体数字，则符号化未生效，判据 4 不成立。

**留存**：把两段 `[ShapeDialect]` 分别截取存为
`shape_dialect_origin.txt` / `shape_dialect_after_pass.txt`；
`pd_op.data` 那一行截图（**图 9.3.3-4**）。

**注意**：不要开 `FLAGS_logging_pir_py_code_dump_symbolic_dims=1`，本 build 下会导致
`storage_manager.cc:85` 崩溃；不开该开关时落盘的 py code 不含符号维，无法当证据用。

**已知缺陷（与本指标无关，但会影响本步的执行体验）**：本 build 中只要某个 `GLOG_*`
**字符串型**环境变量（`vmodule` / `log_dir` / `log_backtrace_at`）的值长度 >= 16 字节，
进程就会在**打完全部输出、进入退出析构阶段**时触发 glibc 堆校验并挂死
（报错文本漂移：`corrupted double-linked list` / `double free or corruption (!prev)` /
`malloc_consolidate(): unaligned fastbin chunk detected`），需 Ctrl+C。

- 根因是 `libglog.a` 被静态链进 4 个 .so（`base/libpaddle.so`、`libs/libphi_core.so`、
  `libs/libphi_gpu.so`、`libs/libcinnapi.so`）且 glog 全局符号均导出：ELF 符号插入使
  4 份 `fLS::FLAGS_vmodule_buf` 数据坍缩为 1 个实例，但 4 个静态初始化器各跑一次，
  同一个 `std::string` 被构造 4 次并注册 4 次 `__cxa_atexit` 析构 ⇒ 退出期同一堆指针被
  free 4 次。`<= 15` 字节走 SSO（无堆缓冲）故无害，分界线实测正好在 15/16 字节；
  `GLOG_v` 是 int32、不涉堆，故 `GLOG_v=3` 不复现。
- 与 CINN / 符号化 shape pass / IR dump **无关**：`import paddle` 一行加长 vmodule 即复现
  （无 GPU、无 tensor、无 `to_static`）。
- 挂死而非直接退出是次级原因：Paddle 的 glog failure signal handler
  （`init.cc:437` `InstallFailureSignalHandler` + `SignalHandle`）在 SIGABRT 里做
  `ostringstream` + `backtrace_symbols`，都要 malloc，而 abort 是从持 `main_arena` 锁的
  `free` 校验路径发出的 ⇒ 自死锁。
- **本步采用的规避即上面的短 glob 写法**，`exit=0`，产物与长写法逐字节相同（除时间戳）。
  不要用 `os._exit(0)` 跳过析构：那会把 abort 压成 `EXIT=0`，对验收产物是完整性问题。

---

## T5 采集 CINN 生成的 CUDA 源码

**操作**

```bash
cd /work/PaddleTest/framework/e2e/PaddleLT_new
CASE=layercase/sublayer1000/Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py

export FLAGS_cinn_source_code_save_path=$PWD/cinn_source.cu
python ../../../evidence_dynamicShape/ir_probe.py "$CASE" > /dev/null 2>&1

grep -n "int32_t S0"  cinn_source.cu | head          # (a) 符号形参
grep -c "COND__"      cinn_source.cu                 # (c) 分桶 kernel 个数
grep -E "\b(88|132|44)\b" cinn_source.cu | grep -v COND__ | head   # 应无尺寸常量
```

> `FLAGS_cinn_dump_group_lowered_func` / `_source_code` / `_ptx` 三个 flag 在本 build 中
> 没有调用点，设了也不产出文件，不要用。

**预期**（`cinn_source.cu` 约 8 KB，2 个 `extern "C"` group、共 8 个 `__global__` kernel）

- (a) kernel 签名末尾带符号形参：

```cuda
__global__ void __launch_bounds__(1) fn_..._COND__..._kernel(
    const float* __restrict__ var, ..., float* __restrict__ var_24,
    int32_t S0, int32_t S1, int32_t S2)
```

- (b) 边界与索引是符号表达式，例如
  `__builtin_assume(((int)blockIdx.x < (((S0 * S1) * S2) * 72)));`，
  且全文搜不到 88 / 44 / 132 这类具体尺寸常量；
- (c) 同一 group 有多个 `COND__` 后缀 kernel，谓词按符号表达式取值区间分桶。
  以第一个 group 为例解码后应为 4 桶（名字编码：`_FPA_`=`(`、`_BPA_`=`)`、`MUL`=`*`）：

| kernel | 生效条件 | `__launch_bounds__` | 调度 |
|--------|---------|--------------------|------|
| #1 | `1 <= S0*18 <= 1023`，`<= INT32_MAX` | 1 | 一元素一 block |
| #2 | 同上，`> INT32_MAX` | 1 | 索引改 int64 |
| #3 | `S0*18 >= 1024`，`<= INT32_MAX` | 1024 | 1024 线程 × 4 循环 tile |
| #4 | 同上，`> INT32_MAX` | 1024 | 索引改 int64 |

**判定**：(a)(b)(c) 三点同时成立。(c) 是"按不同张量形状自动调优"的直接产物证据 ——
调优发生在编译期，一次编译生成多份 tile 策略，运行期按谓词选桶。

**留存**：`cinn_source.cu`；kernel 签名 + `COND__` 分桶清单截图（**图 9.3.3-5**）。

---

## T6 编译次数计数与关缓存对照

T3 的 `codegen` 列已能反映每个 shape 是否触发 codegen；本步进一步做**缓存开/关对照**，
排除"看起来没重编译只是因为命中了缓存"这一质疑。计数方式与 T3 相同：
`FLAGS_cinn_source_code_save_path` 指向的文件里 `extern "C" {` 块的个数
== 该进程内发生的 codegen 次数。

```bash
grep -c 'extern "C" {' <src.cu>    # codegen 事件数
grep -c '__global__'   <src.cu>    # 分桶 kernel 总数
```

### T6.1 单 case × 7 shape，缓存开/关对照

**操作**

```bash
cd /work/PaddleTest/framework/e2e/PaddleLT_new
CASE=layercase/sublayer1000/Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py
PROBE=../../../evidence_dynamicShape/cache_probe.py

for MODE in on off; do
  mkdir -p /tmp/cache_$MODE
  export FLAGS_cinn_source_code_save_path=/tmp/cache_$MODE/src.cu
  [ "$MODE" = off ] && export FLAGS_enable_cinn_compile_cache=false \
                    || unset FLAGS_enable_cinn_compile_cache
  python "$PROBE" "$CASE" > /tmp/cache_$MODE/run.log 2>&1
  echo "$MODE: codegen=$(grep -c 'extern \"C\" {' /tmp/cache_$MODE/src.cu)" \
       "kernel=$(grep -c '__global__' /tmp/cache_$MODE/src.cu)"
done
```

**预期**

```
on:  codegen=2 kernel=8
off: codegen=2 kernel=8
```

> `cache_probe.py` 内的 7 组 shape 是按基准 `[1,72,88,88]` 写死的，换 case 时需同步调整；
> `ir_probe.py` 同理（内含 3 组 shape）。

**判定**：codegen 事件数等于该子图的 group 数（本例 2），且**与 shape 个数无关、
关缓存后不增加** ⇒ 一次编译服务全部 7 个 shape，无 per-shape 重编译。
若关缓存后变成 14（7 shape × 2 group），则说明每个 shape 都重编译了，判据 4-d 不成立。

**顺带记录**：`run.log` 中各 shape 首次前向耗时。S0 数千毫秒属编译成本；
S1/S5 出现几十毫秒、其余亚毫秒，**不是重调优**——依据就是 codegen 恒为 2。
重复 S1 的 S6 应回落到亚毫秒。这几十毫秒的正向归因见 T6.3（cuDNN 按 conv shape 选算法）。

### T6.2 确认关缓存开关本身生效（对照实验）

若不做这一步，T6.1 的"关缓存后仍为 2"可能被质疑为 flag 未起作用。

**操作**：同 T6.1 的循环结构，驱动换成 `two_case_probe.py`，在同一进程内先后跑两个
**拓扑相同、通道数相同（C=72）、基准 shape 不同**的 case。

```bash
cd /work/PaddleTest/framework/e2e/PaddleLT_new
B=layercase/sublayer1000/Det_cases
C1=$B/ttfnet_pafnet_lite_mobilenet_v3_20x_coco/SIR_22.py      # 基准 44×44
C2=$B/centernet_centernet_mbv3_large_140e_coco/SIR_22.py      # 基准 64×64

for MODE in on off; do
  mkdir -p /tmp/two_$MODE
  export FLAGS_cinn_source_code_save_path=/tmp/two_$MODE/src.cu
  [ "$MODE" = off ] && export FLAGS_enable_cinn_compile_cache=false \
                    || unset FLAGS_enable_cinn_compile_cache
  python ../../../evidence_dynamicShape/two_case_probe.py "$C1" "$C2" \
      > /tmp/two_$MODE/run.log 2>&1
  echo "$MODE: codegen=$(grep -c 'extern \"C\" {' /tmp/two_$MODE/src.cu)" \
       "kernel=$(grep -c '__global__' /tmp/two_$MODE/src.cu)"
done
```

**预期**

| | cache=on | cache=off |
|---|---:|---:|
| 第 1 个 case 首次前向 | 数千 ms | 数千 ms |
| 第 2 个 case 首次前向 | **百毫秒量级**（命中缓存） | **数千 ms**（重新编译） |
| codegen 事件数 | **2** | **4** |
| `__global__` kernel 数 | 8 | 16 |

**判定**：codegen 由 2 变 4 即证明 `FLAGS_enable_cinn_compile_cache=false` 生效，
从而 T6.1 的结论有效。第 2 个 case 在开缓存时耗时极低是缓存命中，
同时也旁证了符号化（基准 shape 不同仍复用同一份编译产物）。

> 跨 case 复用只在"拓扑相同**且**通道数相同"时发生。通道数不同即视为不同子图，不复用。

**留存**：四组 `run.log` 与 `src.cu`。

### T6.3 归因 S1/S5 的几十毫秒（三探针，选做但推荐）

T6.1 只证明了那几十毫秒**不是**重编译。本步给出正向归因，避免评审时留下悬案。
被测 `ppyolo_mbv3_large/SIR_77.py`（C=960，基准 `[1,960,10,10]`，两个 conv 接在
`adaptive_avg_pool2d(output_size=1)` 之后）。

**操作**

```bash
cd /work/PaddleTest/framework/e2e/PaddleLT_new
E=/work/PaddleTest/evidence_dynamicShape
CASE=layercase/sublayer1000/Det_cases/ppyolo_ppyolo_mbv3_large_coco/SIR_77.py
python $E/eager_baseline_probe.py "$CASE" > $E/eager_baseline.log   2>&1
python $E/perop_probe.py          "$CASE" > $E/perop.log            2>&1
python $E/conv_shape_cost_probe.py         > $E/conv_shape_cost.log 2>&1
```

**预期**

| 探针 | 预期观测 | 说明 |
|------|---------|------|
| `eager_baseline_probe.py`（纯 eager，不 to_static） | S1≈91ms、S5≈61ms，S2/S3/S4/S6 亚毫秒 | 无 CINN 也复现 ⇒ 与 CINN 无关 |
| `perop_probe.py`（逐 op 计时 + 显存） | 开销全在 conv1+conv2（S1: 48.55+34.23，S5: 30.93+29.90），其余 op 亚毫秒 | 定位到 conv |
| 同上，显存列 | S3 `+resv`=1.6MB > S1 的 1.5MB，但 S3 只花 0.16ms | 否证"显存增长致慢" |
| `conv_shape_cost_probe.py`（干净进程裸 conv） | 首次 N=2/4/8 各 51.49/34.69/43.95ms，重复同 N 恒 ~0.26ms | cuDNN 按 conv shape 选算法并缓存 |

**判定**：三条同时成立即可判定该开销为 cuDNN 的一次性算法选择成本，与 CINN 无关
（CINN 不 lower conv2d）。只有 batch 敏感、H/W 不敏感的原因是 conv 前有全局池化，
其问题 shape 为 `[N, C, 1, 1]`，只随 N 变。

> **必须用干净进程跑第 3 个探针。** 若把裸 conv 放在整网循环之后、同一进程里跑，
> N=1/2/4 的 cuDNN 缓存已被前面的整网 conv 热过，会测出平坦的 ~0.15ms 并误判为
> "cuDNN 假说不成立"。

**留存**：`eager_baseline.log`、`perop.log`、`conv_shape_cost.log`。

---

## T8 自动调优的触发、选择与代价（补评审意见）

T5 只证明了"编译产物里存在多份按形状分桶的 schedule"。评审指出这不足以证明"根据不同
张量形状自动调优"，还需要**触发方式、候选配置、选择结果、调优前后性能对比**四项。
T8 三步把后三项补齐，触发方式在本节开头以源码位置给出。

### T8.0 触发方式（源码定位，无需执行）

| 环节 | 位置 | 说明 |
|------|------|------|
| 策略开关 | `paddle/cinn/runtime/flags.cc:53` | `FLAGS_tile_config_policy`，默认 `"default"` |
| 策略分发 | `paddle/cinn/ir/group_schedule/config/schedule_config_manager.cc:34-72` | `default` → 规则生成 `BuildScheduleConfig(group_info, target)`；`optimal`/`hybrid` → 从 `FileTileConfigDatabase` 读已调优配置；`search` → 实测搜索 |
| 搜索器 | `paddle/cinn/ir/group_schedule/search/config_searcher.cc` | 候选枚举 + 实测打分 |
| 搜索期计时 | `paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc:121-152` | `policy=="search"` 时切到 CUDA Graph 重放 25 次计时 |

**必须如实声明的口径问题**：`FLAGS_enable_auto_tuner`（`flags.cc:243`）在整个 `paddle/`
里**只有定义、没有任何调用点**，是死 flag。因此默认配置（`policy=default`）下的"自动调优"
= **按形状区间分桶的规则化 schedule 特化**，不是 profiling 搜索式 autotuner。若验收方把
"自动调优"定义为后者，须显式设 `FLAGS_tile_config_policy=optimal|hybrid|search` 才走那条路。
本轮验收按前者取证，并在报告中写明该定义。

### T8.1 运行期分桶命中（选择结果）

**操作**

```bash
cd /work/PaddleTest/framework/e2e/PaddleLT_new
CASE=layercase/sublayer1000/Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py
mkdir -p /tmp/e1

# ir_probe.py 的 SHAPES 改为单一 shape 后分三次跑；三个 shape 故意跨过谓词分界线：
#   group1 谓词 = S0*18       → S0=1 得 18（<=1023）、S0=64 得 1152（>=1024）
#   group2 谓词 = S0*S1*S2*72 → 1x1 空间得 72（<=1023）、88x88 得 557568（>=1024）
for SH in 1x72x1x1 1x72x88x88 64x72x88x88; do
  FLAGS_cinn_source_code_save_path=/tmp/e1/src_$SH.cu \
  nsys profile -t cuda -o /tmp/e1/rep_$SH --force-overwrite=true \
  python ../../../evidence_dynamicShape/ir_probe.py "$CASE" > /tmp/e1/run_$SH.log 2>&1
  echo "$SH exit=$?"
done

md5sum /tmp/e1/src_*.cu                      # 三次编译产物是否一致
for SH in 1x72x1x1 1x72x88x88 64x72x88x88; do
  nsys stats --report cuda_gpu_trace --format csv /tmp/e1/rep_$SH.nsys-rep 2>/dev/null \
    | grep fn_reshape | awk -F',' '{print $4"x"$5"x"$6, $7, $NF}' | sort -u
done
```

**实测结果**（A100/SM80，`bucket_dispatch.log`）

| 运行 shape | group1 谓词 `S0*18` | 命中桶 | 实测 grid/block/reg | group2 谓词 `S0*S1*S2*72` | 命中桶 | 实测 grid/block/reg |
|---|---|---|---|---|---|---|
| `[1,72,1,1]` | 18 | `LE1023` | 18×1×1 / 1 / 16 | 72 | `LE1023` | 72×1×1 / 1 / 16 |
| `[1,72,88,88]` | 18 | `LE1023` | 18×1×1 / 1 / 16 | 557568 | **`GE1024`** | 137×1×1 / 1024 / 22 |
| `[64,72,88,88]` | 1152 | **`GE1024`** | 1×1×1 / 1024 / 19 | 35684352 | `GE1024` | 8713×1×1 / 1024 / 22 |

三次运行的 `src_*.cu` md5 **完全相同**（`09c5ab1c800755b6f644f5074f67b1c4`，与归档的
`cinn_source.cu` 一致）⇒ 同一份编译产物内含全部 4 桶，shape 只决定**启动哪一个**。

**判定**：三点同时成立即通过 ——
(a) 两个 group 各自按**自己的**谓词独立选桶（group1 在第 2→3 行翻转，group2 在第 1→2 行翻转）；
(b) 命中的 kernel 名字里的谓词与实际 shape 代入后的算术结果一致；
(c) 不同桶的 `grid/block/寄存器数`实测不同 ⇒ 是不同的已编译 schedule，不是同一 kernel 换启动参数。

**留存**：`bucket_dispatch.log`。

### T8.2 静态 shape 负对照（证明分桶是动态 shape 特有产物）

**操作**

```bash
cd /work/PaddleTest/framework/e2e/PaddleLT_new
CASE=layercase/sublayer1000/Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py
mkdir -p /tmp/e2

# static_shape_probe.py 与 ir_probe.py 唯一差别：input_spec 用全静态 shape，不调
# create_inputspec()（其 shape 含 -1）
FLAGS_cinn_source_code_save_path=/tmp/e2/src_static.cu \
nsys profile -t cuda -o /tmp/e2/rep_static --force-overwrite=true \
python ../../../evidence_dynamicShape/static_shape_probe.py "$CASE" > /tmp/e2/run_static.log 2>&1
echo "exit=$?"

grep -c 'extern "C" {' /tmp/e2/src_static.cu   # group 数
grep -c '__global__'   /tmp/e2/src_static.cu   # kernel 数
grep -c 'COND__F'      /tmp/e2/src_static.cu   # 谓词分桶数
grep -cE 'int(32|64)_t S[0-9]' /tmp/e2/src_static.cu   # 符号形参数
```

**实测结果**（`static_shape_control.log`）

| 指标 | 动态 spec（含 -1） | 静态 spec | 说明 |
|---|---|---|---|
| `extern "C"` group 数 | 2 | 2 | 图结构相同，融合结果相同 |
| `__global__` kernel 数 | 8 | **2** | 动态 = 每 group 4 桶；静态 = 每 group 1 个 |
| `COND__` 谓词桶数 | 8 | **0** | 静态下谓词退化为 `COND_true__` |
| 符号形参 `S0/S1/S2` | 8 | **0** | 静态下尺寸已内联为常量 |
| group1 实测 launch | 18×1×1 / block 1 | 1×1×1 / **block 32** | tile 由确切 numel 定 |
| group2 实测 launch | 137×1×1 / block 1024 | 8×72×1 / **block 256** | 静态用 2D grid，完全不同的 tiling |

静态编译选出的 block（32 / 256）**不等于**动态四桶里的任何一个（1 / 1024）。这说明 tile
配置是 numel 的函数：尺寸已知就按确切值定一份，尺寸未知就按 numel 区间枚举多份、运行期选。

**判定**：`COND__` 谓词桶数与符号形参数在静态下均为 0，且 kernel 数从 8 降到 2 —— 即
T5/T8.1 观察到的分桶结构确由"动态 shape"引入，不是 CINN 对任意图的固有行为。

**留存**：`static_shape_control.log`、`static_shape_probe.py`。

### T8.3 调优前后性能对比

对比对象是**同一个 case、同一个运行 shape `[1,72,88,88]`** 下的两种 schedule：
"尺寸已知按确切 numel 定 tile"（static，视为调优上界）与"尺寸未知按 numel 区间分桶"
（dyn，实际动态路径）。差值即为形状泛化所付的代价。

**操作**

```bash
cd /work/PaddleTest/framework/e2e/PaddleLT_new
CASE=layercase/sublayer1000/Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py
mkdir -p /tmp/e4

for MODE in dyn static; do
  nsys profile -t cuda -o /tmp/e4/rep_$MODE --force-overwrite=true \
  python ../../../evidence_dynamicShape/tune_perf_probe.py "$CASE" $MODE > /tmp/e4/run_$MODE.log 2>&1
  echo "$MODE exit=$?"
done

# 按 kernel 聚合均值（tune_perf_probe.py 固定 5 warmup + 50 iters，故每 kernel n=55）
for MODE in static dyn; do
  nsys stats --report cuda_gpu_trace --format csv /tmp/e4/rep_$MODE.nsys-rep 2>/dev/null \
    | grep fn_reshape | awk -F',' '{print $4"x"$5"x"$6, $7, $2}'
done
```

**实测结果**（A100/SM80，n=55/kernel，`tune_perf.log`）

| group | 元素数 | static（确切尺寸） | dyn（分桶命中） | dyn / static |
|---|---|---|---|---|
| group1 | 18 | grid 1×1×1 / block 32，**2143 ns** | grid 18×1×1 / block 1，**2019 ns** | 0.94×（均在 kernel 启动开销量级，无实质差异） |
| group2 | 557568 | grid 8×72×1 / block 256，**3951 ns** | grid 137×1×1 / block 1024，**7259 ns** | **1.84×** |
| 合计 | — | 6094 ns | 9278 ns | 1.52× |

min/max 离散度均在 ±5% 内（如 group2 dyn 7200~7360 ns），差异远大于噪声。

**判定**：本步**不设通过/不通过门限**，只作为代价量化如实记录。可得的结论是：
分桶 schedule 确实随形状变化（T8.1），但默认 `policy=default` 的规则化分桶在大 numel
group 上比确切尺寸调优慢 1.84×，这个差距正是 `optimal`/`hybrid` 策略（T8.0）要填的。
若验收要求"调优后优于调优前"，需另跑 `FLAGS_tile_config_policy=search` 生成
`FileTileConfigDatabase` 再以 `optimal` 复测 —— 注意空库会因
`tile_config_data_.count(policy_)==0` 静默退回 `default`，且 search 分支每 kernel
重放 25 次，耗时显著。本轮未做。

**留存**：`tune_perf.log`、`tune_perf_probe.py`。




---

## T7 归档

把本轮产物统一放入 `/work/PaddleTest/evidence_dynamicShape/`：

| 文件 | 来自 | 用途 |
|------|------|------|
| `dynamic_shape_cinn.log` | T3 | 70/70 PASS、exit 0 |
| `ir_symbolic.log` | T4 | `ir_probe.py` 全量输出（两段 `[ShapeDialect]` 的出处） |
| `shape_dialect_origin.txt` / `shape_dialect_after_pass.txt` | T4 | 符号化 PIR |
| `cinn_source.cu` | T5 | 符号形参 + `COND__` 分桶 kernel |
| `cache_on_run.log` / `cache_on_src.cu` | T6.1 | 缓存开：codegen=2 |
| `cache_off_run.log` / `cache_off_src.cu` | T6.1 | 缓存关：codegen 仍为 2 |
| `two_case_cache_on_run.log` / `..._src.cu` | T6.2 | codegen=2（命中缓存） |
| `two_case_cache_off_run.log` / `..._src.cu` | T6.2 | codegen=4（缓存生效） |
| `eager_baseline.log` | T6.3 | 纯 eager 复现 S1/S5 慢（排除 CINN） |
| `perop.log` | T6.3 | 逐 op 定位到两个 conv2d；否证显存增长 |
| `conv_shape_cost.log` | T6.3 | 干净进程裸 conv：每个首见 N 付一次几十毫秒 |
| `ir_probe.py` / `cache_probe.py` / `two_case_probe.py` | 驱动脚本 | 复现用 |
| `eager_baseline_probe.py` / `perop_probe.py` / `conv_shape_cost_probe.py` | 驱动脚本 | T6.3 复现用 |
| `glog_multilink_evidence.log` | T4 附带 | glog 已知缺陷的取证记录（与本指标无关，见 T4 末尾） |
| `glog_multilink_probe.py` / `glog_sso_matrix.sh` | 驱动脚本 | 上述缺陷的复现用 |
| `bucket_dispatch.log` | T8.1 | 三个 shape 的运行期分桶命中表（含 md5 一致性） |
| `bucket_dispatch_probe.py` | 驱动脚本 | T8.1 复现用（单进程单 shape，配合 nsys） |
| `static_shape_control.log` | T8.2 | 静态 spec 负对照：8 kernel → 2 kernel、`COND__`=0 |
| `tune_perf.log` | T8.3 | static vs dyn 的 kernel 实测耗时（n=55） |
| `static_shape_probe.py` / `tune_perf_probe.py` | 驱动脚本 | T8.2 / T8.3 复现用 |
| `three_c_probe.py` / `three_c_run.log` / `three_c_src.cu` | T6.2 附加 | 三个不同 C 的 picodet 同进程对照：codegen=6、kernel=24，正面复核"不同 C 不复用编译" |
| 图 9.3.3-1 ~ -5 | T1/T3/T4/T5 | 截图 |

### 目录组织与文件依赖

`evidence_dynamicShape/` **平铺存放，不分子目录**。里面的脚本彼此独立，没有任何跨文件
import，各自服务上表中对应的一个步骤；`.log` / `.cu` / `.txt` 都是产物，复现时不需要读取。

因此**只跑主用例与符号化 PIR 这两步时，最小可运行集是 12 个文件**：

| 文件 | 说明 |
|------|------|
| `framework/e2e/PaddleLT_new/test_dynamic_shape_cinn.py` | T3 主驱动。`SUBLAYER_DIR` 按**自身文件位置**推出 `./layercase/sublayer1000/`，故**必须留在 `PaddleLT_new/` 下**才能找到用例 |
| `evidence_dynamicShape/ir_probe.py` | T4/T5 驱动。用例路径由 `argv[1]` 传入，脚本自身位置无所谓 |
| `layercase/sublayer1000/Det_cases/…` 下 10 个 SIR 文件 | 用例本体（仓库原有，清单见测试计划 §8） |

其余依赖只有运行环境：`paddle`（本机 build，需 CINN + CUDA）与 `numpy`。

若还要跑 T8，再加两个脚本 `evidence_dynamicShape/static_shape_probe.py`、
`evidence_dynamicShape/tune_perf_probe.py`（同样只靠 `argv` 传用例路径，位置无所谓），
以及一个外部工具 **`nsys`**（Nsight Systems，用于 `cuda_gpu_trace` 读实际启动的 kernel 名
与 grid/block）。T8.1 还需把 `ir_probe.py` 的 `SHAPES` 临时改为单一 shape 分三次跑。

`three_c_probe.py` 不接参数，三个用例路径按**自身位置**推导
（须保持 `evidence_dynamicShape/` 原位，其上级须是 PaddleTest 仓库根）；
`FLAGS_cinn_source_code_save_path` 由调用方在 `import paddle` 之前 export。

两点实现细节，改动这些文件时需留意：

- 两个驱动都用 `importlib.util.spec_from_file_location` 按绝对路径加载用例，**不走包导入**，
  所以 `layercase/` 各级目录的 `__init__.py` 是否存在都不影响这两步。
- `ir_probe.py` 的 `SHAPES` 硬编码为 `[1,72,88,88] / [2,72,88,88] / [1,72,44,44]`，
  **只适用于 C=72 的用例**（`picodet_l_640_coco/SIR_17.py`、
  `ttfnet_pafnet_lite_mobilenet_v3_20x_coco/SIR_22.py`、
  `centernet_centernet_mbv3_large_140e_coco/SIR_22.py`）。换成其他 C 的用例需同步改 `SHAPES`，
  否则 conv 权重与输入通道数不匹配而报错。

---

## 3 合格判据（逐条打勾）

| 编号 | 判据 | 判定方式 | 来源 |
|------|------|---------|------|
| 1 | 功能正确性 | 每 case 7 组 shape 全 `PASS`，合计 70/70，无 FAIL/SKIP | T3 |
| 2 | 数值精度 | `assert_allclose(atol=1e-5, rtol=1e-5)` 逐张量通过 | T3 |
| 3 | 动态场景覆盖 | 同时覆盖 batch 单变(S1,S5)、空间维缩放(S2,S3)、多维同变(S4) | T3 |
| 4-a | 动态维符号化 | PIR 中 `pd_op.data` 标为 `shape[S0, 72, S1, S2]` | T4 |
| 4-b | 符号进入 kernel | kernel 形参含 `int32_t S0, S1, S2`，无尺寸常量 | T5 |
| 4-c | 按形状分桶调优 | 同一 group 多个 `COND__` kernel，谓词为符号区间 | T5 |
| 4-d | 一次编译服务全部 shape | codegen 只在首个 shape 触发（T3 的 `codegen` 列），且事件数 = group 数、关缓存后不变 | T3 / T6 |
| 4-e | 运行期确实按形状选桶 | 跨谓词分界线的三个 shape 命中不同 `COND__` kernel，且 grid/block/寄存器数实测不同；三次编译产物 md5 相同 | T8.1 |
| 4-f | 分桶为动态 shape 特有 | 静态 spec 下 kernel 数 8→2、`COND__` 谓词桶 8→0、符号形参 8→0 | T8.2 |
| 5 | 自动化判定 | 主脚本退出码为 `0` | T3 |

十条全部满足即判定技术指标 3 合格。判据 1~3、5 由脚本自动判定，判据 4-a~4-f 由采集产物人工核验。
T8.3（调优代价量化）**不作为通过门限**，作为附带数据如实记录。

### 明确不作为判据的项

| 项 | 为什么不作为判据 |
|----|----------------|
| 单个 case 的 `codegen` 合计绝对值 | 受编译缓存影响：同拓扑同通道数的后来 case 会命中缓存而计 0（本轮 ttfnet=1、centernet=0）。可作为判据的是**分布形态**（只在首个 shape 为正）与 T6 的开/关缓存对照 |
| 旧版日志里的 `compiles` 列 | 已从脚本移除。它基于 `Compiling subgraph with CINN backend` 这条 `LOG_FIRST_N(INFO, 1)` 日志（`add_cinn_pass.cc:334`），进程内最多打印一次，导致第 2 个 case 起恒为 0。历史日志中该列的 0 值不代表未编译 |
| `--static-tuning` / `--static-only` 探针结果 | 这两个探针对每个 shape 各自 `to_static` 一次，测的是"分别编译"而非动态 spec 下的行为，属额外观测。**正式验收不带这两个参数** |
| 各 shape 首次前向耗时高低 | 混合了编译成本与运行期首次成本，不能据此推断重调优 |

### 关于"根据不同张量形状自动调优"如何算通过

调优决策落在**编译期**，选择落在**运行期**，验收方式为：

| 评审要求的环节 | 对应判据 / 步骤 |
|---|---|
| 触发方式 | T8.0（`FLAGS_tile_config_policy` → `schedule_config_manager.cc` 的三条分支） |
| 候选配置 | 判据 4-c（T5 的 4 桶表：numel 区间 × 索引位宽的叉积） |
| 选择结果 | 判据 4-e（T8.1 三 shape 命中不同桶，grid/block/reg 实测不同） |
| 调优前后性能对比 | T8.3（static 6094 ns vs dyn 9278 ns，group2 差 1.84×） |
| 分桶归因于动态 shape | 判据 4-f（T8.2 静态负对照） |

配合判据 1（一次编译的符号 kernel 能正确服务全部动态 shape）与判据 4-d（分桶 kernel 在
首次编译时一并生成，codegen 数与 shape 数无关）。
**不要求**运行期观测到 per-shape 重编译事件 —— 符号化编译下本就不应发生。

**口径声明（必须写进报告）**：本指标下的"自动调优"取"按形状区间分桶的规则化 schedule
特化"之义。搜索式 autotuner 需显式设 `FLAGS_tile_config_policy=optimal|hybrid|search`
才启用，默认关闭；`FLAGS_enable_auto_tuner` 是无调用点的死 flag。详见 T8.0。

### 已知局限（在报告中如实声明）

1. SE 子图 reduce 到 1×1，观察到的分桶维度是"总元素数区间 + 索引位宽"，
   而非 reduce 维相关的 tile 差异。若需展现更丰富的按形状调优差异，
   应补充含**动态 reduce 维**的子图（softmax / layernorm）。
2. 本轮仅做 CINN vs eager 两方比对，未纳入 dy2st(`backend=None`) 作为第三方基线，
   属可选补强项，不影响上述判据成立。
3. T5 的 4 桶候选集是**机械叉积**，其中"numel ≤ 1023 且索引 > INT32_MAX"一桶在逻辑上
   不可达（两条件互斥）。这反映候选由规则生成、未做可达性剪枝，不影响判据成立，但说明
   默认策略不是搜索式调优。
4. T8.3 的 static 一侧是"确切尺寸下的规则化 tile"，作为调优上界的**近似**；它不等于
   `policy=search` 实测搜索出的最优配置。真正的"调优前 vs 调优后"需按 T8.3 末尾的方法
   另跑，本轮未做。

## 4 结果记录表模板

| 步骤 | 执行时间 | 关键输出 | 结论 |
|------|---------|---------|------|
| T1 | | version / cinn / cuda = | 通过 / 不通过 |
| T2 | | inspec = | 通过 / 不通过 |
| T3 | | PASS 总数 = ___/70，exit = ___ | 通过 / 不通过 |
| T4 | | `pd_op.data` 标注 = | 通过 / 不通过 |
| T5 | | 符号形参 = ，`COND__` kernel 数 = | 通过 / 不通过 |
| T6.1 | | codegen on/off = ___/___ | 通过 / 不通过 |
| T6.2 | | codegen on/off = ___/___ | 通过 / 不通过 |
| T8.1 | | 三 shape 命中桶 = ___/___/___，src md5 一致 = 是/否 | 通过 / 不通过 |
| T8.2 | | 静态 kernel 数 = ___，`COND__` = ___，符号形参 = ___ | 通过 / 不通过 |
| T8.3 | | static ___ ns vs dyn ___ ns，比值 ___× | 记录（不设门限） |
| 总体 | | | **合格 / 不合格** |








