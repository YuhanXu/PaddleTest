# CINN 可变形状输入张量自动调优 —— 测试安排与计划

## 1. 背景与目标

- 技术指标 3：**支持可变形状（dynamic shape）输入张量，支持根据不同张量形状自动调优。**
- 结论预期：Paddle 3.0 的 CINN 已支持该能力，但当前缺少一条"同一模型 + 多种输入 shape + 自动调优"的直接对应用例。
- 本计划目标：设计并落地一组能够**直接证明**该指标的测试，并明确验收标准。

## 2. 关键结论：两个 SIR_17.py 是什么关系？

对比 `picodet_legacy_model_picodet_m_320_coco/SIR_17.py` 与
`picodet_legacy_model_picodet_l_640_coco/SIR_17.py`：

| 维度 | picodet_m_320 | picodet_l_640 |
|------|---------------|---------------|
| 算子拓扑 | adaptive_avg_pool2d→conv1x1→relu→conv1x1→hardsigmoid→multiply | 完全相同 |
| 语义 | SE(Squeeze-Excitation)通道注意力子图 | 完全相同 |
| 通道数 C | 56 | 72 |
| 参数 shape | [14],[56,14,1,1],[14,56,1,1],[56] | [18],[72,18,1,1],[18,72,1,1],[72] |
| 输入 tensor | [1, **56**, 48, 48] | [1, **72**, 88, 88] |
| InputSpec | (-1, **56**, -1, -1) | (-1, **72**, -1, -1) |

**判断：这两个文件是"相同算子结构、不同模型实例"，而不是"同一个模型、仅输入 shape 不同"。**

理由：
1. 通道数不同（56 vs 72）导致**权重参数不同**，是两个独立的网络实例，不是同一份权重喂不同 shape。
2. 两个文件里 InputSpec 的**通道维 C 都是固定值**（56 / 72），只有 batch 和 H/W 是 `-1`。
   即：单个文件内部确实声明了动态维（N、H、W），但跨文件的差异主要是模型差异，不是纯 shape 差异。

因此：**这两个文件本身不能直接、干净地证明"同一模型对不同输入 shape 自动调优"**，
它们更适合作为"不同模型/不同 C 各自具备动态 N/H/W 能力"的旁证。要直接证明指标 3，
需要构造"固定同一份权重 + 遍历多组具体 shape（匹配 `-1` 动态维）"的测试（见第 4 节）。

## 3. 框架现状（可复用能力）

`generator/builder_data.py` 已提供多种 spec 生成方式：
- `get_single_input_and_spec()`：优先取用例内 `create_inputspec()`（含 `-1` 动态维）。
- `get_single_input_and_static_spec()`：由具体数据推导**全静态** spec。
- `get_single_input_and_dynamic_spec()`：把所有维置为 `-1`（全动态）。
- `get_single_input_and_multi_spec()`：`SpecStrategy` 生成**多组** spec 组合，用于遍历搜索。

`engine/paddle_eval.py` 已有 dy2st + CINN 的动态/静态 InputSpec 评估分支
（`dy2st_eval_cinn_inputspec` / `dy2st_eval_cinn_static_inputspec`）。

## 4. 正确的测试设计（核心）

以**单个已固定权重的模型**（选 `picodet_l_640/SIR_17.py`，InputSpec `(-1,72,-1,-1)`）为被测对象，
用 `paddle.jit.to_static(net, backend="CINN", full_graph=True, input_spec=<动态spec>)` 编译一次，
然后**依次喂入多组匹配动态维的具体 shape**，每组都：
1. 与动态图（eager）输出做数值一致性校验（`assert_allclose`）；
2. 用中间 IR dump 确认动态维在 CINN 内部以符号（`S0`/`S1`…）表示、并以运行期整型形参进入 kernel——
   这是"一份编译产物服务多种 shape"的机制性证据（详见 §9.2 与 9.3.3 步骤 7）。
   注意：早期方案用 glog 标记计数（compiles）判定编译次数，该方法**已被证伪**，
   因为对应日志是 `LOG_FIRST_N(INFO, 1)`，进程内最多打印一次（见 §9.2）。

被测的动态维：N（batch）、H、W；固定维：C=72。

### 4.1 测试用例矩阵（shape 遍历表，与脚本 VARIANTS 一致）

固定维 C，动态维 N/H/W 按 batch/spatial 因子缩放：

| 组别 | 因子(batch,spatial) | 以 picodet_l_640 为例 | 覆盖点 |
|------|--------------------|----------------------|--------|
| S0 基准 | (1, 1) | [1, 72, 88, 88] | 与源文件一致，首次编译 |
| S1 变 batch | (2, 1) | [2, 72, 88, 88] | N 变化 |
| S2 变 H/W(缩小) | (1, 0.5) | [1, 72, 44, 44] | 空间尺寸缩小 |
| S3 变 H/W(放大) | (1, 1.5) | [1, 72, 132, 132] | 空间尺寸放大 |
| S4 batch+空间同变 | (2, 0.5) | [2, 72, 44, 44] | 多维同时变化 |
| S5 大 batch | (4, 1) | [4, 72, 88, 88] | N 进一步变化 |
| S6 重复 S1 | (2, 1) | [2, 72, 88, 88] | 对照组：与 S1 同 shape，用于检查是否有额外编译 |

### 4.2 对比基线

- **动态图 eager**：作为数值 golden。
- **dy2st（backend=None）静态图**：作为第二参照，排除 dy2st 本身问题（**本轮未纳入，可选补充**）。
- **dy2st + CINN 动态 InputSpec**：被测目标。
- 三者两两 `assert_allclose(atol=1e-5, rtol=1e-5)`（float32；SE 子图含 conv，放宽于源文件的 1e-8）。

## 5. 落地步骤

1. 新增/复用测试脚本（放在 `PaddleLT_new` 下，或独立 `test_dynamic_shape_cinn.py`）：
   - 构建 `picodet_l_640/SIR_17.py` 的 `LayerCase()`（权重 `paddle.seed` 固定）。
   - 用 `create_inputspec()`（`(-1,72,-1,-1)`）作为编译期 spec，`backend="CINN"`。
   - 循环 4.1 矩阵中的每个 shape：`paddle.rand(shape)` → 前向 → 与 eager 对比。
2. 采集编译期证据：
   - **主证据（中间 IR）**：`GLOG_vmodule='shape_o*=3'`（短 glob，长写法会触发 glog
     退出期挂死，见 §9.4(d) 第 3 项）打印带符号标注的 PIR
     （`ShapeOptimizationPass` 的 `PrintHook`），并用 `FLAGS_cinn_source_code_save_path` 导出
     生成的 CUDA 源码，确认动态维为符号形参而非固化常量（**不要用 `FLAGS_cinn_dump_group_*`，
     本 build 中是死 flag，见 §9.4(d)**）。
   - **辅助证据（编译事件）**：脚本设置 `GLOG_logbufsecs=0` 并在 fd 级捕获 stderr，统计标记
     `Compiling subgraph with CINN backend` 出现次数（compiles）。该计数**只能说明 CINN 后端被激活**，
     不能用于比较编译次数（原因见 §9.2）。
3. 汇总：形成"shape → 是否通过数值校验 → IR 中动态维是否符号化"的结果表。

## 6. 验收标准（证明指标 3 成立）

- [x] 同一份权重、同一次 CINN 编译入口，能对 S0~S8 全部不同 shape 正确前向且数值与 eager 一致
      （实测 10 case × 9 shape = **90/90 PASS**，见 §9.1）。
- [x] 覆盖 batch 维、空间维、以及多维同时变化三类动态场景（S1/S5、S2/S3、S4），
      并含两个跨 1023/1024 桶边界的变体（S7 翻入小桶、S8 翻入大桶，评审补测 2026-09-07）。
- [x] 中间 IR 证据：动态维在 PIR 中标注为符号 `S0`/`S1`/`S2`，且 CINN 生成的 CUDA 源码中以
      运行期整型形参 `int32_t S0, S1, S2` 进入 kernel —— **已于 2026-08-25 在 A100 上实测采集**，
      产物见 `evidence_dynamicShape/`（详见 §9.4）。
- [x] 编译产物按形状区间分桶（同一 group 多个 `COND__` 谓词 kernel，tile 策略不同）——
      这是"按形状自动调优"的直接产物证据，**已实测**，见 §9.4(c)。
- [ ] ~~有可观测的编译事件证据（compiles 计数）~~ —— **该判据失效**：对应日志为 `LOG_FIRST_N(INFO, 1)`，
      进程内最多打印一次，无法反映真实编译次数（见 §9.2）。
- [ ] ~~"首见 shape 触发编译/调优、再见 shape 复用缓存"~~ —— **该判据不适用**：符号化 kernel 在运行期
      不做 per-shape 重编译，S1 vs S6 无差异可比；且原有观测手段本身无判别力（见 §9.2）。
- [ ] CINN vs dy2st(backend=None) vs eager 三方一致性 —— 本轮脚本只做了 **CINN vs eager** 两方比对，
      dy2st 基线未纳入，属可选补充项。

## 7. 备注与风险

- 环境提醒：当前开发机为 A100(SM80)，需确认本机 Paddle 已编译 CINN（`paddle.is_compiled_with_cinn()`）且 GPU 可用；否则本计划仅能静态审阅、无法实跑验证。
- **关于"自动调优"的关键澄清（见 §9.4/§10）**：SE 子图这类结构下，CINN 采用**符号化动态 shape**编译，
  动态维以符号变量进入 kernel；一次编译产出按形状区间分桶的多份 kernel，运行期按符号谓词选桶，
  不做 per-shape 重编译。这一点已由**中间 IR + 生成的 CUDA 源码**实测证明（§9.4），而非日志计数。
- **关于早期 62ms 耗时推断**：早期"S1/S5≈62ms 是按 batch 重调优、S6 是命中缓存"的说法不可采信——
  wall-clock 耗时无法区分"重编译"与"新具体 shape 首次执行的运行时开销（显存分配 / launch 配置 /
  cudnn algo 选择）"；后续用 compiles=0 否证它的那套证据同样无效（`LOG_FIRST_N` 问题）。
  §9.5 用 codegen 计数证明了不存在重编译，§9.7 进一步查明该开销的真实来源是
  **conv2d 的 cuDNN 按 conv 问题 shape 选算法**（三条列举里的第三条），
  与 CINN 无关：纯 eager 同样复现。至于"batch 变大让谓词选中另一个已编译的桶"这个中间猜想，
  已由 §9.7 排除（本子图的桶边界在 `S0*K` 上，batch 1→2→4 并不跨界；且慢的是 conv 而非 CINN kernel）。
- `picodet_m_320` 与 `picodet_l_640` 两个文件可作为"不同 C 通道各自支持动态 N/H/W"的横向旁证，但**不作为指标 3 的主证据**。

## 8. 已落地：可执行脚本与 10 个用例

脚本：`framework/e2e/PaddleLT_new/test_dynamic_shape_cinn.py`

运行：
```
cd framework/e2e/PaddleLT_new
python test_dynamic_shape_cinn.py                 # 跑内置 10 个 case（动态 spec，快）
python test_dynamic_shape_cinn.py <case.py>       # 跑指定单个 layercase
python test_dynamic_shape_cinn.py --static-tuning # 额外做"每个 shape 各自静态编译"探针
python test_dynamic_shape_cinn.py --static-only <case.py>  # 干净进程只跑静态编译探针
```

脚本对每个 case：eager 与 CINN 两个实例共享同一 `state_dict` → 用带 `-1` 的动态 InputSpec 对 CINN
只 `to_static(backend="CINN")` 编译一次 → 依次喂入 9 组 shape（S0~S8，其中 S7/S8 为跨桶边界变体）
→ 与 eager 逐一 `assert_allclose(atol=1e-5)`。脚本同时统计每个 shape 前后 `extern "C" {` 块数的
差分作为 codegen 事件数（§9.6）；早期的 glog compiles 计数已删除（不能作为编译次数证据，见 §9.2）。

脚本已按评审意见修正三处（提升作为"验证脚本"的可信度）：
1. **退出码收紧**：每个 case 必须 S0~S8 **全 PASS**（无 FAIL、无 SKIP）才算 OK，否则 `exit 1`；与 §6 一致。
2. **数值失败与非法 shape 分离**：`assert_allclose` 不一致判 `FAIL:numeric`（真 bug，绝不降级）；
   仅"构造输入/eager 前向即失败"才判 `SKIP:shape`；CINN 前向异常判 `FAIL:cinn`。
3. **编译计数取代耗时推断**：`GLOG_logbufsecs=0` + fd 级捕获 stderr，统计
   `Compiling subgraph with CINN backend` 出现次数。
   **该项已被证伪、不可用于比较编译次数**：该日志语句为
   `LOG_FIRST_N(INFO, 1)`（`paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.cc:334`），
   静态计数器使其在进程内最多打印一次。计数保留价值仅在于"确认 CINN 后端被激活"。
   编译行为的正确观测方式改为 IR dump（见 §9.2）。

选定的 10 个用例（均为单输入 float32、InputSpec=`(-1, C, -1, -1)`、SE 风格子图，来自 8 个检测模型）：

| # | 模型子图 | 通道 C | 基准 shape |
|---|----------|--------|-----------|
| 1 | picodet_legacy_model_picodet_l_640_coco/SIR_17.py | 72 | [1,72,88,88] |
| 2 | picodet_legacy_model_picodet_m_320_coco/SIR_17.py | 56 | [1,56,48,48] |
| 3 | picodet_legacy_model_picodet_s_320_coco/SIR_17.py | 44 | [1,44,32,32] |
| 4 | ttfnet_pafnet_lite_mobilenet_v3_20x_coco/SIR_22.py | 72 | [1,72,44,44] |
| 5 | ssd_ssdlite_mobilenet_v3_large_320_coco/SIR_31.py | 120 | [1,120,40,40] |
| 6 | yolov3_yolov3_mobilenet_v3_large_ssld_270e_voc/SIR_64.py | 480 | [1,480,38,38] |
| 7 | centernet_centernet_mbv3_large_140e_coco/SIR_22.py | 72 | [1,72,64,64] |
| 8 | ppyolo_ppyolo_tiny_650e_coco/SIR_31.py | 64 | [1,64,56,56] |
| 9 | ppyolo_ppyolo_mbv3_small_coco/SIR_49.py | 240 | [1,240,20,20] |
| 10 | ppyolo_ppyolo_mbv3_large_coco/SIR_77.py | 960 | [1,960,10,10] |

每个用例统一遍历以下 shape 变体（在动态维 N/H/W 上缩放，固定维 C 不变）：
S0 基准 / S1 batchx2 / S2 spatial÷2 / S3 spatialx1.5 / S4 batchx2+spatial÷2 / S5 batchx4 / S6 重复 S1
（S6 原设计用于对比缓存复用，因观测手段失效而降为纯重复用例）。

**环境说明（已跑通）**：需要用如下环境变量（关键：把 `/usr/lib64` 加入 `LD_LIBRARY_PATH`，
并激活 `/work/env3.10` venv）才能 `import paddle`；此前的 `libcuda.so.1` 报错是环境变量未配置所致，
并非缺驱动（`/usr/lib64/libcuda.so.1` 存在，driver 535.230.02，nvidia-smi 正常）：
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
export PYTHONPATH=/work/Paddle/build/python
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source /work/env3.10/bin/activate
```
本机 `paddle 3.5.0.dev20260526`，`is_compiled_with_cinn()=True`、`is_compiled_with_cuda()=True`（GPU SM80/A100）。

## 9. 实跑结果（2026-08，A100）——数值结论成立，编译计数方法作废（第二次更正稿）

10 个用例全部通过数值校验，**总计 90/90 shape**（`assert_allclose(atol=rtol=1e-5)`，CINN vs eager 逐一一致；
原 7 组 S0~S6 为 70/70，2026-09-07 评审补测追加 S7/S8 两个跨桶边界变体后为 90/90）。
这一部分结论稳定可复现。但本轮曾用作"编译/调优证据"的 **compiles 计数已被证伪**，
相关推论一并撤回，见 §9.2。

### 9.1 数值正确性（全部 PASS）

| # | 模型子图 | 结果 (S0~S8) |
|---|----------|--------------|
| 1 | picodet_l_640/SIR_17 (C=72) | 9/9 PASS |
| 2 | picodet_m_320/SIR_17 (C=56) | 9/9 PASS |
| 3 | picodet_s_320/SIR_17 (C=44) | 9/9 PASS |
| 4 | ttfnet_pafnet_lite/SIR_22 (C=72) | 9/9 PASS |
| 5 | ssdlite_mbv3_large/SIR_31 (C=120) | 9/9 PASS |
| 6 | yolov3_mbv3_ssld/SIR_64 (C=480) | 9/9 PASS |
| 7 | centernet_mbv3_large/SIR_22 (C=72) | 9/9 PASS |
| 8 | ppyolo_tiny/SIR_31 (C=64) | 9/9 PASS |
| 9 | ppyolo_mbv3_small/SIR_49 (C=240) | 9/9 PASS |
| 10 | ppyolo_mbv3_large/SIR_77 (C=960) | 9/9 PASS |

最终判定：全部通过（exit 0）。

### 9.2 编译事件计数方法作废（关键更正）

用 `_CaptureStderrFd` 在 fd 级捕获 glog，按 shape 统计 `Compiling subgraph with CINN backend`，
观测到的原始数据是：

- 全程只有 case1 / S0 记到 compiles=1；
- 其余所有 variant（S1~S6）compiles=0；
- case2~case10 的全部 shape 也都 compiles=0；
- `--static-only` 干净进程探针：第 1 个具体 shape compiles=1，第 2、3 个不同具体 shape 仍 compiles=0。

**这组数据不具备解释力。** 该日志语句是

```cpp
// paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.cc:334
LOG_FIRST_N(INFO, 1) << "Compiling subgraph with CINN backend ...";
```

`LOG_FIRST_N(severity, 1)` 由站点静态计数器控制，**整个进程生命周期内最多输出一次**，
与实际编译了多少个子图无关。因此"首个 shape 为 1、其余全为 0"是日志宏的必然结果，
无论底层编译多少次都会得到同样的数字。

据此撤回以下三条曾写入本文的推论：

1. ~~"其余 shape compiles=0 ⇒ 只编译了一次"~~ —— 无效推理。
2. ~~"case2~case10 通道数 C 不同却仍 compiles=0 ⇒ 复用同一次编译"~~ —— 不仅无效，且方向上很可能是错的：
   不同 C 对应不同权重与不同算子形状，属结构不同的子图，不应复用同一份 kernel。
3. ~~"compiles=0 直接证伪了 62ms 是重调优"~~ —— 证伪手段本身无效，该问题回到未结论状态。

**正确的观测方式**（已实测走通，产物见 §9.4）：

- 打印中间 IR 验证 shape 的符号化表示：`GLOG_vmodule='shape_o*=3'`（短 glob，见
  §9.4(d) 第 3 项）触发 `ShapeOptimizationPass` 的 `PrintProgram` + `PrintHook`
  （`paddle/pir/src/dialect/shape/transforms/shape_optimization_pass.cc:31,54-64`），
  符号名 `S0`/`S1`… 由 `GetNextSymName()` 生成（`paddle/pir/src/dialect/shape/utils/shape_analysis.cc:113-114`）。
- 导出生成的 CUDA 源码验证 kernel 是否形状泛化：`FLAGS_cinn_source_code_save_path`
  （`SourceCodePrint`，`paddle/cinn/backends/compiler.cc:220-253`）。
  `FLAGS_cinn_dump_group_lowered_func` / `_source_code` / `_ptx` **不可用**——
  消费函数在 `compiler.cc:104-190`，但无任何调用点（§9.4(d)）。
- 观测编译次数用 `FLAGS_enable_cinn_compile_cache=false` 关缓存后比较产物增量
  （`paddle/common/flags.cc:1266-1273`，缓存命中判定 `pir_compiler.cc:451-484`），不要再用日志计数。

### 9.3 对指标 3 的诚实拆解

- "**支持可变形状输入张量**"：**已证明**。同一份权重、单次动态编译入口，可正确前向 batch/空间/多维同变的
  所有 shape，数值与 eager 一致（90/90，含跨桶边界 S7/S8）。该结论不依赖已作废的编译计数。
- "**根据不同张量形状自动调优**"：**已由编译产物实测证明**（见 §9.4）。
  CINN 的形状自适应调优由编译期 group_schedule 按形状区间分桶（bucket）+ 逐桶 tile 完成，
  一个 group 产出多份调度策略不同的符号 kernel，运行期按符号谓词选桶、不重编译。

### 9.4 中间 IR 与编译产物证据（2026-08-25 实测补齐）

采集用最小驱动 `evidence_dynamicShape/ir_probe.py`（不做 fd 重定向，否则 glog 被脚本的
`_CaptureStderrFd` 吞掉），被测 case 为 `picodet_l_640/SIR_17.py`。

**(a) PIR 层：动态维是符号。** `GLOG_vmodule` 只放开 `shape_optimization_pass`
（不要用全局 `GLOG_v=3`；实际采集用通配短写法 `'shape_o*=3'`——长写法在本 build
会触发退出期挂死，见 (d) 第 3 项），`[ShapeDialect]ShapeOptimizationPass Program` 段中：

```
(%4) = "pd_op.data" () {... shape:[-1,72,-1,-1] ...} : () -> tensor<-1x72x-1x-1xf32>
        { (shape[S0, 72, S1, S2], data[NULL]) }
```

三个 `-1` 分别标注为 `S0`/`S1`/`S2`，固定通道维为常量 `72`。符号沿数据流正确推导：
`pool2d`(adaptive→1×1) 后为 `[S0, 72, 1, 1]`，SE 相乘输出又回到 `[S0, 72, S1, S2]`。

**(b) Kernel 层：符号以运行期整型形参进入 kernel。** 用
`FLAGS_cinn_source_code_save_path=<file>` 导出（**注意 `FLAGS_cinn_dump_group_*` 三个 flag
在本 build 中无调用点，是死代码，设了不产出任何文件**）：

```cuda
__global__ void __launch_bounds__(1) fn_..._kernel(
    const float* __restrict__ var, const float* __restrict__ var_1,
    const float* __restrict__ var_9, float* __restrict__ var_24,
    int32_t S0, int32_t S1, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * S1) * S2) * 72)));
  ...
}
```

全篇无 88/44/132 这类具体尺寸字面常量，只有 `S0`/`S1`/`S2` 与固定维 72。

**(c) 编译产物按形状区间分桶 —— "按形状自动调优"的直接证据。**
同一 group 生成的不是一份 kernel，而是多份，函数名 `COND__` 之后编码了生效谓词
（编码规则见 `paddle/cinn/backends/codegen_device_util.cc:117-174` 的 `PredicatePrinter`：
`_FPA_`=`(`、`_BPA_`=`)`、`MUL`=`*`）。第一个 group 的 4 个 kernel 解码后：

| kernel | 谓词 | `__launch_bounds__` | 调度策略 |
|--------|------|--------------------|---------|
| #1 | `1 <= S0*18 <= 1023` 且 `<= INT32_MAX` | 1 | 一元素一 block，无循环 |
| #2 | `1 <= S0*18 <= 1023` 且 `> INT32_MAX` | 1 | 同上，索引 `int64_t` |
| #3 | `S0*18 >= 1024` 且 `<= INT32_MAX` | 1024 | 1024 线程 × 4 次循环 tile |
| #4 | `S0*18 >= 1024` 且 `> INT32_MAX` | 1024 | 同上，索引 `int64_t` |

代码链路：`ScheduleConfigManager::ExtractConfigs(target_, group_info_)` →
`DynamicShapeGroupScheduler::InitBuckets()` 用 `MakeBucketPredicate()` 为每桶造符号谓词
（`paddle/cinn/ir/group_schedule/dy_shape_group_scheduler.cc:48-83, 198-226`）→
逐桶套 tactics 做 tile（同文件 85-95）→ `GetIRs()` 返回 `vector<pair<SymbolicPredicate, Expr>>`
（`paddle/cinn/hlir/framework/pir/op_lowering_impl.cc:167,193`）→
`predicate2funcs`（同文件 242-245）→ codegen 把谓词写入函数名 + 生成 host 侧 switch
（`codegen_device_util.cc:100-115` 的 `CreateSwitchFunction`）。

**(d) 采集过程中发现的三个 build 问题（建议单独报 issue）：**

1. `FLAGS_logging_pir_py_code_dump_symbolic_dims=1` 导致崩溃：
   `SystemError: (Fatal) The input data pointer is null. (paddle/pir/src/core/storage_manager.cc:85)`，
   抛在 `pir_partial_program.py:879` 的 `apply_cinn_pass`，`original_programs.py` 落盘 0 字节。
   已隔离：只设 `FLAGS_logging_pir_py_code_dir` 正常（5 个文件）；加该开关必崩。
   因此落盘 PIR 里 `__results_symbols_signature__` 全是 `s_null()`，不含符号维信息。
2. `FLAGS_cinn_dump_group_lowered_func` / `_source_code` / `_ptx` 是死 flag：
   消费函数 `CompilationInfoDumper::Dump*ByGroupIndex` 定义在
   `paddle/cinn/backends/compiler.cc:104-190`，但全 `paddle/cinn` 目录下**无任何调用点**。
3. `GLOG_*` **字符串型**环境变量（`vmodule` / `log_dir` / `log_backtrace_at`）值长度
   >= 16 字节时，进程在打完全部输出后的退出析构期触发 glibc 堆校验并挂死。根因：
   `libglog.a` 被静态链进 4 个 .so（`base/libpaddle.so` / `libs/libphi_core.so` /
   `libs/libphi_gpu.so` / `libs/libcinnapi.so`）且 glog 全局符号导出，ELF 符号插入把
   4 份 `fLS::FLAGS_vmodule_buf` 坍缩成 1 个实例，但 4 个静态初始化器各构造一次、
   各注册一次 `__cxa_atexit` ⇒ 退出期同一堆指针被 free 4 次；<= 15 字节走 SSO
   （无堆缓冲）故无害，分界线实测正好在 15/16 字节。与 CINN 无关
   （`import paddle` 一行加长 vmodule 即复现）。**规避**：vmodule 用通配短写法
   （`'shape_o*=3'`），产物与长写法逐字节相同。完整取证：
   `glog_multilink_evidence.log` 及配套 `glog_multilink_probe.py` / `glog_sso_matrix.sh`
   （SSO 分界线矩阵、ctypes 读 4 份 buf、gdb 4 次初始化 hit、ASAN double-free）。

**(e) 补充实测数据，独立佐证 §9.2 的结论。** 本轮 10 case 全量重跑中，S0 的 `cinn_ms` 与
`compiles` 的对照：

| case | S0 shape | compiles | S0 cinn_ms |
|------|----------|---------:|-----------:|
| picodet_l_640/SIR_17 | [1,72,88,88] | 1 | 5284.12 |
| picodet_m_320/SIR_17 | [1,56,48,48] | 0 | 5100.86 |
| picodet_s_320/SIR_17 | [1,44,32,32] | 0 | 4996.55 |
| ttfnet_pafnet_lite/SIR_22 | [1,72,44,44] | 0 | 2917.89 |
| ssdlite_mbv3_large/SIR_31 | [1,120,40,40] | 0 | 5055.53 |
| yolov3_mbv3_ssld/SIR_64 | [1,480,38,38] | 0 | 5064.07 |
| centernet_mbv3_large/SIR_22 | [1,72,64,64] | 0 | 191.90 |
| ppyolo_tiny/SIR_31 | [1,64,56,56] | 0 | 4978.96 |
| ppyolo_mbv3_small/SIR_49 | [1,240,20,20] | 0 | 5158.18 |
| ppyolo_mbv3_large/SIR_77 | [1,960,10,10] | 0 | 5018.00 |

case2~10 各自在 S0 上花了约 5 秒（明显是完整编译），却全部报 `compiles=0`。
这是"计数器坏了"的**直接测量证据**。至于 `centernet` 的 191.90ms 例外，见 §9.5——
已查明是编译缓存命中。

脚本改用 codegen 事件计数后（§9.6），同一张表的 `codegen` 列不再恒为 0：通道数互不相同的
8 个 case 各自记到 2，ttfnet 记 1、centernet 记 0（命中缓存）。这既复核了本条结论，
也把 §9.2 撤回的第 2 条推论正面证否。

### 9.5 编译次数的可靠计数法与两个悬案的结论（2026-08-25 实测）

**计数方法**：`SourceCodePrint` 的 ofstream 在单例构造时以 `trunc` 打开一次，
之后每次 `write()` 追加（`paddle/cinn/backends/compiler.cc:220-253`）。
因此 `FLAGS_cinn_source_code_save_path` 指向的文件中 `extern "C" {` 块的个数
**等于进程内的 codegen 事件数**。这是一个真正可比的计数，替代已作废的 glog 计数。

**实验 1：同一 case，7 个 shape，缓存开/关对照。**

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

**结论：codegen 事件恒为 2（= 该子图的 group 数），与缓存开关无关，与 7 个不同 shape 无关。
即"一次编译服务全部 shape"得到了直接测量证明，不再依赖任何日志宏。**

**实验 2：两个结构相同的 case 放同一进程，缓存开/关对照。**
选 `ttfnet_pafnet_lite/SIR_22`（C=72，基准 [1,72,44,44]）与
`centernet_mbv3_large/SIR_22`（C=72，基准 [1,72,64,64]）：

| | cache=on | cache=off |
|---|---:|---:|
| ttfnet/SIR_22 | 5444.85 ms | 5963.00 ms |
| centernet/SIR_22 | **141.77 ms** | **5698.96 ms** |
| codegen 事件数 | **2** | **4** |
| `__global__` kernel 数 | 8 | 16 |

由此得到三条结论：

1. **§9.1 表中 centernet 的 191.90ms 异常已查明**：是**编译缓存命中**。
   两个 case 拓扑相同、通道数同为 72，动态维符号化后是同一张图，
   FusionInfo 一致 → 复用 ttfnet 已编译的 kernel。
2. **`FLAGS_enable_cinn_compile_cache=false` 确实生效**（codegen 事件 2→4）。
   这反过来确认了实验 1 中"关缓存后仍只有 2 次 codegen"是有效证据，而非 flag 没起作用。
3. 顺带补充了一条符号化的旁证：两个**基准 shape 不同**（44×44 vs 64×64）的模型子图，
   因符号化后结构相同而能复用同一份编译产物。

关于 §9.2 撤回的第 2 条推论（"不同 C 复用同一次编译"），修正后的准确表述是：
**跨 case 复用发生在"拓扑相同 + 通道数相同"时**，不是"不同 C 也复用"。原推论方向仍是错的。

**§7 中"62ms 悬案"至此排除 CINN 重编译**：S1(81ms) / S5(62ms) 均**不是**重编译或重调优——
codegen 事件数在缓存关闭下仍为 2，没有新增产物；S6 重复 S1 只花 0.56ms。
早期"按 batch 重调优"的说法正式否证，这次用的是有判别力的手段。
至于这笔开销**到底花在哪**，本节尚未给出正向归因（当时猜"显存分配增长 / kernel 首次 launch"，
其中显存那条后来被否证）；正向归因见 §9.7：是 **conv2d 的 cuDNN 按 conv 问题 shape 选算法**。

（CINN 编译期调优机制的进一步说明见 §10。）

### 9.6 主脚本已改用 codegen 事件计数（2026-08-31 实测）

§9.5 的计数法已内置进 `test_dynamic_shape_cinn.py`：在 `import paddle` 之前
`os.environ.setdefault("FLAGS_cinn_source_code_save_path", ...)`，每个 shape 前向后读一次
`extern "C" {` 块数并做差分，填入报告的 `codegen` 列；旧的 `compiles` 列与配套的 fd 级
stderr 重定向已删除。全量重跑（`python test_dynamic_shape_cinn.py`）结果：

| case | C | codegen | 说明 |
|------|--:|--------:|------|
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

全进程合计 17，`总计 shape 通过: 90/90`，`exit 0`。两次独立重跑分布一致
（S7/S8 为 2026-09-07 追加后重跑，codegen 合计仍为 17）。

读法：**每个 case 只有 S0 的 `codegen` 为正、S1~S8 恒为 0** ⇒ 一次符号化编译服务全部 shape；
其中 S7/S8 触发桶翻转（运行期谓词改选另一桶）却 codegen=0，说明**翻桶不触发重编译**。
单 case 合计允许为 0 或小于 group 数（ttfnet=1、centernet=0），那是同拓扑同 C 的子图命中
FusionInfo 缓存，因此报告里这一行的措辞是"本 case 新增 codegen 事件: N（未命中缓存的 group 数 /
全部命中编译缓存）"，不能读作"该子图只有 N 个 group"。

本轮产物归档于 `/work/PaddleTest/evidence_dynamicShape/`：
`dynamic_shape_cinn.log`（本节全量日志，90/90）、`shape_dialect_origin.txt` /
`shape_dialect_after_pass.txt`（§9.4 符号化 PIR）、`cinn_source.cu`（§9.4 生成源码）、
`cache_{on,off}_{run.log,src.cu}`（§9.5 实验 1）、
`two_case_cache_{on,off}_{run.log,src.cu}`（§9.5 实验 2）、
`eager_baseline.log` / `perop.log` / `conv_shape_cost.log`（§9.7 三个探针）、
`boundary_shape_acc_probe.py` / `boundary_shape_acc_{small,big}.log`
（跨桶边界 shape 的 eager vs CINN 精度探针，评审补测 2026-09-07，双 PASS）、
`nsys_raw/`（T8.1/T8.3/T9 全部原始 nsys-rep 与 trace CSV，含 e4/t9/si2 等 76 文件，
全部性能数字可由其复算）、
以及驱动脚本 `ir_probe.py` / `cache_probe.py` / `two_case_probe.py` /
`eager_baseline_probe.py` / `perop_probe.py` / `conv_shape_cost_probe.py`。

### 9.7 S1/S5 那几十毫秒的真实来源：cuDNN 按 conv 问题 shape 选算法（2026-08-31 实测）

现象（以 `ppyolo_mbv3_large/SIR_77`，C=960，基准 `[1,960,10,10]` 为例）：
S1(batch×2)=60.54ms、S5(batch×4)=61.19ms，而 S2/S3/S4/S6 都在亚毫秒级。
**只有 batch 首次变大时慢，空间维怎么变都不慢，重复 shape（S6）不慢。**

三个探针依次排除/确认（产物见本节末）：

**(a) 拿掉 CINN，纯 eager 同样复现** ⇒ 与 CINN、与符号化编译、与分桶调优全部无关。
`eager_baseline_probe.py`（不 `to_static`，直接调 `LayerCase`）：

| variant | shape | eager_ms |
|---------|-------|---------:|
| S0 | [1,960,10,10] | 1924.06 |
| **S1** | [2,960,10,10] | **90.93** |
| S2 | [1,960,5,5] | 0.60 |
| S3 | [1,960,15,15] | 0.58 |
| S4 | [2,960,5,5] | 0.43 |
| **S5** | [4,960,10,10] | **60.83** |
| S6 | [2,960,10,10]（重复 S1） | 0.57 |

**(b) 逐 op 计时定位到两个 conv2d，并否证"显存增长"假说。**
`perop_probe.py` 在 op 之间插 `synchronize()` 分别计时，同时打印 `memory_reserved()`：

| variant | shape | pool | conv1 | relu | conv2 | hsig | mul | resvMB | +resv |
|---------|-------|-----:|------:|-----:|------:|-----:|----:|-------:|------:|
| S0 | [1,960,10,10] | 1.21 | 1593.35 | 107.22 | 37.64 | 0.17 | 0.24 | 2.5 | 0.4 |
| S1 | [2,960,10,10] | 0.20 | **48.55** | 0.05 | **34.23** | 0.05 | 0.46 | 4.0 | 1.5 |
| S2 | [1,960,5,5] | 0.11 | 0.30 | 0.04 | 0.17 | 0.03 | 0.04 | 4.0 | 0.0 |
| S3 | [1,960,15,15] | 0.07 | 0.16 | 0.03 | 0.14 | 0.03 | 0.22 | 5.7 | **1.6** |
| S4 | [2,960,5,5] | 0.06 | 0.17 | 0.03 | 0.15 | 0.03 | 0.03 | 5.7 | 0.0 |
| S5 | [4,960,10,10] | 0.07 | **30.93** | 0.03 | **29.90** | 0.04 | 0.29 | 8.6 | 2.9 |
| S6 | [2,960,10,10] | 0.09 | 0.20 | 0.03 | 0.16 | 0.03 | 0.04 | 8.6 | 0.0 |

开销 100% 落在 conv1+conv2 上，其余 op 全程亚毫秒。
**"显存分配增长"假说被此表直接否证**：S3 的 reserved 增长 (+1.6MB) 比 S1 (+1.5MB) 还多，
但 S3 只花 0.16ms。所以贵的不是新分配显存，而是 conv 本身。

**(c) 干净进程里裸 conv2d 复现，确认是 cuDNN 的按 shape 算法选择缓存。**
`conv_shape_cost_probe.py`（新进程，只跑 `paddle.nn.Conv2D(960, 240, 1)` 喂 `[N,960,1,1]`）：

| N | ms | 说明 |
|--:|---:|------|
| 1 | 514.18 | 首次 N=1（含 cuDNN 首次初始化） |
| 2 | **51.49** | 首次 N=2 |
| 1 | 0.28 | N=1 重复 |
| 4 | **34.69** | 首次 N=4 |
| 2 | 0.24 | N=2 重复 |
| 8 | **43.95** | 首次 N=8 |
| 4 | 0.27 | N=4 重复 |

**每个首次见到的 N 都付一次几十毫秒，之后同 N 恒为 ~0.26ms。**
这就是 cuDNN 对每个新 conv 问题 shape 做一次算法选择（heuristic/benchmark）并缓存的行为。

**为什么只有 batch 敏感、H/W 不敏感？** 结构原因：本子图两个 conv 都接在
`adaptive_avg_pool2d(output_size=1)` 之后，输入已被 reduce 成 `[N, C, 1, 1]`，
**conv 的问题 shape 只随 N 变**。所以 S2/S3（只改 H/W）对 conv 而言是同一个 shape，直接命中缓存；
S1/S5（改 batch）才是新 shape。

**注意一个已被自查纠正的无效对照**：最初把裸 conv 循环放在整网循环之后、同一进程里跑，
N=1/2/4 早已被前面的整网 conv 走过、cuDNN 缓存已热，测出来是平坦的 ~0.15ms，
一度被误读为"否证 cuDNN 假说"。必须在**全新进程**里测（即 (c) 的做法）。

**结论**：S1/S5 的几十毫秒是 **conv2d 的 cuDNN 按 conv 问题 shape 选算法**的一次性成本，
与 CINN 无关（CINN 不 lower conv2d，它走 cuDNN kernel），也不是显存增长。
§9.5 已用 codegen 计数（关缓存下恒为 2）证明不存在重编译，本节进一步给出了正向归因。
这一点反而**加强**了指标 3 的结论：动态 shape 下的残余 per-batch 开销可归因到 CINN 之外的算子库行为。

产物归档于 `/work/PaddleTest/evidence_dynamicShape/`：
`eager_baseline_probe.py` + `eager_baseline.log`、`perop_probe.py` + `perop.log`、
`conv_shape_cost_probe.py` + `conv_shape_cost.log`。

### 9.8 搜索式调优的读侧验证与收益实测（2026-09-07）

§9.7 / T8.3 量化了 `policy=default` 与 static 确切尺寸的差距（group2 1.84×）并声明
"调优前 vs 调优后"未测。本节闭环该缺口（完整步骤与数据见测试步骤文档 T9）：

- **Paddle wheel 自带预搜索库**：`paddle/cinn_config/tile_config/`（A100-40GB / V100
  等机型 JSON），`paddle/__init__.py:856` import 时设 `CINN_CONFIG_PATH` 指向它，
  **会覆盖用户先设的值**，自定义库须 import 后用 `os.environ` 再覆盖。本机
  A100-80GB 无条目，`policy=optimal` + 空库 = 硬崩溃（`ir_analyzer.cc:107` 抛
  "Didn't find blocks in expr" → SIGABRT），**不是静默回退**（此前文档说法已修正）。
- **控制实验**：自定义单桶 JSON（warpNum=8）→ kernel 8→4 个、谓词变为自定义区间、
  全部 `__launch_bounds__(256)` = warpNum×32 ⇒ 数据库 tileConfig 确实进入调度。
- **手工网格 11 候选**（搜索器 `ScheduleConfigSearcher` 无产线调用方，等价复刻
  穷举）：最优 warpNum=8 + spatialInnerNum=2（block 256 / grid 1090）。
- **收益（G2@[1,72,88,88]，n=55，nsys）**：default 7252 ns → 搜索最优 5651 ns，
  **1.28×（−22.2%）**；与 static 上界差距 1.84× → 1.43×；即使 block 同为 1024
  仍快于 default（tileConfig 还改变 grid/循环结构）。
- **双桶最终配置**（小桶 [1,1023] warpNum=1 + 大桶 (8,2)）：一次编译 8 kernel、
  运行期选桶（[1,72,88,88]→5651 ns，[1,72,1,1]→2380 ns，均优于 default）。
  证据：`optimal_search.log` 等 5 个文件（evidence_dynamicShape/）。

含义：指标 3 的功能主张（按形状区间分桶 + 运行期选桶）不变；性能主张现也有实测
支撑——`policy=optimal` 读落库配置在运行点上比 default 快 1.28×，但默认关闭、
空库崩溃属可用性风险（报 issue 的候选）。

## 10. CINN 的"按形状自动调优"到底发生在哪里

"根据不同张量形状自动调优"这句指标落在**编译期**。对应的代码路径（已由 §9.4 的实测产物验证）：

- **符号化动态 shape**：带 `-1` 的 InputSpec 进入 PIR 后，动态维以符号变量（`S0`、`S1`…）表示，
  CINN 生成的 kernel 以这些符号为参数，因此一份 kernel 对所有满足约束的具体 shape 通用。
- **编译期 tile 配置（自动调优的实体）**：分桶与逐桶 tile 由
  `FLAGS_tile_config_policy` 决定——`default`（默认）走规则化分桶（§9.4(c) 的
  4 桶即其产物）；`optimal`/`hybrid` 从 `FileTileConfigDatabase`（wheel 自带的
  `paddle/cinn_config/tile_config/<arch>_<device>/...json`，§9.8）读**已调优落盘**
  的 bucket→tileConfig。搜索器 `ScheduleConfigSearcher` 无产线调用方，落库配置
  需离线搜索产生（§9.8 用手工网格等价复刻）。这一步依据**形状特征（静态维/动态维、
  reduce 维是否动态、numel 区间）**决定分块与并行策略——这就是"根据形状调优"的实体，
  它在**编译时**完成一次，产物随符号 kernel 固化。
- 因此指标 3 的两半应这样表述：
  1. 支持可变形状输入 —— 运行期能力，已由 90/90 数值一致直接证明；
  2. 根据不同张量形状自动调优 —— **编译期**能力：按形状区间分桶 + 逐桶 tile，
     一次编译产出多份形状分桶的符号 kernel，运行期按谓词选桶、不再 per-shape 重调优。
     此项已由 §9.4(c) 的 `COND__` 分桶 kernel 实测证据佐证。

### 10.1 若要观测"按形状分别调优"的差异

本子图（SE，reduce 到 1×1）对 H/W 天然形状无关，§9.4(c) 观察到的分桶维度是
"总元素数区间 + 索引位宽"两类，尚看不出 reduce 维相关的 tile 差异。若需在实验上放大
"不同形状 → 不同编译期调优策略"的差异，应改用**含动态 reduce 维**的子图（如 softmax / layernorm / 大 reduce），
比较各桶选中的 tile_config 是否随 reduce 维特征变化。

观测手段须注意三点：

- 不要用旧版 `--static-only` 的 `compiles` 计数做判据（`LOG_FIRST_N` 问题，见 §9.2）；
  本轮该探针 compiles=0 只是日志宏行为，与"结构缓存"无关，此前的归因是错的。
  脚本已改用 codegen 事件计数（§9.6），该探针的"每个 shape 是否独立触发编译"判定因此重新可用，
  但它测的是逐 shape **各自静态编译**，不代表动态 spec 下的行为，**正式验收不带这两个参数**。
- **不要用 `FLAGS_cinn_dump_group_*`**，本 build 中是死 flag（§9.4(d)）；
  改用 `FLAGS_cinn_source_code_save_path` 导出生成的 CUDA 源码，直接看
  `COND__` 分桶后缀与各桶的 `__launch_bounds__` / 循环 tile 因子。
- 需要排除缓存复用时先设 `FLAGS_enable_cinn_compile_cache=false`。



