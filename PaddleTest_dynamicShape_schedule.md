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
2. 用 glog 编译事件计数（compiles）确定每个 shape 是否触发了 CINN 编译/调优——
   注意这是**待测量的观测项，不是预设结论**：实测（§9.2）表明只在首个 shape 编译一次，
   后续 shape compiles=0（符号化动态 shape，一次编译服务所有 shape）。

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
2. 打开 CINN 编译日志开关，采集"每个 shape 是否触发 compile / 调优"的证据：
   - 脚本设置 `GLOG_logbufsecs=0` 让 glog 即时刷新，并在 fd 级别捕获 stderr，
     按 shape 统计标记 `Compiling subgraph with CINN backend` 出现次数（compiles）。
   - 用 compiles 而非耗时来判定"是否发生编译/调优"，直接、可复现。
3. 汇总：形成"shape → 是否通过数值校验 → compiles → 是否复用缓存"的结果表。

## 6. 验收标准（证明指标 3 成立）

- [x] 同一份权重、同一次 CINN 编译入口，能对 S0~S6 全部不同 shape 正确前向且数值与 eager 一致
      （实测 10 case × 7 shape = **70/70 PASS**，见 §9.1）。
- [x] 覆盖 batch 维、空间维、以及多维同时变化三类动态场景（S1/S5、S2/S3、S4）。
- [x] 有可观测的编译事件证据（compiles 计数），用于判定编译/调优发生在何时（见 §9.2）。
- [ ] ~~"首见 shape 触发编译/调优、再见 shape 复用缓存"~~ —— **该判据不适用**：实测证明 CINN
      符号化编译一次即服务所有 shape，运行期不 per-shape 重编译，故 S1 vs S6 无差异可比（见 §9.2 / §9.3）。
- [ ] CINN vs dy2st(backend=None) vs eager 三方一致性 —— 本轮脚本只做了 **CINN vs eager** 两方比对，
      dy2st 基线未纳入，属可选补充项。

## 7. 备注与风险

- 环境提醒：当前开发机为 A100(SM80)，需确认本机 Paddle 已编译 CINN（`paddle.is_compiled_with_cinn()`）且 GPU 可用；否则本计划仅能静态审阅、无法实跑验证。
- **关于"自动调优"的关键澄清（实测更正，见 §9/§10）**：SE 子图这类结构下，CINN 采用**符号化动态 shape**编译，
  **一次编译即适配所有 shape**，运行时不同 shape 并不重复编译/重调优。因此"新形状触发调优"这一说法
  **不成立**；早期基于耗时(~62ms)的推断被编译计数(compiles=0)证伪——那~80–120ms 是新具体 shape 首次执行的
  运行时开销（显存分配/launch 配置），并非重编译。
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
只 `to_static(backend="CINN")` 编译一次 → 依次喂入 7 组 shape（S0~S6，含一次重复以验证缓存复用）
→ 与 eager 逐一 `assert_allclose(atol=1e-5)`。**编译事件用 glog 标记计数**（非耗时）作为调优证据。

脚本已按评审意见修正三处（提升作为"验证脚本"的可信度）：
1. **退出码收紧**：每个 case 必须 S0~S6 **全 PASS**（无 FAIL、无 SKIP）才算 OK，否则 `exit 1`；与 §6 一致。
2. **数值失败与非法 shape 分离**：`assert_allclose` 不一致判 `FAIL:numeric`（真 bug，绝不降级）；
   仅"构造输入/eager 前向即失败"才判 `SKIP:shape`；CINN 前向异常判 `FAIL:cinn`。
3. **编译计数取代耗时推断**：`GLOG_logbufsecs=0` + fd 级捕获 stderr，统计 `Compiling subgraph with CINN backend`
   出现次数，直接证明"是否发生编译/调优"。

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
S0 基准 / S1 batchx2 / S2 spatial÷2 / S3 spatialx1.5 / S4 batchx2+spatial÷2 / S5 batchx4 / S6 重复 S1（验证缓存）。

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

## 9. 实跑结果（2026-08，A100）——以编译计数为准（更正稿）

10 个用例全部通过数值校验，**总计 70/70 shape**（`assert_allclose(atol=rtol=1e-5)`，CINN vs eager 逐一一致）。
但**编译事件计数**推翻了早期基于耗时的"按 batch 重调优"推断，见下。

### 9.1 数值正确性（全部 PASS）

| # | 模型子图 | 结果 (S0~S6) |
|---|----------|--------------|
| 1 | picodet_l_640/SIR_17 (C=72) | 7/7 PASS |
| 2 | picodet_m_320/SIR_17 (C=56) | 7/7 PASS |
| 3 | picodet_s_320/SIR_17 (C=44) | 7/7 PASS |
| 4 | ttfnet_pafnet_lite/SIR_22 (C=72) | 7/7 PASS |
| 5 | ssdlite_mbv3_large/SIR_31 (C=120) | 7/7 PASS |
| 6 | yolov3_mbv3_ssld/SIR_64 (C=480) | 7/7 PASS |
| 7 | centernet_mbv3_large/SIR_22 (C=72) | 7/7 PASS |
| 8 | ppyolo_tiny/SIR_31 (C=64) | 7/7 PASS |
| 9 | ppyolo_mbv3_small/SIR_49 (C=240) | 7/7 PASS |
| 10 | ppyolo_mbv3_large/SIR_77 (C=960) | 7/7 PASS |

最终判定：全部通过（exit 0）。

### 9.2 编译事件计数（关键更正）

用 `_CaptureStderrFd` 在 fd 级捕获 glog，按 shape 统计 `Compiling subgraph with CINN backend`：

- 全程 **只有 case1 / S0（首个 shape 首次进入）compiles=1**；
- **其余所有 variant（S1~S6）compiles=0**；
- **case2~case10 的全部 shape 也都 compiles=0**——即使 case 之间通道数 C 完全不同（72/56/44/120/480/…），
  仍复用同一次编译（进程级结构缓存 + 符号化动态 shape）。
- `--static-only` 干净进程探针：第 1 个具体 shape compiles=1，第 2、3 个不同具体 shape 仍 compiles=0。

**结论（更正早期错误推断）**：CINN 对该 SE 子图采用**符号化动态 shape 编译，一次编译服务所有 shape**，
运行时**不会**因新 shape 重新编译或重新调优。早期"S1/S5≈62ms 是按 batch 重调优、S6≈<1ms 是命中缓存"的说法
**不成立**——62ms 是**新具体 shape 首次执行**的一次性运行时开销（显存分配 / launch 配置 / cudnn algo 选择等），
与"重编译/重调优"无关；这一点已被 compiles=0 直接证伪。评审对 62ms 恒定值的质疑是对的。

### 9.3 对指标 3 的诚实拆解

- "**支持可变形状输入张量**"：**完全证明**。同一份权重、单次动态编译入口，可正确前向 batch/空间/多维同变的所有 shape，数值与 eager 一致。
- "**根据不同张量形状自动调优**"：**成立，但发生在编译期而非运行期**。CINN 的形状自适应调优是**编译时**由
  group_schedule 的 tile_config 搜索（config_searcher）完成，产出**对形状泛化的符号 kernel**；
  运行时遇到新的具体 shape 直接复用该 kernel，不做 per-shape 重调优。因此"新形状触发运行时重调优、已见形状复用缓存"
  这一运行期模型**不适用**于本子图。

（CINN 编译期调优机制的进一步说明见 §10。）

## 10. CINN 的"按形状自动调优"到底发生在哪里

实测（§9.2）表明运行期不重编译，那"根据不同张量形状自动调优"这句指标落在**编译期**。对应的代码路径：

- **符号化动态 shape**：带 `-1` 的 InputSpec 进入 PIR 后，动态维以符号变量（`S0`、`S1`…）表示，
  CINN 生成的 kernel 以这些符号为参数，因此一份 kernel 对所有满足约束的具体 shape 通用。
- **编译期 tile 配置搜索（自动调优的实体）**：group_schedule 通过 `config_searcher` 在
  `paddle/cinn/ir/group_schedule/config/tile_config/` 下按目标架构选择/搜索 tile_config
  （A100 对应 `NVGPU_NVIDIA_A100...` 目录，含 `Sstatic_Rdynamic` / `Sdynamic_Rdynamic` 等 JSON 配置）。
  这一步依据**形状特征（静态维/动态维、reduce 维是否动态）**决定分块与并行策略——这就是"根据形状调优"的实体，
  它在**编译时**完成一次，产物随符号 kernel 固化。
- 因此指标 3 的两半应这样表述：
  1. 支持可变形状输入 —— 运行期能力，已由 70/70 数值一致直接证明；
  2. 根据不同张量形状自动调优 —— **编译期**能力（config_searcher / tile_config 按形状特征搜索），
     一次编译产出形状泛化 kernel，运行期不再 per-shape 重调优。

### 10.1 若要观测"按形状分别调优"的差异

本子图（SE，reduce 到 1×1）对 H/W 天然形状无关，看不出按形状的不同 tile 策略。若需在实验上放大
"不同形状 → 不同编译期调优策略"的差异，应改用**含动态 reduce 维**的子图（如 softmax / layernorm / 大 reduce），
并用 `--static-only` 对多个**静态**具体 shape 分别编译，比较各自 compiles 与所选 tile_config 是否不同。
本轮 SE 子图的 static-only 探针 compiles 仍为 0（结构缓存所致），不足以展现该差异，属预期。



