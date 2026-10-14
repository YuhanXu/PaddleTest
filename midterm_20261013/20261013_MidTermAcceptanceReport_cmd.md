
## 环境检测

cd /work/PaddleTest/midterm_20261013
    export PYTHONPATH=/usr/local/lib/python3.10/dist-packages
    export LD_LIBRARY_PATH=:/usr/lib64/:/usr/local/lib/
    /work/env3.10/bin/python env_check.py

### 任务1
## 测试项1
cd PaddleTest/framework/e2e/PaddleLT_new
export MIN_GRAPH_SIZE=0
export FLAGS_cinn_debug=1
export FLAGS_prim_forward_blacklist=pd_op.dropout
export PLT_DEVICE_ID=0
export PLT_GET_NV_MEMORY=False
export PLT_BM_DB=non-db
source scene/set_pts_env.sh

#手动修改layertest.py
python layertest.py

#精度测试
testing = "yaml/dy^dy2stcinn_eval_inputspec.yml"
single_test = LayerTest(title="your_name", layerfile=layerfile, testing=testing)
single_test._case_run()

#跑性能测试
testing = "yaml/dy^dy2stcinn_eval-dy2st^dy2stcinn_eval_benchmark.yml"
single_test = LayerTest(title="your_name", layerfile=layerfile, testing=testing)
single_test._perf_case_run()



#批跑10个case精度测试
bash /work/PaddleTest/midterm_20261013/run_precision_top10.sh > /work/run_precision_top10.log 2>&1; echo "exit=$?"

#批跑10个case性能测试
cd /work/PaddleTest/framework/e2e/PaddleLT_new
unset CASE_TYPE CASE_DIR TESTING_MODE TESTING
PLT_CSV_IN=/work/PaddleTest/midterm_20261013/top10_cases.csv PLT_CSV_TOP_N=10 \
PLT_CSV_OUT=/work/all_perf_results_top10.csv \
PLT_CSV_LOG_DIR=/work/case_logs_top10 \
python run.py

100 个单 iter 耗时样本（秒）取截尾均值，再乘 100 输出。


性能：midterm_20261013/all_perf_results_top10.csv（10/10 cinn 胜出，加速比 1.5x~11x）
精度：midterm_20261013/precision_results_top10.csv（10/10 passed）
日志：/work/case_logs_top10/（性能）、/work/case_logs_top10_prec/（精度，运行时生成）
复现脚本：midterm_20261013/run_precision_top10.sh + single_case_run_runner.py（PaddleLT_new）

## 测试项3
# 动态Shape功能及正确性验证
cd /work/PaddleTest/framework/e2e/PaddleLT_new
unset GLOG_vmodule GLOG_logtostderr
python test_dynamic_shape_cinn.py 2>&1 | tee dynamic_shape_cinn.log
echo "exit=$?"

# 动态Shape符号化验证--在PIR中间表示阶段，动态维度的shape是用-1表示的
cd /work/PaddleTest/framework/e2e/PaddleLT_new
CASE=layercase/sublayer1000/Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py

export GLOG_logtostderr=1
export GLOG_vmodule='shape_o*=3'
python ../../../evidence_dynamicShape/ir_probe.py "$CASE" > ir_symbolic.log 2>&1

grep -n "\[ShapeDialect\]" ir_symbolic.log
grep -n "pd_op.data" ir_symbolic.log | head -2

# 动态Shape自动调优Kernel验证--生成的kernel是支持动态shape作为参数传入的
cd /work/PaddleTest/framework/e2e/PaddleLT_new
CASE=layercase/sublayer1000/Det_cases/picodet_legacy_model_picodet_l_640_coco/SIR_17.py

export FLAGS_cinn_source_code_save_path=$PWD/cinn_source.cu
python ../../../evidence_dynamicShape/ir_probe.py "$CASE" > /dev/null 2>&1

grep -n "int32_t S0"  cinn_source.cu | head          # (a) 符号形参
grep -c "COND__"      cinn_source.cu            # (c) 分桶 kernel 个数
grep -E "\b(88|132|44)\b" cinn_source.cu | grep -v COND__ | head   # 应无尺寸常量

## 动态Shape场景性能测试--动态shape场景性能提升测试
cd /work/PaddleTest/framework/e2e/PaddleLT_new

# 关键: 清掉残留模式变量(上次 rc=127 的根因), TESTING 必须显式设为双执行器 yaml
unset CASE_TYPE CASE_DIR TESTING_MODE MULTI_WORKER
export TESTING="yaml/dy^dy2stcinn_eval-dy2st^dy2stcinn_eval_benchmark.yml"
export MIN_GRAPH_SIZE=0 FLAGS_cinn_debug=1 FLAGS_prim_forward_blacklist=pd_op.dropout
export PLT_DEVICE_ID=0 PLT_GET_NV_MEMORY=False PLT_BM_DB=non-db
export FRAMEWORK=paddle PLT_SET_DEVICE=gpu
export LD_LIBRARY_PATH=/usr/lib64
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:.

PLT_CSV_BATCH=True \
PLT_CSV_IN=/work/PaddleTest/midterm_20261013/top3_cases.csv \
PLT_CSV_OUT=/work/all_perf_results_top3.csv \
PLT_CSV_TOP_N=3 \
PLT_CSV_LOG_DIR=/work/case_logs_top3 \
python run.py

## 动态Shape场景精度测试
bash /work/PaddleTest/midterm_20261013/run_precision_top3.sh



git status --porcelain | grep -v '^??'

验收标准(固定):  atol = 0.01,  rtol = 0.01

逐点容差(每个输出元素各算一次):
    tol[i] = atol + rtol × |expect[i]|
           = 0.01 + 0.01 × 该点期望值

实测(找最大绝对误差点):
    max_abs_diff     = max_i |result[i] − expect[i]|        ← 最坏点的实际误差
    tol_at_max_diff  = atol + rtol × |expect[该最坏点]|     ← 该点按标准算出的容差

额度占用率:
    diff/tol = max_abs_diff ÷ tol_at_max_diff × 100%        ← <100% 即通过
代入 SIR_14：最坏点期望值 ≈0.166 → tol = 0.01 + 0.01×0.166 = 1.166e-02；实际误差 7.69e-05；占比 0.66%（裕量 99.34%）。assert_allclose 的完整判定是所有点都满足 |diff[i]| ≤ tol[i]，明细行展示的是最坏绝对误差点的情形。


