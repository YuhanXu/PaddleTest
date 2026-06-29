一、性能单例调试模式（默认为GPU）：
1、拉取PaddleTest：
git clone https://github.com/PaddlePaddle/PaddleTest.git
cd PaddleTest/framework/e2e/PaddleLT_new

2、设定环境变量：
#使用CINN环境变量：
export MIN_GRAPH_SIZE=0
export FLAGS_cinn_debug=1
export FLAGS_prim_forward_blacklist=pd_op.dropout;

# 重要！设定默认环境变量用于测试框架相关参数设定
export PLT_DEVICE_ID=0
export PLT_GET_NV_MEMORY=False
export PLT_BM_DB=non-db
source ./scene/set_pts_env.sh

3、找到layertest.py，找到代码最后几行，设定需要排查的 子图（layerfile）和 测试配置（testing）
注意：执行路径下应该没有framework/e2e/PaddleLT_new，只用写layertest/…/ 后面的路径

if __name__ == "__main__":
    layerfile = "layercase/sublayer1000/Clas_cases/CSWinTransformer_CSWinTransformer_base_384/SIR_236.py" # 子图case路径
    testing = "yaml/dy^dy2stcinn_eval-dy2st^dy2stcinn_eval_benchmark.yml"
    single_test = LayerTest(title="your_name", layerfile=layerfile, testing=testing)
    single_test._perf_case_run()  # 此处与功能精度不同，使用性能接口 _perf_case_run
运行：
python layertest.py

/work/PaddleTest/framework/e2e/PaddleLT_new/yaml/dy^dy2stcinn_eval-dy2st^dy2stcinn_eval_benchmark.yml里是配置跑那种测试场景。
这样写是表示跑”动态图“场景：
testings:
  dy_eval_perf:
    model_dtype: "float32"

这样写表示跑”静态图开cinn“场景：
testings:
  dy2st_eval_cinn_perf:
    model_dtype: "float32"


子图列表：
Det_cases/rcnn_enhance_faster_rcnn_enhance_3x_coco/SIR_107
Det_cases/picodet_legacy_model_picodet_s_416_coco/SIR_295
Det_cases/picodet_legacy_model_picodet_l_320_coco/SIR_341
Det_cases/gfl_gfl_r50_fpn_1x_coco/SIR_140
Det_cases/picodet_legacy_model_picodet_m_416_coco/SIR_348
Det_cases/picodet_legacy_model_picodet_s_416_coco/SIR_285
Det_cases/rotate_s2anet_s2anet_conv_2x_dota/SIR_109
Det_cases/rotate_s2anet_s2anet_conv_2x_dota/SIR_108
Clas_cases/Twins_alt_gvt_base/SIR_5
Clas_cases/Twins_alt_gvt_base/SIR_153
Det_cases/rotate_s2anet_s2anet_conv_2x_dota/SIR_106
Det_cases/picodet_legacy_model_picodet_s_416_coco/SIR_286
Det_cases/rotate_s2anet_s2anet_conv_2x_dota/SIR_107
Det_cases/faster_rcnn_faster_rcnn_swin_tiny_fpn_2x_coco/SIR_116
Det_cases/picodet_legacy_model_picodet_s_416_coco/SIR_296
Clas_cases/Twins_alt_gvt_base/SIR_165
Clas_cases/Twins_alt_gvt_base/SIR_87
Clas_cases/Twins_alt_gvt_base/SIR_17
Det_cases/rotate_s2anet_s2anet_conv_2x_dota/SIR_105
Det_cases/hrnet_faster_rcnn_hrnetv2p_w18_2x_coco/SIR_60
Det_cases/sniper_faster_rcnn_r50_fpn_1x_sniper_visdrone/SIR_64
Det_cases/tood_tood_r50_fpn_1x_coco/SIR_34
Det_cases/rotate_fcosr_fcosr_x50_3x_dota/SIR_35
Det_cases/sniper_faster_rcnn_r50_fpn_1x_visdrone/SIR_64
Det_cases/ppyoloe_ppyoloe_plus_crn_l_80e_coco/SIR_167
Clas_cases/Twins_alt_gvt_base/SIR_99
Det_cases/gfl_gfl_r50_fpn_1x_coco/SIR_141
Clas_cases/Twins_alt_gvt_base/SIR_29
Clas_cases/Twins_alt_gvt_base/SIR_177
Clas_cases/Twins_alt_gvt_base/SIR_73
Seg_cases/danet_danet_resnet50_os8_voc12aug_512x512_40k/SIR_31
Det_cases/picodet_legacy_model_picodet_m_416_coco/SIR_349
Det_cases/rotate_ppyoloe_r_ppyoloe_r_crn_m_3x_dota_ms/SIR_168
Det_cases/picodet_legacy_model_picodet_l_320_coco/SIR_342
Det_cases/sparse_rcnn_sparse_rcnn_r50_fpn_3x_pro300_coco/SIR_62
Det_cases/rotate_ppyoloe_r_ppyoloe_r_crn_x_3x_dota/SIR_167
Clas_cases/LeViT_LeViT_128/SIR_37
Det_cases/rotate_ppyoloe_r_ppyoloe_r_crn_s_3x_dota/SIR_169
Det_cases/mot_fairmot_fairmot_dla34_30e_1088x608_bytetracker/SIR_79
Det_cases/yolox_yolox_m_300e_coco/SIR_123
Det_cases/yolox_yolox_nano_300e_coco/SIR_126
Det_cases/yolox_yolox_crn_s_300e_coco/SIR_201
Det_cases/semi_det_baseline_ppyoloe_plus_crn_s_80e_coco_sup005/SIR_50
Det_cases/rotate_ppyoloe_r_ppyoloe_r_crn_l_3x_dota/SIR_167
Det_cases/sparse_rcnn_sparse_rcnn_r50_fpn_3x_pro100_coco/SIR_62
Det_cases/yolox_yolox_cdn_tiny_300e_coco/SIR_125
Clas_cases/Twins_alt_gvt_base/SIR_243
Det_cases/mot_fairmot_fairmot_enhance_hardnet85_30e_1088x608/SIR_84
Det_cases/ppyoloe_voc_ppyoloe_plus_crn_s_30e_voc/SIR_21
Seg_cases/danet_danet_resnet50_os8_cityscapes_1024x512_80k/SIR_31
Det_cases/yolox_yolox_crn_s_300e_coco/SIR_63
Clas_cases/Twins_alt_gvt_base/SIR_111
Det_cases/rcnn_enhance_faster_rcnn_enhance_3x_coco/SIR_106
Det_cases/ppyoloe_ppyoloe_crn_s_400e_coco/SIR_48
Clas_cases/LeViT_LeViT_128/SIR_32
Det_cases/ppyoloe_ppyoloe_plus_crn_x_80e_coco/SIR_74
Det_cases/rcnn_enhance_faster_rcnn_enhance_3x_coco/SIR_103


跑性能PaddleLayerTest子图nsys方案：
子图列表里是开CINN和不开CINN性能差距很大子图，性能提升来自于"静态图开cinn"场景的算子融合。
/work/PaddleTest/PaddleLayerTest.md是列表里这些子图运行的方式。
现在我想针对列表里每一个子图测试，分别开nsys profile --stats=true，分别跑”动态图“和”静态图开cinn“这两种场景。并把nsys profile的结果分别存到/work/PaddleTest/PaddleTest_QA_test_ops_20260629文件夹里。
给个最佳实践：
对于列表里的第一个子图Det_cases/rcnn_enhance_faster_rcnn_enhance_3x_coco/SIR_107，我可以通过修改layertest.py里第260行控制跑哪个子图。通过修改/work/PaddleTest/framework/e2e/PaddleLT_new/yaml/dy^dy2stcinn_eval-dy2st^dy2stcinn_eval_benchmark.yml控制跑那种场景。通过
nsys profile --stats=true python layertest.py > /work/PaddleTest/PaddleTest_QA_test_ops_20260629/Det_cases__rcnn_enhance_faster_rcnn_enhance_3x_coco__SIR_107_dy.log 2>&1
nsys profile --stats=true python layertest.py > /work/PaddleTest/PaddleTest_QA_test_ops_20260629/Det_cases__rcnn_enhance_faster_rcnn_enhance_3x_coco__SIR_107_cinn.log 2>&1
将nsys结果存下来。

具体安排如下：
1.你先学习一下我以上提到的脚本。
2.先跑Det_cases/rcnn_enhance_faster_rcnn_enhance_3x_coco/SIR_107这一个子图，让我看一下两种场景的log
3.再批跑列表中所有子图
4.有关键报错或者进展更新到/work/PaddleTest/PaddleLayerTest.md文件里

进展记录（2026-06-29）：
- 已创建输出目录：/work/PaddleTest/PaddleTest_QA_test_ops_20260629
- 运行 nsys 前需要补充环境：export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH，否则 libcuda.so.1 not found 导致 import paddle 失败。
- SIR_107 初次运行报错：AttributeError: module 'paddle.tensor' has no attribute 'ops'。已将 paddle.tensor.ops.atan(...) 改为 paddle.atan(...)，其余子图如遇相同问题同理修复。
- 使用 _perf_case_run() + nsys profile --stats=true 方式跑性能。
- 批量脚本：/work/PaddleTest/framework/e2e/PaddleLT_new/run_nsys_batch.sh，已全部完成，共 57 个子图 × 2 场景 = 114 个 nsys profile。
- 输出目录：/work/PaddleTest/PaddleTest_QA_test_ops_20260629，包含 57 个 dy log、57 个 cinn log、114 个 nsys-rep。

失败子图（7个，均为 `paddle.tensor.ops` 旧 API 报错）：
- Det_cases/mot_fairmot_fairmot_dla34_30e_1088x608_bytetracker/SIR_79
- Det_cases/yolox_yolox_m_300e_coco/SIR_123
- Det_cases/yolox_yolox_nano_300e_coco/SIR_126
- Det_cases/yolox_yolox_crn_s_300e_coco/SIR_201
- Det_cases/yolox_yolox_cdn_tiny_300e_coco/SIR_125
- Det_cases/rcnn_enhance_faster_rcnn_enhance_3x_coco/SIR_106
- Det_cases/rcnn_enhance_faster_rcnn_enhance_3x_coco/SIR_103

nsys profile 性能结果（dy / cinn，单位秒）：

| 子图 | dy (s) | cinn (s) | 加速比 |
|------|--------|----------|--------|
| Det_cases/rcnn_enhance_faster_rcnn_enhance_3x_coco/SIR_107 | 0.158151 | 0.018099 | 8.7x |
| Det_cases/picodet_legacy_model_picodet_s_416_coco/SIR_295 | 0.075549 | 0.022549 | 3.4x |
| Det_cases/picodet_legacy_model_picodet_l_320_coco/SIR_341 | 0.076040 | 0.018708 | 4.1x |
| Det_cases/gfl_gfl_r50_fpn_1x_coco/SIR_140 | 0.071923 | 0.020138 | 3.6x |
| Det_cases/picodet_legacy_model_picodet_m_416_coco/SIR_348 | 0.075757 | 0.018493 | 4.1x |
| Det_cases/picodet_legacy_model_picodet_s_416_coco/SIR_285 | 0.074098 | 0.017964 | 4.1x |
| Det_cases/rotate_s2anet_s2anet_conv_2x_dota/SIR_109 | 0.021662 | 0.018198 | 1.2x |
| Det_cases/rotate_s2anet_s2anet_conv_2x_dota/SIR_108 | 0.021067 | 0.020479 | 1.0x |
| Clas_cases/Twins_alt_gvt_base/SIR_5 | 0.137408 | 0.109571 | 1.3x |
| Clas_cases/Twins_alt_gvt_base/SIR_153 | 0.137253 | 0.111058 | 1.2x |
| Det_cases/rotate_s2anet_s2anet_conv_2x_dota/SIR_106 | 0.250143 | 0.018402 | 13.6x |
| Det_cases/picodet_legacy_model_picodet_s_416_coco/SIR_286 | 0.062264 | 0.026433 | 2.4x |
| Det_cases/rotate_s2anet_s2anet_conv_2x_dota/SIR_107 | 0.022306 | 0.245636 | 0.1x |
| Det_cases/faster_rcnn_faster_rcnn_swin_tiny_fpn_2x_coco/SIR_116 | 0.029786 | 0.032104 | 0.9x |
| Det_cases/picodet_legacy_model_picodet_s_416_coco/SIR_296 | 0.062254 | 0.027193 | 2.3x |
| Clas_cases/Twins_alt_gvt_base/SIR_165 | 0.065547 | 0.085213 | 0.8x |
| Clas_cases/Twins_alt_gvt_base/SIR_87 | 0.043549 | 0.042003 | 1.0x |
| Clas_cases/Twins_alt_gvt_base/SIR_17 | 0.064312 | 0.084761 | 0.8x |
| Det_cases/rotate_s2anet_s2anet_conv_2x_dota/SIR_105 | 0.020404 | 0.017941 | 1.1x |
| Det_cases/hrnet_faster_rcnn_hrnetv2p_w18_2x_coco/SIR_60 | 0.030338 | 0.028782 | 1.1x |
| Det_cases/sniper_faster_rcnn_r50_fpn_1x_sniper_visdrone/SIR_64 | 0.028803 | 0.019853 | 1.5x |
| Det_cases/tood_tood_r50_fpn_1x_coco/SIR_34 | 0.376660 | 0.038919 | 9.7x |
| Det_cases/rotate_fcosr_fcosr_x50_3x_dota/SIR_35 | 0.301914 | 0.026529 | 11.4x |
| Det_cases/sniper_faster_rcnn_r50_fpn_1x_visdrone/SIR_64 | 0.029532 | 0.019675 | 1.5x |
| Det_cases/ppyoloe_ppyoloe_plus_crn_l_80e_coco/SIR_167 | 0.212065 | 0.039221 | 5.4x |
| Clas_cases/Twins_alt_gvt_base/SIR_99 | 0.026948 | 0.038532 | 0.7x |
| Det_cases/gfl_gfl_r50_fpn_1x_coco/SIR_141 | 0.065564 | 0.031819 | 2.1x |
| Clas_cases/Twins_alt_gvt_base/SIR_29 | 0.059930 | 0.077067 | 0.8x |
| Clas_cases/Twins_alt_gvt_base/SIR_177 | 0.059979 | 0.076148 | 0.8x |
| Clas_cases/Twins_alt_gvt_base/SIR_73 | 0.057283 | 0.070279 | 0.8x |
| Seg_cases/danet_danet_resnet50_os8_voc12aug_512x512_40k/SIR_31 | 0.048617 | 0.065295 | 0.7x |
| Det_cases/picodet_legacy_model_picodet_m_416_coco/SIR_349 | 0.065755 | 0.026582 | 2.5x |
| Det_cases/rotate_ppyoloe_r_ppyoloe_r_crn_m_3x_dota_ms/SIR_168 | 0.180990 | 0.025004 | 7.2x |
| Det_cases/picodet_legacy_model_picodet_l_320_coco/SIR_342 | 0.060914 | 0.030894 | 2.0x |
| Det_cases/sparse_rcnn_sparse_rcnn_r50_fpn_3x_pro300_coco/SIR_62 | 0.073148 | 0.063359 | 1.2x |
| Det_cases/rotate_ppyoloe_r_ppyoloe_r_crn_x_3x_dota/SIR_167 | 0.184397 | 0.025661 | 7.2x |
| Clas_cases/LeViT_LeViT_128/SIR_37 | 0.295428 | 0.025493 | 11.6x |
| Det_cases/rotate_ppyoloe_r_ppyoloe_r_crn_s_3x_dota/SIR_169 | 0.181015 | 0.026407 | 6.9x |
| Det_cases/mot_fairmot_fairmot_dla34_30e_1088x608_bytetracker/SIR_79 | FAIL | FAIL | - |
| Det_cases/yolox_yolox_m_300e_coco/SIR_123 | FAIL | FAIL | - |
| Det_cases/yolox_yolox_nano_300e_coco/SIR_126 | FAIL | FAIL | - |
| Det_cases/yolox_yolox_crn_s_300e_coco/SIR_201 | FAIL | FAIL | - |
| Det_cases/semi_det_baseline_ppyoloe_plus_crn_s_80e_coco_sup005/SIR_50 | 0.033456 | 0.032609 | 1.0x |
| Det_cases/rotate_ppyoloe_r_ppyoloe_r_crn_l_3x_dota/SIR_167 | 0.186875 | 0.023129 | 8.1x |
| Det_cases/sparse_rcnn_sparse_rcnn_r50_fpn_3x_pro100_coco/SIR_62 | 0.075210 | 0.057657 | 1.3x |
| Det_cases/yolox_yolox_cdn_tiny_300e_coco/SIR_125 | FAIL | FAIL | - |
| Clas_cases/Twins_alt_gvt_base/SIR_243 | 0.025274 | 0.037537 | 0.7x |
| Det_cases/mot_fairmot_fairmot_enhance_hardnet85_30e_1088x608/SIR_84 | 0.023607 | 0.043193 | 0.5x |
| Det_cases/ppyoloe_voc_ppyoloe_plus_crn_s_30e_voc/SIR_21 | 0.032545 | 0.031919 | 1.0x |
| Seg_cases/danet_danet_resnet50_os8_cityscapes_1024x512_80k/SIR_31 | 0.079542 | 0.099967 | 0.8x |
| Det_cases/yolox_yolox_crn_s_300e_coco/SIR_63 | 0.040573 | 0.037982 | 1.1x |
| Clas_cases/Twins_alt_gvt_base/SIR_111 | 0.026218 | 0.035524 | 0.7x |
| Det_cases/rcnn_enhance_faster_rcnn_enhance_3x_coco/SIR_106 | FAIL | FAIL | - |
| Det_cases/ppyoloe_ppyoloe_crn_s_400e_coco/SIR_48 | 0.032669 | 0.035288 | 0.9x |
| Clas_cases/LeViT_LeViT_128/SIR_32 | 0.009176 | 0.019733 | 0.5x |
| Det_cases/ppyoloe_ppyoloe_plus_crn_x_80e_coco/SIR_74 | 0.034751 | 0.031726 | 1.1x |
| Det_cases/rcnn_enhance_faster_rcnn_enhance_3x_coco/SIR_103 | FAIL | FAIL | - |