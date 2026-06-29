#!/bin/bash
# Batch nsys profile for all sublayers, both dy and cinn modes
# Usage: bash run_nsys_batch.sh

set -e

export MIN_GRAPH_SIZE=0
export FLAGS_cinn_debug=1
export FLAGS_prim_forward_blacklist=pd_op.dropout
export PLT_DEVICE_ID=0
export PLT_GET_NV_MEMORY=False
export PLT_BM_DB=non-db
source ./scene/set_pts_env.sh
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

OUT=/work/PaddleTest/PaddleTest_QA_test_ops_20260629
mkdir -p "$OUT"

YAML=yaml/dy^dy2stcinn_eval-dy2st^dy2stcinn_eval_benchmark.yml
LAYERTEST=layertest.py

run_case() {
    local subgraph="$1"   # e.g. Det_cases/rcnn_enhance_faster_rcnn_enhance_3x_coco/SIR_107
    local mode="$2"       # dy or cinn
    local safe_name="${subgraph//\//__}"
    local log="$OUT/${safe_name}_${mode}.log"
    local nsys_out="$OUT/${safe_name}_${mode}"

    echo "=== Running $subgraph [$mode] ==="

    # Set yaml
    if [ "$mode" = "dy" ]; then
        printf 'testings:\n  dy_eval_perf:\n    model_dtype: "float32"\n\n' > "$YAML"
    else
        printf 'testings:\n  dy2st_eval_cinn_perf:\n    model_dtype: "float32"\n\n' > "$YAML"
    fi

    # Set layerfile in layertest.py (line 260)
    sed -i "260s|.*|    layerfile = \"layercase/sublayer1000/${subgraph}.py\" # 子图case路径|" "$LAYERTEST"

    nsys profile --stats=true --force-overwrite=true -o "$nsys_out" python layertest.py > "$log" 2>&1
    echo "  -> $log"
}

CASES=(
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
)

for case in "${CASES[@]}"; do
    run_case "$case" dy  || echo "FAILED: $case dy"
    run_case "$case" cinn || echo "FAILED: $case cinn"
done

echo "=== All done ==="
