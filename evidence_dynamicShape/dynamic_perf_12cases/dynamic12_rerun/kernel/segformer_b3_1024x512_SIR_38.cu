
extern "C" {

__global__
void __launch_bounds__(320) fn_bc_add__COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_2)
{
  __builtin_assume(((int)blockIdx.x < 512));
  __builtin_assume(((int)threadIdx.x < 320));
  for (int32_t i_j_fused_k_fused_0 = 0; i_j_fused_k_fused_0 < 4; i_j_fused_k_fused_0 += 1) {
    float var_1_local = var_1[(((((int)blockIdx.x * 4) + i_j_fused_k_fused_0) * 320) + (int)threadIdx.x)];
    float var_local = var[(int)threadIdx.x];
    var_2[(((((int)blockIdx.x * 4) + i_j_fused_k_fused_0) * 320) + (int)threadIdx.x)] = (var_1_local + var_local);
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_transpose_reshape__COND_true__kernel(const float* __restrict__ var, float* __restrict__ var_1)
{
  __builtin_assume(((int)blockIdx.x < 640));
  __builtin_assume(((int)threadIdx.x < 256));
  for (int32_t i_j_k_fused_0 = 0; i_j_k_fused_0 < 4; i_j_k_fused_0 += 1) {
    float var_local_0 = var[((((((((((int)blockIdx.x * 1024) + (i_j_k_fused_0 * 256)) + (int)threadIdx.x) % 320) / 64) * 2048) + (((((int)blockIdx.x * 1024) + (i_j_k_fused_0 * 256)) + (int)threadIdx.x) / 320)) * 64) + ((int)threadIdx.x & 63))];
    var_1[((((int)blockIdx.x * 1024) + (i_j_k_fused_0 * 256)) + (int)threadIdx.x)] = var_local_0;
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_bc_add_reshape_transpose__COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_4)
{
  __builtin_assume(((int)blockIdx.x < 640));
  __builtin_assume(((int)threadIdx.x < 256));
  for (int32_t i_j_k_a_fused_3 = 0; i_j_k_a_fused_3 < 4; i_j_k_a_fused_3 += 1) {
    float var_1_local_0 = var_1[((((((((int)blockIdx.x * 1024) + (i_j_k_a_fused_3 * 256)) + (int)threadIdx.x) / 131072) * 64) + ((int)threadIdx.x & 63)) + (((((((int)blockIdx.x * 1024) + (i_j_k_a_fused_3 * 256)) + (int)threadIdx.x) & 131071) / 64) * 320))];
    float var_local_1 = var[(((((((int)blockIdx.x * 1024) + (i_j_k_a_fused_3 * 256)) + (int)threadIdx.x) / 131072) * 64) + ((int)threadIdx.x & 63))];
    var_4[((((int)blockIdx.x * 1024) + (i_j_k_a_fused_3 * 256)) + (int)threadIdx.x)] = (var_1_local_0 + var_local_1);
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_sum_full_div_bc_sub_mul_sum_full_div_full_add_rsqrt_bc_mul_bc_mul_bc_add__COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_14, const float* __restrict__ var_17, float* __restrict__ var_19)
{
  __builtin_assume(((int)blockIdx.x < 256));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 8));
  float _var_0_rf_temp_buffer [ 1 ];
  float _var_0_temp_buffer [ 1 ];
  float _var_6_rf_temp_buffer [ 1 ];
  float _var_6_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_0 = _var_0_temp_buffer;
  float* var_0_rf = _var_0_rf_temp_buffer;
  float* var_0_rf__reduce_init = _var_0_rf_temp_buffer;
  float* var_6 = _var_6_temp_buffer;
  float* var_6_rf = _var_6_rf_temp_buffer;
  float* var_6_rf__reduce_init = _var_6_rf_temp_buffer;
  var_0_rf__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_5 = 0; reduce_k_0_5 < 10; reduce_k_0_5 += 1) {
    float var_local_2 = var[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 320) + (reduce_k_0_5 * 32)) + (int)threadIdx.x)];
    var_0_rf[0] = (var_0_rf[0] + var_local_2);
  };
  var_0[0] = cinn_block_reduce_sum_fp32(var_0_rf[0], shm32__fp32_reduce, true);
  var_6_rf__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_1 = 0; reduce_k_0_1 < 10; reduce_k_0_1 += 1) {
    float var_local_3 = var[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 320) + (reduce_k_0_1 * 32)) + (int)threadIdx.x)];
    var_6_rf[0] = (var_6_rf[0] + ((var_local_3 - (var_0[0] / 320.000000f)) * (var_local_3 - (var_0[0] / 320.000000f))));
  };
  var_6[0] = cinn_block_reduce_sum_fp32(var_6_rf[0], shm32__fp32_reduce, true);
  for (int32_t k_1 = 0; k_1 < 10; k_1 += 1) {
    float var_local_4 = var[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 320) + (k_1 * 32)) + (int)threadIdx.x)];
    float var_14_local = var_14[((k_1 * 32) + (int)threadIdx.x)];
    float var_17_local = var_17[((k_1 * 32) + (int)threadIdx.x)];
    var_19[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 320) + (k_1 * 32)) + (int)threadIdx.x)] = ((((var_local_4 - (var_0[0] / 320.000000f)) * cinn_nvgpu_rsqrt_fp32(((var_6[0] / 320.000000f) + 9.99999997e-07f))) * var_14_local) + var_17_local);
  };
}

}

extern "C" {

__global__
void __launch_bounds__(64) fn_reshape_gs_bc_add_reshape_transpose_sum_full_gs_bc_div_gs_bc_sub_mul_sum_full_gs_bc_div_full_gs_bc_add_rsqrt_gs_bc_mul_gs_bc_mul_gs_bc_add__COND__FPA__FPA__FPA_S5MULS6_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S5MULS6_BPA_MUL320ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_30, const float* __restrict__ var_34, float* __restrict__ var_36, int32_t S5, int32_t S6)
{
  __builtin_assume(((int)blockIdx.x < (S5 * S6)));
  __builtin_assume(((int)threadIdx.x < 64));
  float _var_16_loopalign_1_rf_temp_buffer [ 1 ];
  float _var_16_loopalign_1_temp_buffer [ 1 ];
  float _var_7_rf_temp_buffer [ 1 ];
  float _var_7_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_16_loopalign_1 = _var_16_loopalign_1_temp_buffer;
  float* var_16_loopalign_1_rf = _var_16_loopalign_1_rf_temp_buffer;
  float* var_16_loopalign_1_rf__reduce_init = _var_16_loopalign_1_rf_temp_buffer;
  float* var_7 = _var_7_temp_buffer;
  float* var_7_rf = _var_7_rf_temp_buffer;
  float* var_7_rf__reduce_init = _var_7_rf_temp_buffer;
  var_7_rf__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_3 = 0; reduce_k_0_3 < 5; reduce_k_0_3 += 1) {
    float var_1_local_1 = var_1[((((((((int)blockIdx.x / (S5 * S6)) * 320) + (reduce_k_0_3 * 64)) + (int)threadIdx.x) * S5) * S6) + ((int)blockIdx.x % (S5 * S6)))];
    float var_local_5 = var[((reduce_k_0_3 * 64) + (int)threadIdx.x)];
    var_7_rf[0] = (var_7_rf[0] + (var_1_local_1 + var_local_5));
  };
  var_7[0] = cinn_block_reduce_sum_fp32(var_7_rf[0], shm32__fp32_reduce, false);
  var_16_loopalign_1_rf__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_4 = 0; reduce_k_0_4 < 5; reduce_k_0_4 += 1) {
    float var_1_local_2 = var_1[(((((reduce_k_0_4 * 64) + (int)threadIdx.x) * S5) * S6) + ((int)blockIdx.x % (S5 * S6)))];
    float var_local_6 = var[((reduce_k_0_4 * 64) + (int)threadIdx.x)];
    var_16_loopalign_1_rf[0] = (var_16_loopalign_1_rf[0] + (((var_1_local_2 + var_local_6) - (var_7[0] / 320.000000f)) * ((var_1_local_2 + var_local_6) - (var_7[0] / 320.000000f))));
  };
  var_16_loopalign_1[0] = cinn_block_reduce_sum_fp32(var_16_loopalign_1_rf[0], shm32__fp32_reduce, false);
  for (int32_t k_5 = 0; k_5 < 5; k_5 += 1) {
    float var_1_local_3 = var_1[((((((((int)blockIdx.x / (S5 * S6)) * 320) + (k_5 * 64)) + (int)threadIdx.x) * S5) * S6) + ((int)blockIdx.x % (S5 * S6)))];
    float var_local_7 = var[((k_5 * 64) + (int)threadIdx.x)];
    float var_30_local = var_30[((k_5 * 64) + (int)threadIdx.x)];
    float var_34_local = var_34[((k_5 * 64) + (int)threadIdx.x)];
    var_36[(((k_5 * 64) + (int)threadIdx.x) + ((int)blockIdx.x * 320))] = (((((var_1_local_3 + var_local_7) - (var_7[0] / 320.000000f)) * cinn_nvgpu_rsqrt_fp32(((var_16_loopalign_1[0] / 320.000000f) + 9.99999975e-06f))) * var_30_local) + var_34_local);
  };
}__global__
void __launch_bounds__(64) fn_reshape_gs_bc_add_reshape_transpose_sum_full_gs_bc_div_gs_bc_sub_mul_sum_full_gs_bc_div_full_gs_bc_add_rsqrt_gs_bc_mul_gs_bc_mul_gs_bc_add__COND__FPA__FPA__FPA_S5MULS6_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S5MULS6_BPA_MUL320ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_30, const float* __restrict__ var_34, float* __restrict__ var_36, int64_t S5, int64_t S6)
{
  __builtin_assume(((int)blockIdx.x < (S5 * S6)));
  __builtin_assume(((int)threadIdx.x < 64));
  float _var_16_loopalign_1_rf_temp_buffer [ 1 ];
  float _var_16_loopalign_1_temp_buffer [ 1 ];
  float _var_7_rf_temp_buffer [ 1 ];
  float _var_7_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_16_loopalign_1 = _var_16_loopalign_1_temp_buffer;
  float* var_16_loopalign_1_rf = _var_16_loopalign_1_rf_temp_buffer;
  float* var_16_loopalign_1_rf__reduce_init = _var_16_loopalign_1_rf_temp_buffer;
  float* var_7 = _var_7_temp_buffer;
  float* var_7_rf = _var_7_rf_temp_buffer;
  float* var_7_rf__reduce_init = _var_7_rf_temp_buffer;
  var_7_rf__reduce_init[0ll] = 0.00000000f;
  for (int32_t reduce_k_0_3 = 0; reduce_k_0_3 < 5; reduce_k_0_3 += 1) {
    float var_1_local_1 = var_1[((((((((int)blockIdx.x / (S5 * S6)) * 320ll) + (reduce_k_0_3 * 64ll)) + (int)threadIdx.x) * S5) * S6) + ((int)blockIdx.x % (S5 * S6)))];
    float var_local_5 = var[((reduce_k_0_3 * 64ll) + (int)threadIdx.x)];
    var_7_rf[0ll] = (var_7_rf[0ll] + (var_1_local_1 + var_local_5));
  };
  var_7[0ll] = cinn_block_reduce_sum_fp32(var_7_rf[0ll], shm32__fp32_reduce, false);
  var_16_loopalign_1_rf__reduce_init[0ll] = 0.00000000f;
  for (int32_t reduce_k_0_4 = 0; reduce_k_0_4 < 5; reduce_k_0_4 += 1) {
    float var_1_local_2 = var_1[(((((reduce_k_0_4 * 64ll) + (int)threadIdx.x) * S5) * S6) + ((int)blockIdx.x % (S5 * S6)))];
    float var_local_6 = var[((reduce_k_0_4 * 64ll) + (int)threadIdx.x)];
    var_16_loopalign_1_rf[0ll] = (var_16_loopalign_1_rf[0ll] + (((var_1_local_2 + var_local_6) - (var_7[0ll] / 320.000000f)) * ((var_1_local_2 + var_local_6) - (var_7[0ll] / 320.000000f))));
  };
  var_16_loopalign_1[0ll] = cinn_block_reduce_sum_fp32(var_16_loopalign_1_rf[0ll], shm32__fp32_reduce, false);
  for (int32_t k_5 = 0; k_5 < 5; k_5 += 1) {
    float var_1_local_3 = var_1[((((((((int)blockIdx.x / (S5 * S6)) * 320ll) + (k_5 * 64ll)) + (int)threadIdx.x) * S5) * S6) + ((int)blockIdx.x % (S5 * S6)))];
    float var_local_7 = var[((k_5 * 64) + (int)threadIdx.x)];
    float var_30_local = var_30[((k_5 * 64) + (int)threadIdx.x)];
    float var_34_local = var_34[((k_5 * 64) + (int)threadIdx.x)];
    var_36[(((k_5 * 64ll) + (int)threadIdx.x) + ((int)blockIdx.x * 320ll))] = (((((var_1_local_3 + var_local_7) - (var_7[0ll] / 320.000000f)) * cinn_nvgpu_rsqrt_fp32(((var_16_loopalign_1[0ll] / 320.000000f) + 9.99999975e-06f))) * var_30_local) + var_34_local);
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA__FPA_S3GE1ll_BPA_AND_FPA_S3LE2048ll_BPA__BPA_AND_FPA__FPA_S3MUL10240ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int32_t S3)
{
  __builtin_assume(((int)blockIdx.x < 10240));
  __builtin_assume(((int)threadIdx.x < 256));
  float _var_1_rf_temp_buffer [ 1 ];
  float _var_1_temp_buffer [ 1 ];
  float _var_6_loopalign_1_rf_temp_buffer [ 1 ];
  float _var_6_loopalign_1_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_1 = _var_1_temp_buffer;
  float* var_1_rf = _var_1_rf_temp_buffer;
  float* var_1_rf__reduce_init = _var_1_rf_temp_buffer;
  float* var_6_loopalign_1 = _var_6_loopalign_1_temp_buffer;
  float* var_6_loopalign_1_rf = _var_6_loopalign_1_rf_temp_buffer;
  float* var_6_loopalign_1_rf__reduce_init = _var_6_loopalign_1_rf_temp_buffer;
  var_1_rf__reduce_init[0] = -3.40282347e+38f;
  for (int32_t reduce_k_0_0 = 0; reduce_k_0_0 < ((S3 / 256) + 1); reduce_k_0_0 += 1) {
    int64_t reduce_k_0_0_strided = (reduce_k_0_0 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_0_strided))) < S3), 256);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_0_strided))) < S3)) {
      float var_local_8 = var[(((int)threadIdx.x + reduce_k_0_0_strided) + (S3 * (int)blockIdx.x))];
      var_1_rf[0] = max(var_1_rf[0], (0.125000000f * var_local_8));
    };
  };
  var_1[0] = cinn_block_reduce_max_fp32(var_1_rf[0], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_2 = 0; reduce_k_0_2 < ((S3 / 256) + 1); reduce_k_0_2 += 1) {
    int64_t reduce_k_0_2_strided = (reduce_k_0_2 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_2_strided))) < S3), 256);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_2_strided))) < S3)) {
      float var_local_9 = var[(((((int)blockIdx.x % 10240) * S3) + (int)threadIdx.x) + reduce_k_0_2_strided)];
      var_6_loopalign_1_rf[0] = (var_6_loopalign_1_rf[0] + cinn_nvgpu_exp_fp32(((0.125000000f * var_local_9) - var_1[0])));
    };
  };
  var_6_loopalign_1[0] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf[0], shm32__fp32_reduce, false);
  for (int32_t a = 0; a < ((S3 / 256) + 1); a += 1) {
    int64_t a_strided = (a * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(a_strided, ((((int32_t)(a_strided)) + (int)threadIdx.x) < S3), 256);
    if (((((int32_t)(a_strided)) + (int)threadIdx.x) < S3)) {
      float var_local_10 = var[((a_strided + (int)threadIdx.x) + (S3 * (int)blockIdx.x))];
      var_9[((a_strided + (int)threadIdx.x) + (S3 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.125000000f * var_local_10) - var_1[0])) / var_6_loopalign_1[0]);
    };
  };
}__global__
void __launch_bounds__(256) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA__FPA_S3GE1ll_BPA_AND_FPA_S3LE2048ll_BPA__BPA_AND_FPA__FPA_S3MUL10240ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int64_t S3)
{
  __builtin_assume(((int)blockIdx.x < 10240ll));
  __builtin_assume(((int)threadIdx.x < 256ll));
  float _var_1_rf_temp_buffer [ 1 ];
  float _var_1_temp_buffer [ 1 ];
  float _var_6_loopalign_1_rf_temp_buffer [ 1 ];
  float _var_6_loopalign_1_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_1 = _var_1_temp_buffer;
  float* var_1_rf = _var_1_rf_temp_buffer;
  float* var_1_rf__reduce_init = _var_1_rf_temp_buffer;
  float* var_6_loopalign_1 = _var_6_loopalign_1_temp_buffer;
  float* var_6_loopalign_1_rf = _var_6_loopalign_1_rf_temp_buffer;
  float* var_6_loopalign_1_rf__reduce_init = _var_6_loopalign_1_rf_temp_buffer;
  var_1_rf__reduce_init[0ll] = -3.40282347e+38f;
  for (int32_t reduce_k_0_0 = 0ll; reduce_k_0_0 < ((S3 / 256ll) + 1ll); reduce_k_0_0 += 1) {
    int64_t reduce_k_0_0_strided = (reduce_k_0_0 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_strided, (((int)threadIdx.x + reduce_k_0_0_strided) < S3), 256ll);
    if ((((int)threadIdx.x + reduce_k_0_0_strided) < S3)) {
      float var_local_8 = var[(((int)threadIdx.x + reduce_k_0_0_strided) + (S3 * (int)blockIdx.x))];
      var_1_rf[0ll] = max(var_1_rf[0ll], (0.125000000f * var_local_8));
    };
  };
  var_1[0ll] = cinn_block_reduce_max_fp32(var_1_rf[0ll], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf__reduce_init[0ll] = 0.00000000f;
  for (int32_t reduce_k_0_2 = 0ll; reduce_k_0_2 < ((S3 / 256ll) + 1ll); reduce_k_0_2 += 1) {
    int64_t reduce_k_0_2_strided = (reduce_k_0_2 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_strided, (((int)threadIdx.x + reduce_k_0_2_strided) < S3), 256ll);
    if ((((int)threadIdx.x + reduce_k_0_2_strided) < S3)) {
      float var_local_9 = var[(((((int)blockIdx.x % 10240ll) * S3) + (int)threadIdx.x) + reduce_k_0_2_strided)];
      var_6_loopalign_1_rf[0ll] = (var_6_loopalign_1_rf[0ll] + cinn_nvgpu_exp_fp32(((0.125000000f * var_local_9) - var_1[0ll])));
    };
  };
  var_6_loopalign_1[0ll] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf[0ll], shm32__fp32_reduce, false);
  for (int32_t a = 0ll; a < ((S3 / 256ll) + 1ll); a += 1) {
    int64_t a_strided = (a * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(a_strided, ((a_strided + (int)threadIdx.x) < S3), 256ll);
    if (((a_strided + (int)threadIdx.x) < S3)) {
      float var_local_10 = var[((a_strided + (int)threadIdx.x) + (S3 * (int)blockIdx.x))];
      var_9[((a_strided + (int)threadIdx.x) + (S3 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.125000000f * var_local_10) - var_1[0ll])) / var_6_loopalign_1[0ll]);
    };
  };
}__global__
void __launch_bounds__(1024) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA_S3GE2049ll_BPA_AND_FPA__FPA_S3MUL10240ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int32_t S3)
{
  __builtin_assume(((int)blockIdx.x < 10240));
  __builtin_assume(((int)threadIdx.x < 1024));
  float _var_1_rf_0_temp_buffer [ 1 ];
  float _var_1_temp_buffer [ 1 ];
  float _var_6_loopalign_1_rf_0_temp_buffer [ 1 ];
  float _var_6_loopalign_1_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_1 = _var_1_temp_buffer;
  float* var_1_rf_0 = _var_1_rf_0_temp_buffer;
  float* var_1_rf_0__reduce_init = _var_1_rf_0_temp_buffer;
  float* var_6_loopalign_1 = _var_6_loopalign_1_temp_buffer;
  float* var_6_loopalign_1_rf_0 = _var_6_loopalign_1_rf_0_temp_buffer;
  float* var_6_loopalign_1_rf_0__reduce_init = _var_6_loopalign_1_rf_0_temp_buffer;
  var_1_rf_0__reduce_init[0] = -3.40282347e+38f;
  for (int32_t reduce_k_0_0_1 = 0; reduce_k_0_0_1 < ((S3 / 1024) + 1); reduce_k_0_0_1 += 1) {
    int64_t reduce_k_0_0_1_strided = (reduce_k_0_0_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_1_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_0_1_strided))) < S3), 1024);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_0_1_strided))) < S3)) {
      float var_local_12 = var[(((int)threadIdx.x + reduce_k_0_0_1_strided) + (S3 * (int)blockIdx.x))];
      var_1_rf_0[0] = max(var_1_rf_0[0], (0.125000000f * var_local_12));
    };
  };
  var_1[0] = cinn_block_reduce_max_fp32(var_1_rf_0[0], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf_0__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_2_1 = 0; reduce_k_0_2_1 < ((S3 / 1024) + 1); reduce_k_0_2_1 += 1) {
    int64_t reduce_k_0_2_1_strided = (reduce_k_0_2_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_1_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_2_1_strided))) < S3), 1024);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_2_1_strided))) < S3)) {
      float var_local_13 = var[(((((int)blockIdx.x % 10240) * S3) + (int)threadIdx.x) + reduce_k_0_2_1_strided)];
      var_6_loopalign_1_rf_0[0] = (var_6_loopalign_1_rf_0[0] + cinn_nvgpu_exp_fp32(((0.125000000f * var_local_13) - var_1[0])));
    };
  };
  var_6_loopalign_1[0] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf_0[0], shm32__fp32_reduce, false);
  for (int32_t a_5 = 0; a_5 < ((S3 / 1024) + 1); a_5 += 1) {
    int64_t a_5_strided = (a_5 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(a_5_strided, ((((int32_t)(a_5_strided)) + (int)threadIdx.x) < S3), 1024);
    if (((((int32_t)(a_5_strided)) + (int)threadIdx.x) < S3)) {
      float var_local_14 = var[((a_5_strided + (int)threadIdx.x) + (S3 * (int)blockIdx.x))];
      var_9[((a_5_strided + (int)threadIdx.x) + (S3 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.125000000f * var_local_14) - var_1[0])) / var_6_loopalign_1[0]);
    };
  };
}__global__
void __launch_bounds__(1024) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA_S3GE2049ll_BPA_AND_FPA__FPA_S3MUL10240ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int64_t S3)
{
  __builtin_assume(((int)blockIdx.x < 10240ll));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  float _var_1_rf_0_temp_buffer [ 1 ];
  float _var_1_temp_buffer [ 1 ];
  float _var_6_loopalign_1_rf_0_temp_buffer [ 1 ];
  float _var_6_loopalign_1_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_1 = _var_1_temp_buffer;
  float* var_1_rf_0 = _var_1_rf_0_temp_buffer;
  float* var_1_rf_0__reduce_init = _var_1_rf_0_temp_buffer;
  float* var_6_loopalign_1 = _var_6_loopalign_1_temp_buffer;
  float* var_6_loopalign_1_rf_0 = _var_6_loopalign_1_rf_0_temp_buffer;
  float* var_6_loopalign_1_rf_0__reduce_init = _var_6_loopalign_1_rf_0_temp_buffer;
  var_1_rf_0__reduce_init[0ll] = -3.40282347e+38f;
  for (int32_t reduce_k_0_0_1 = 0ll; reduce_k_0_0_1 < ((S3 / 1024ll) + 1ll); reduce_k_0_0_1 += 1) {
    int64_t reduce_k_0_0_1_strided = (reduce_k_0_0_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_1_strided, (((int)threadIdx.x + reduce_k_0_0_1_strided) < S3), 1024ll);
    if ((((int)threadIdx.x + reduce_k_0_0_1_strided) < S3)) {
      float var_local_12 = var[(((int)threadIdx.x + reduce_k_0_0_1_strided) + (S3 * (int)blockIdx.x))];
      var_1_rf_0[0ll] = max(var_1_rf_0[0ll], (0.125000000f * var_local_12));
    };
  };
  var_1[0ll] = cinn_block_reduce_max_fp32(var_1_rf_0[0ll], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf_0__reduce_init[0ll] = 0.00000000f;
  for (int32_t reduce_k_0_2_1 = 0ll; reduce_k_0_2_1 < ((S3 / 1024ll) + 1ll); reduce_k_0_2_1 += 1) {
    int64_t reduce_k_0_2_1_strided = (reduce_k_0_2_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_1_strided, (((int)threadIdx.x + reduce_k_0_2_1_strided) < S3), 1024ll);
    if ((((int)threadIdx.x + reduce_k_0_2_1_strided) < S3)) {
      float var_local_13 = var[(((((int)blockIdx.x % 10240ll) * S3) + (int)threadIdx.x) + reduce_k_0_2_1_strided)];
      var_6_loopalign_1_rf_0[0ll] = (var_6_loopalign_1_rf_0[0ll] + cinn_nvgpu_exp_fp32(((0.125000000f * var_local_13) - var_1[0ll])));
    };
  };
  var_6_loopalign_1[0ll] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf_0[0ll], shm32__fp32_reduce, false);
  for (int32_t a_5 = 0ll; a_5 < ((S3 / 1024ll) + 1ll); a_5 += 1) {
    int64_t a_5_strided = (a_5 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(a_5_strided, ((a_5_strided + (int)threadIdx.x) < S3), 1024ll);
    if (((a_5_strided + (int)threadIdx.x) < S3)) {
      float var_local_14 = var[((a_5_strided + (int)threadIdx.x) + (S3 * (int)blockIdx.x))];
      var_9[((a_5_strided + (int)threadIdx.x) + (S3 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.125000000f * var_local_14) - var_1[0ll])) / var_6_loopalign_1[0ll]);
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_slice__COND__FPA__FPA__FPA__FPA_S2MUL320ll_BPA_GE1ll_BPA_AND_FPA__FPA_S2MUL320ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S2MUL640ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (S2 * 320)));
  var_0[(((((((int)blockIdx.x % (S2 * 320)) / (S2 * 64)) + (((int)blockIdx.x / (S2 * 320)) * 5)) * S2) * 64) + ((int)blockIdx.x % (S2 * 64)))] = var[((((((((int)blockIdx.x % (S2 * 320)) / (S2 * 64)) + (((int)blockIdx.x / (S2 * 320)) * 5)) + 5) * S2) * 64) + ((int)blockIdx.x % (S2 * 64)))];
}__global__
void __launch_bounds__(1) fn_slice__COND__FPA__FPA__FPA__FPA_S2MUL320ll_BPA_GE1ll_BPA_AND_FPA__FPA_S2MUL320ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S2MUL640ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (S2 * 320ll)));
  var_0[(((((((int)blockIdx.x % (S2 * 320ll)) / (S2 * 64ll)) + (((int)blockIdx.x / (S2 * 320ll)) * 5ll)) * S2) * 64ll) + ((int)blockIdx.x % (S2 * 64ll)))] = var[((((((((int)blockIdx.x % (S2 * 320ll)) / (S2 * 64ll)) + (((int)blockIdx.x / (S2 * 320ll)) * 5ll)) + 5ll) * S2) * 64ll) + ((int)blockIdx.x % (S2 * 64ll)))];
}__global__
void __launch_bounds__(1024) fn_slice__COND__FPA__FPA__FPA_S2MUL320ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S2MUL640ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((S2 * 320) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_0 = 0; i_j_k_a_fused_0 < 4; i_j_k_a_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) < (S2 * 320))) {
      float var_local_11 = var[((((((((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) % (S2 * 320)) / (S2 * 64)) + (((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) / (S2 * 320)) * 5)) + 5) * S2) * 64) + ((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) % (S2 * 64)))];
      var_0[(((((((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) % (S2 * 320)) / (S2 * 64)) + (((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) / (S2 * 320)) * 5)) * S2) * 64) + ((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) % (S2 * 64)))] = var_local_11;
    };
  };
}__global__
void __launch_bounds__(1024) fn_slice__COND__FPA__FPA__FPA_S2MUL320ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S2MUL640ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((S2 * 320ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_fused_0 = 0ll; i_j_k_a_fused_0 < 4ll; i_j_k_a_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) < (S2 * 320ll))) {
      float var_local_11 = var[((((((((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) % (S2 * 320ll)) / (S2 * 64ll)) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) / (S2 * 320ll)) * 5ll)) + 5ll) * S2) * 64ll) + ((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) % (S2 * 64ll)))];
      var_0[(((((((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) % (S2 * 320ll)) / (S2 * 64ll)) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) / (S2 * 320ll)) * 5ll)) * S2) * 64ll) + ((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) % (S2 * 64ll)))] = var_local_11;
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_transpose_gs_reshape__COND__FPA__FPA__FPA__FPA_S0MULS1_BPA_MUL320ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S0MULS1_BPA_MUL320ll_BPA_LE1023ll_BPA__BPA___kernel(const float* __restrict__ var, const int32_t* __restrict__ var_1, const int32_t* __restrict__ var_2, float* __restrict__ var_4, int32_t S0, int32_t S1)
{
  __builtin_assume(((int)blockIdx.x < ((S0 * S1) * 320)));
  var_4[(int)blockIdx.x] = var[((((((int)blockIdx.x / 655360) * 2048) + ((int)blockIdx.x & 2047)) * 320) + (((int)blockIdx.x % 655360) / 2048))];
}__global__
void __launch_bounds__(1024) fn_transpose_gs_reshape__COND__FPA__FPA__FPA_S0MULS1_BPA_MUL320ll_BPA_GE1024ll_BPA___kernel(const float* __restrict__ var, const int32_t* __restrict__ var_1, const int32_t* __restrict__ var_2, float* __restrict__ var_4, int32_t S0, int32_t S1)
{
  __builtin_assume(((int)blockIdx.x < ((((S0 * S1) * 320) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_6 = 0; i_j_k_a_fused_6 < 4; i_j_k_a_fused_6 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) < ((S0 * S1) * 320))) {
      float var_local_15 = var[((((((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) / 655360) * 2048) + ((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) & 2047)) * 320) + (((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) % 655360) / 2048))];
      var_4[(((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x)] = var_local_15;
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA__FPA_S4MUL640ll_BPA_GE1ll_BPA_AND_FPA__FPA_S4MUL640ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S4MUL640ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int32_t S4)
{
  __builtin_assume(((int)blockIdx.x < (S4 * 640)));
  var_5[(((((((int)blockIdx.x % (S4 * 320)) / (S4 * 64)) + (((int)blockIdx.x / (S4 * 320)) * 5)) * S4) * 64) + ((int)blockIdx.x % (S4 * 64)))] = (var[(((((((int)blockIdx.x % (S4 * 320)) / (S4 * 64)) * 64) + (((int)blockIdx.x / (S4 * 320)) * 320)) + ((int)blockIdx.x & 63)) + ((((int)blockIdx.x % (S4 * 64)) / 64) * 640))] + var_1[((((((int)blockIdx.x % (S4 * 320)) / (S4 * 64)) * 64) + (((int)blockIdx.x / (S4 * 320)) * 320)) + ((int)blockIdx.x & 63))]);
  if ((((int)blockIdx.x / (S4 * 320)) == 0)) {
    var_7[(((((((int)blockIdx.x % (S4 * 320)) / (S4 * 64)) * 64) + ((int)blockIdx.x & 63)) * S4) + (((int)blockIdx.x % (S4 * 64)) / 64))] = (var[((((((int)blockIdx.x % (S4 * 320)) / (S4 * 64)) * 64) + ((int)blockIdx.x & 63)) + ((((int)blockIdx.x % (S4 * 64)) / 64) * 640))] + var_1[(((((int)blockIdx.x % (S4 * 320)) / (S4 * 64)) * 64) + ((int)blockIdx.x & 63))]);
  };
}__global__
void __launch_bounds__(1) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA__FPA_S4MUL640ll_BPA_GE1ll_BPA_AND_FPA__FPA_S4MUL640ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S4MUL640ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int64_t S4)
{
  __builtin_assume(((int)blockIdx.x < (S4 * 640ll)));
  var_5[(((((((int)blockIdx.x % (S4 * 320ll)) / (S4 * 64ll)) + (((int)blockIdx.x / (S4 * 320ll)) * 5ll)) * S4) * 64ll) + ((int)blockIdx.x % (S4 * 64ll)))] = (var[(((((((int)blockIdx.x % (S4 * 320ll)) / (S4 * 64ll)) * 64ll) + (((int)blockIdx.x / (S4 * 320ll)) * 320ll)) + ((int)blockIdx.x & 63)) + ((((int)blockIdx.x % (S4 * 64ll)) / 64ll) * 640ll))] + var_1[((((((int)blockIdx.x % (S4 * 320ll)) / (S4 * 64ll)) * 64ll) + (((int)blockIdx.x / (S4 * 320ll)) * 320ll)) + ((int)blockIdx.x & 63))]);
  if ((((int)blockIdx.x / (S4 * 320ll)) == 0ll)) {
    var_7[(((((((int)blockIdx.x % (S4 * 320ll)) / (S4 * 64ll)) * 64ll) + ((int)blockIdx.x & 63)) * S4) + (((int)blockIdx.x % (S4 * 64ll)) / 64ll))] = (var[((((((int)blockIdx.x % (S4 * 320ll)) / (S4 * 64ll)) * 64ll) + ((int)blockIdx.x & 63)) + ((((int)blockIdx.x % (S4 * 64ll)) / 64ll) * 640ll))] + var_1[(((((int)blockIdx.x % (S4 * 320ll)) / (S4 * 64ll)) * 64ll) + ((int)blockIdx.x & 63))]);
  };
}__global__
void __launch_bounds__(1024) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA_S4MUL640ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S4MUL640ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int32_t S4)
{
  __builtin_assume(((int)blockIdx.x < (((S4 * 640) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_b_fused_0 = 0; i_j_k_a_b_fused_0 < 4; i_j_k_a_b_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) < (S4 * 640))) {
      float var_local_16 = var[(((((((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) % (S4 * 320)) / (S4 * 64)) * 64) + (((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) / (S4 * 320)) * 320)) + ((int)threadIdx.x & 63)) + ((((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) % (S4 * 64)) / 64) * 640))];
      float var_1_local_4 = var_1[((((((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) % (S4 * 320)) / (S4 * 64)) * 64) + (((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) / (S4 * 320)) * 320)) + ((int)threadIdx.x & 63))];
      var_5[(((((((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) % (S4 * 320)) / (S4 * 64)) + (((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) / (S4 * 320)) * 5)) * S4) * 64) + ((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) % (S4 * 64)))] = (var_local_16 + var_1_local_4);
    };
  };
  for (int32_t append_var_0_i_j_a_k_fused_0 = 0; append_var_0_i_j_a_k_fused_0 < 4; append_var_0_i_j_a_k_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) < (S4 * 640))) {
      if ((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) / (S4 * 320)) == 0)) {
        var_7[(((((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S4 * 320)) / (S4 * 64)) * 64) + ((int)threadIdx.x & 63)) * S4) + (((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S4 * 64)) / 64))] = (var[((((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S4 * 320)) / (S4 * 64)) * 64) + ((int)threadIdx.x & 63)) + ((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S4 * 64)) / 64) * 640))] + var_1[(((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S4 * 320)) / (S4 * 64)) * 64) + ((int)threadIdx.x & 63))]);
      };
    };
  };
}__global__
void __launch_bounds__(1024) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA_S4MUL640ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S4MUL640ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int64_t S4)
{
  __builtin_assume(((int)blockIdx.x < (((S4 * 640ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_b_fused_0 = 0ll; i_j_k_a_b_fused_0 < 4ll; i_j_k_a_b_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) < (S4 * 640ll))) {
      float var_local_16 = var[(((((((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) % (S4 * 320ll)) / (S4 * 64ll)) * 64ll) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) / (S4 * 320ll)) * 320ll)) + ((int)threadIdx.x & 63)) + ((((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) % (S4 * 64ll)) / 64ll) * 640ll))];
      float var_1_local_4 = var_1[((((((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) % (S4 * 320ll)) / (S4 * 64ll)) * 64ll) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) / (S4 * 320ll)) * 320ll)) + ((int)threadIdx.x & 63))];
      var_5[(((((((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) % (S4 * 320ll)) / (S4 * 64ll)) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) / (S4 * 320ll)) * 5ll)) * S4) * 64ll) + ((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) % (S4 * 64ll)))] = (var_local_16 + var_1_local_4);
    };
  };
  for (int32_t append_var_0_i_j_a_k_fused_0 = 0ll; append_var_0_i_j_a_k_fused_0 < 4ll; append_var_0_i_j_a_k_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) < (S4 * 640ll))) {
      if ((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) / (S4 * 320ll)) == 0ll)) {
        var_7[(((((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S4 * 320ll)) / (S4 * 64ll)) * 64ll) + ((int)threadIdx.x & 63)) * S4) + (((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S4 * 64ll)) / 64ll))] = (var[((((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S4 * 320ll)) / (S4 * 64ll)) * 64ll) + ((int)threadIdx.x & 63)) + ((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S4 * 64ll)) / 64ll) * 640ll))] + var_1[(((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S4 * 320ll)) / (S4 * 64ll)) * 64ll) + ((int)threadIdx.x & 63))]);
      };
    };
  };
}

}
