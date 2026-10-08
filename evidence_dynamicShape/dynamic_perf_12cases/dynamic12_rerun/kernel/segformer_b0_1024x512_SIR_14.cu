
extern "C" {

__global__
void __launch_bounds__(256) fn_transpose_reshape__COND_true__kernel(const float* __restrict__ var, float* __restrict__ var_1)
{
  __builtin_assume(((int)blockIdx.x < 512));
  __builtin_assume(((int)threadIdx.x < 256));
  for (int32_t i_j_k_fused_0 = 0; i_j_k_fused_0 < 4; i_j_k_fused_0 += 1) {
    float var_local = var[((((((((int)blockIdx.x * 1024) + (i_j_k_fused_0 * 256)) + (int)threadIdx.x) / 64) + ((((int)threadIdx.x & 63) / 32) * 8192)) * 32) + ((int)threadIdx.x & 31))];
    var_1[((((int)blockIdx.x * 1024) + (i_j_k_fused_0 * 256)) + (int)threadIdx.x)] = var_local;
  };
}

}

extern "C" {

__global__
void __launch_bounds__(128) fn_bc_add__COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_2)
{
  __builtin_assume(((int)blockIdx.x < 1024));
  __builtin_assume(((int)threadIdx.x < 64));
  __builtin_assume(((int)threadIdx.y < 2));
  for (int32_t i_j_fused_k_fused_0 = 0; i_j_fused_k_fused_0 < 4; i_j_fused_k_fused_0 += 1) {
    float var_1_local = var_1[(((((int)blockIdx.x * 512) + (i_j_fused_k_fused_0 * 128)) + ((int)threadIdx.y * 64)) + (int)threadIdx.x)];
    float var_local_0 = var[((int)threadIdx.x & 63)];
    var_2[(((((int)blockIdx.x * 512) + (i_j_fused_k_fused_0 * 128)) + ((int)threadIdx.y * 64)) + (int)threadIdx.x)] = (var_1_local + var_local_0);
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_bc_add_reshape_transpose__COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_4)
{
  __builtin_assume(((int)blockIdx.x < 512));
  __builtin_assume(((int)threadIdx.x < 256));
  for (int32_t i_j_k_a_fused_3 = 0; i_j_k_a_fused_3 < 4; i_j_k_a_fused_3 += 1) {
    float var_1_local_0 = var_1[((((((((int)blockIdx.x * 1024) + (i_j_k_a_fused_3 * 256)) + (int)threadIdx.x) / 262144) * 32) + ((int)threadIdx.x & 31)) + (((((((int)blockIdx.x * 1024) + (i_j_k_a_fused_3 * 256)) + (int)threadIdx.x) & 262143) / 32) * 64))];
    float var_local_1 = var[(((((((int)blockIdx.x * 1024) + (i_j_k_a_fused_3 * 256)) + (int)threadIdx.x) / 262144) * 32) + ((int)threadIdx.x & 31))];
    var_4[((((int)blockIdx.x * 1024) + (i_j_k_a_fused_3 * 256)) + (int)threadIdx.x)] = (var_1_local_0 + var_local_1);
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_sum_full_div_bc_sub_mul_sum_full_div_full_add_rsqrt_bc_mul_bc_mul_bc_add__COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_14, const float* __restrict__ var_17, float* __restrict__ var_19)
{
  __builtin_assume(((int)blockIdx.x < 1024));
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
  for (int32_t reduce_k_0_5 = 0; reduce_k_0_5 < 2; reduce_k_0_5 += 1) {
    float var_local_2 = var[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 64) + (reduce_k_0_5 * 32)) + (int)threadIdx.x)];
    var_0_rf[0] = (var_0_rf[0] + var_local_2);
  };
  var_0[0] = cinn_block_reduce_sum_fp32(var_0_rf[0], shm32__fp32_reduce, true);
  var_6_rf__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_1 = 0; reduce_k_0_1 < 2; reduce_k_0_1 += 1) {
    float var_local_3 = var[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 64) + (reduce_k_0_1 * 32)) + (int)threadIdx.x)];
    var_6_rf[0] = (var_6_rf[0] + ((var_local_3 - (var_0[0] / 64.0000000f)) * (var_local_3 - (var_0[0] / 64.0000000f))));
  };
  var_6[0] = cinn_block_reduce_sum_fp32(var_6_rf[0], shm32__fp32_reduce, true);
  for (int32_t k = 0; k < 2; k += 1) {
    float var_local_4 = var[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 64) + (k * 32)) + (int)threadIdx.x)];
    float var_14_local = var_14[((k * 32) + (int)threadIdx.x)];
    float var_17_local = var_17[((k * 32) + (int)threadIdx.x)];
    var_19[((((((int)blockIdx.x * 8) + (int)threadIdx.y) * 64) + (k * 32)) + (int)threadIdx.x)] = ((((var_local_4 - (var_0[0] / 64.0000000f)) * cinn_nvgpu_rsqrt_fp32(((var_6[0] / 64.0000000f) + 9.99999997e-07f))) * var_14_local) + var_17_local);
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA__FPA_S6GE1ll_BPA_AND_FPA_S6LE2048ll_BPA__BPA_AND_FPA__FPA_S6MUL16384ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int32_t S6)
{
  __builtin_assume(((int)blockIdx.x < 16384));
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
  for (int32_t reduce_k_0_0 = 0; reduce_k_0_0 < ((S6 / 256) + 1); reduce_k_0_0 += 1) {
    int64_t reduce_k_0_0_strided = (reduce_k_0_0 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_0_strided))) < S6), 256);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_0_strided))) < S6)) {
      float var_local_5 = var[(((int)threadIdx.x + reduce_k_0_0_strided) + (S6 * (int)blockIdx.x))];
      var_1_rf[0] = max(var_1_rf[0], (0.176776692f * var_local_5));
    };
  };
  var_1[0] = cinn_block_reduce_max_fp32(var_1_rf[0], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_2 = 0; reduce_k_0_2 < ((S6 / 256) + 1); reduce_k_0_2 += 1) {
    int64_t reduce_k_0_2_strided = (reduce_k_0_2 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_2_strided))) < S6), 256);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_2_strided))) < S6)) {
      float var_local_6 = var[(((((int)blockIdx.x & 16383) * S6) + (int)threadIdx.x) + reduce_k_0_2_strided)];
      var_6_loopalign_1_rf[0] = (var_6_loopalign_1_rf[0] + cinn_nvgpu_exp_fp32(((0.176776692f * var_local_6) - var_1[0])));
    };
  };
  var_6_loopalign_1[0] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf[0], shm32__fp32_reduce, false);
  for (int32_t a = 0; a < ((S6 / 256) + 1); a += 1) {
    int64_t a_strided = (a * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(a_strided, ((((int32_t)(a_strided)) + (int)threadIdx.x) < S6), 256);
    if (((((int32_t)(a_strided)) + (int)threadIdx.x) < S6)) {
      float var_local_7 = var[((a_strided + (int)threadIdx.x) + (S6 * (int)blockIdx.x))];
      var_9[((a_strided + (int)threadIdx.x) + (S6 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.176776692f * var_local_7) - var_1[0])) / var_6_loopalign_1[0]);
    };
  };
}__global__
void __launch_bounds__(256) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA__FPA_S6GE1ll_BPA_AND_FPA_S6LE2048ll_BPA__BPA_AND_FPA__FPA_S6MUL16384ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int64_t S6)
{
  __builtin_assume(((int)blockIdx.x < 16384ll));
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
  for (int32_t reduce_k_0_0 = 0ll; reduce_k_0_0 < ((S6 / 256ll) + 1ll); reduce_k_0_0 += 1) {
    int64_t reduce_k_0_0_strided = (reduce_k_0_0 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_strided, (((int)threadIdx.x + reduce_k_0_0_strided) < S6), 256ll);
    if ((((int)threadIdx.x + reduce_k_0_0_strided) < S6)) {
      float var_local_5 = var[(((int)threadIdx.x + reduce_k_0_0_strided) + (S6 * (int)blockIdx.x))];
      var_1_rf[0ll] = max(var_1_rf[0ll], (0.176776692f * var_local_5));
    };
  };
  var_1[0ll] = cinn_block_reduce_max_fp32(var_1_rf[0ll], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf__reduce_init[0ll] = 0.00000000f;
  for (int32_t reduce_k_0_2 = 0ll; reduce_k_0_2 < ((S6 / 256ll) + 1ll); reduce_k_0_2 += 1) {
    int64_t reduce_k_0_2_strided = (reduce_k_0_2 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_strided, (((int)threadIdx.x + reduce_k_0_2_strided) < S6), 256ll);
    if ((((int)threadIdx.x + reduce_k_0_2_strided) < S6)) {
      float var_local_6 = var[(((((int)blockIdx.x & 16383) * S6) + (int)threadIdx.x) + reduce_k_0_2_strided)];
      var_6_loopalign_1_rf[0ll] = (var_6_loopalign_1_rf[0ll] + cinn_nvgpu_exp_fp32(((0.176776692f * var_local_6) - var_1[0ll])));
    };
  };
  var_6_loopalign_1[0ll] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf[0ll], shm32__fp32_reduce, false);
  for (int32_t a = 0ll; a < ((S6 / 256ll) + 1ll); a += 1) {
    int64_t a_strided = (a * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(a_strided, ((a_strided + (int)threadIdx.x) < S6), 256ll);
    if (((a_strided + (int)threadIdx.x) < S6)) {
      float var_local_7 = var[((a_strided + (int)threadIdx.x) + (S6 * (int)blockIdx.x))];
      var_9[((a_strided + (int)threadIdx.x) + (S6 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.176776692f * var_local_7) - var_1[0ll])) / var_6_loopalign_1[0ll]);
    };
  };
}__global__
void __launch_bounds__(1024) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA_S6GE2049ll_BPA_AND_FPA__FPA_S6MUL16384ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int32_t S6)
{
  __builtin_assume(((int)blockIdx.x < 16384));
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
  for (int32_t reduce_k_0_0_1 = 0; reduce_k_0_0_1 < ((S6 / 1024) + 1); reduce_k_0_0_1 += 1) {
    int64_t reduce_k_0_0_1_strided = (reduce_k_0_0_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_1_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_0_1_strided))) < S6), 1024);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_0_1_strided))) < S6)) {
      float var_local_8 = var[(((int)threadIdx.x + reduce_k_0_0_1_strided) + (S6 * (int)blockIdx.x))];
      var_1_rf_0[0] = max(var_1_rf_0[0], (0.176776692f * var_local_8));
    };
  };
  var_1[0] = cinn_block_reduce_max_fp32(var_1_rf_0[0], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf_0__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_2_1 = 0; reduce_k_0_2_1 < ((S6 / 1024) + 1); reduce_k_0_2_1 += 1) {
    int64_t reduce_k_0_2_1_strided = (reduce_k_0_2_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_1_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_2_1_strided))) < S6), 1024);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_2_1_strided))) < S6)) {
      float var_local_9 = var[(((((int)blockIdx.x & 16383) * S6) + (int)threadIdx.x) + reduce_k_0_2_1_strided)];
      var_6_loopalign_1_rf_0[0] = (var_6_loopalign_1_rf_0[0] + cinn_nvgpu_exp_fp32(((0.176776692f * var_local_9) - var_1[0])));
    };
  };
  var_6_loopalign_1[0] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf_0[0], shm32__fp32_reduce, false);
  for (int32_t a_5 = 0; a_5 < ((S6 / 1024) + 1); a_5 += 1) {
    int64_t a_5_strided = (a_5 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(a_5_strided, ((((int32_t)(a_5_strided)) + (int)threadIdx.x) < S6), 1024);
    if (((((int32_t)(a_5_strided)) + (int)threadIdx.x) < S6)) {
      float var_local_10 = var[((a_5_strided + (int)threadIdx.x) + (S6 * (int)blockIdx.x))];
      var_9[((a_5_strided + (int)threadIdx.x) + (S6 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.176776692f * var_local_10) - var_1[0])) / var_6_loopalign_1[0]);
    };
  };
}__global__
void __launch_bounds__(1024) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA_S6GE2049ll_BPA_AND_FPA__FPA_S6MUL16384ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int64_t S6)
{
  __builtin_assume(((int)blockIdx.x < 16384ll));
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
  for (int32_t reduce_k_0_0_1 = 0ll; reduce_k_0_0_1 < ((S6 / 1024ll) + 1ll); reduce_k_0_0_1 += 1) {
    int64_t reduce_k_0_0_1_strided = (reduce_k_0_0_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_1_strided, (((int)threadIdx.x + reduce_k_0_0_1_strided) < S6), 1024ll);
    if ((((int)threadIdx.x + reduce_k_0_0_1_strided) < S6)) {
      float var_local_8 = var[(((int)threadIdx.x + reduce_k_0_0_1_strided) + (S6 * (int)blockIdx.x))];
      var_1_rf_0[0ll] = max(var_1_rf_0[0ll], (0.176776692f * var_local_8));
    };
  };
  var_1[0ll] = cinn_block_reduce_max_fp32(var_1_rf_0[0ll], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf_0__reduce_init[0ll] = 0.00000000f;
  for (int32_t reduce_k_0_2_1 = 0ll; reduce_k_0_2_1 < ((S6 / 1024ll) + 1ll); reduce_k_0_2_1 += 1) {
    int64_t reduce_k_0_2_1_strided = (reduce_k_0_2_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_1_strided, (((int)threadIdx.x + reduce_k_0_2_1_strided) < S6), 1024ll);
    if ((((int)threadIdx.x + reduce_k_0_2_1_strided) < S6)) {
      float var_local_9 = var[(((((int)blockIdx.x & 16383) * S6) + (int)threadIdx.x) + reduce_k_0_2_1_strided)];
      var_6_loopalign_1_rf_0[0ll] = (var_6_loopalign_1_rf_0[0ll] + cinn_nvgpu_exp_fp32(((0.176776692f * var_local_9) - var_1[0ll])));
    };
  };
  var_6_loopalign_1[0ll] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf_0[0ll], shm32__fp32_reduce, false);
  for (int32_t a_5 = 0ll; a_5 < ((S6 / 1024ll) + 1ll); a_5 += 1) {
    int64_t a_5_strided = (a_5 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(a_5_strided, ((a_5_strided + (int)threadIdx.x) < S6), 1024ll);
    if (((a_5_strided + (int)threadIdx.x) < S6)) {
      float var_local_10 = var[((a_5_strided + (int)threadIdx.x) + (S6 * (int)blockIdx.x))];
      var_9[((a_5_strided + (int)threadIdx.x) + (S6 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.176776692f * var_local_10) - var_1[0ll])) / var_6_loopalign_1[0ll]);
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_reshape_gs_bc_add_reshape_transpose_sum_full_gs_bc_div_gs_bc_sub_mul_sum_full_gs_bc_div_full_gs_bc_add_rsqrt_gs_bc_mul_gs_bc_mul_gs_bc_add__COND__FPA__FPA__FPA_S4MULS5_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S4MULS5_BPA_MUL64ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_30, const float* __restrict__ var_34, float* __restrict__ var_36, int32_t S4, int32_t S5)
{
  __builtin_assume(((int)blockIdx.x < (((S4 * S5) / 32) + 1)));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 8));
  float _var_16_loopalign_1_rf_temp_buffer [ 4ll ];
  float _var_16_loopalign_1_temp_buffer [ 4ll ];
  float _var_7_rf_temp_buffer [ 4ll ];
  float _var_7_temp_buffer [ 4ll ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_16_loopalign_1 = _var_16_loopalign_1_temp_buffer;
  float* var_16_loopalign_1_rf = _var_16_loopalign_1_rf_temp_buffer;
  float* var_16_loopalign_1_rf__reduce_init = _var_16_loopalign_1_rf_temp_buffer;
  float* var_7 = _var_7_temp_buffer;
  float* var_7_rf = _var_7_rf_temp_buffer;
  float* var_7_rf__reduce_init = _var_7_rf_temp_buffer;
  for (int32_t i_j_k_fused_5 = 0; i_j_k_fused_5 < 4; i_j_k_fused_5 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_fused_5) * 8) + (int)threadIdx.y) < (S4 * S5))) {
      var_7_rf__reduce_init[i_j_k_fused_5] = 0.00000000f;
      for (int32_t reduce_k_0_3 = 0; reduce_k_0_3 < 2; reduce_k_0_3 += 1) {
        float var_1_local_1 = var_1[((((((((((((int)blockIdx.x * 4) + i_j_k_fused_5) * 8) + (int)threadIdx.y) / (S4 * S5)) * 64) + (reduce_k_0_3 * 32)) + (int)threadIdx.x) * S4) * S5) + ((((((int)blockIdx.x * 4) + i_j_k_fused_5) * 8) + (int)threadIdx.y) % (S4 * S5)))];
        float var_local_12 = var[((reduce_k_0_3 * 32) + (int)threadIdx.x)];
        var_7_rf[i_j_k_fused_5] = (var_7_rf[i_j_k_fused_5] + (var_1_local_1 + var_local_12));
      };
    };
  };
  for (int32_t i_j_k_fused_5 = 0; i_j_k_fused_5 < 4; i_j_k_fused_5 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_fused_5) * 8) + (int)threadIdx.y) < (S4 * S5))) {
      var_7[i_j_k_fused_5] = cinn_block_reduce_sum_fp32(var_7_rf[i_j_k_fused_5], shm32__fp32_reduce, true);
    };
  };
  for (int32_t i_j_append_var_1_fused_0 = 0; i_j_append_var_1_fused_0 < 4; i_j_append_var_1_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_append_var_1_fused_0) * 8) + (int)threadIdx.y) < (S4 * S5))) {
      var_16_loopalign_1_rf__reduce_init[i_j_append_var_1_fused_0] = 0.00000000f;
      for (int32_t reduce_k_0_4 = 0; reduce_k_0_4 < 2; reduce_k_0_4 += 1) {
        float var_1_local_2 = var_1[(((((((int)blockIdx.x * 4) + i_j_append_var_1_fused_0) * 8) + (int)threadIdx.y) % (S4 * S5)) + ((((reduce_k_0_4 * 32) + (int)threadIdx.x) * S4) * S5))];
        float var_local_13 = var[((reduce_k_0_4 * 32) + (int)threadIdx.x)];
        var_16_loopalign_1_rf[i_j_append_var_1_fused_0] = (var_16_loopalign_1_rf[i_j_append_var_1_fused_0] + (((var_1_local_2 + var_local_13) - (var_7[i_j_append_var_1_fused_0] / 64.0000000f)) * ((var_1_local_2 + var_local_13) - (var_7[i_j_append_var_1_fused_0] / 64.0000000f))));
      };
    };
  };
  for (int32_t i_j_append_var_1_fused_0 = 0; i_j_append_var_1_fused_0 < 4; i_j_append_var_1_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_append_var_1_fused_0) * 8) + (int)threadIdx.y) < (S4 * S5))) {
      var_16_loopalign_1[i_j_append_var_1_fused_0] = cinn_block_reduce_sum_fp32(var_16_loopalign_1_rf[i_j_append_var_1_fused_0], shm32__fp32_reduce, true);
    };
  };
  for (int32_t i_j_append_var_2_fused_2 = 0; i_j_append_var_2_fused_2 < 4; i_j_append_var_2_fused_2 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_append_var_2_fused_2) * 8) + (int)threadIdx.y) < (S4 * S5))) {
      for (int32_t k_3 = 0; k_3 < 2; k_3 += 1) {
        float var_1_local_3 = var_1[((((((((((((int)blockIdx.x * 4) + i_j_append_var_2_fused_2) * 8) + (int)threadIdx.y) / (S4 * S5)) * 64) + (k_3 * 32)) + (int)threadIdx.x) * S4) * S5) + ((((((int)blockIdx.x * 4) + i_j_append_var_2_fused_2) * 8) + (int)threadIdx.y) % (S4 * S5)))];
        float var_local_14 = var[((k_3 * 32) + (int)threadIdx.x)];
        float var_30_local = var_30[((k_3 * 32) + (int)threadIdx.x)];
        float var_34_local = var_34[((k_3 * 32) + (int)threadIdx.x)];
        var_36[((((((((int)blockIdx.x * 4) + i_j_append_var_2_fused_2) * 8) + (int)threadIdx.y) * 64) + (k_3 * 32)) + (int)threadIdx.x)] = (((((var_1_local_3 + var_local_14) - (var_7[i_j_append_var_2_fused_2] / 64.0000000f)) * cinn_nvgpu_rsqrt_fp32(((var_16_loopalign_1[i_j_append_var_2_fused_2] / 64.0000000f) + 9.99999975e-06f))) * var_30_local) + var_34_local);
      };
    };
  };
}__global__
void __launch_bounds__(256) fn_reshape_gs_bc_add_reshape_transpose_sum_full_gs_bc_div_gs_bc_sub_mul_sum_full_gs_bc_div_full_gs_bc_add_rsqrt_gs_bc_mul_gs_bc_mul_gs_bc_add__COND__FPA__FPA__FPA_S4MULS5_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S4MULS5_BPA_MUL64ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_30, const float* __restrict__ var_34, float* __restrict__ var_36, int64_t S4, int64_t S5)
{
  __builtin_assume(((int)blockIdx.x < (((S4 * S5) / 32ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 8ll));
  float _var_16_loopalign_1_rf_temp_buffer [ 4ll ];
  float _var_16_loopalign_1_temp_buffer [ 4ll ];
  float _var_7_rf_temp_buffer [ 4ll ];
  float _var_7_temp_buffer [ 4ll ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_16_loopalign_1 = _var_16_loopalign_1_temp_buffer;
  float* var_16_loopalign_1_rf = _var_16_loopalign_1_rf_temp_buffer;
  float* var_16_loopalign_1_rf__reduce_init = _var_16_loopalign_1_rf_temp_buffer;
  float* var_7 = _var_7_temp_buffer;
  float* var_7_rf = _var_7_rf_temp_buffer;
  float* var_7_rf__reduce_init = _var_7_rf_temp_buffer;
  for (int32_t i_j_k_fused_5 = 0ll; i_j_k_fused_5 < 4ll; i_j_k_fused_5 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_fused_5) * 8ll) + (int)threadIdx.y) < (S4 * S5))) {
      var_7_rf__reduce_init[i_j_k_fused_5] = 0.00000000f;
      for (int32_t reduce_k_0_3 = 0; reduce_k_0_3 < 2; reduce_k_0_3 += 1) {
        float var_1_local_1 = var_1[((((((((((((int)blockIdx.x * 4ll) + i_j_k_fused_5) * 8ll) + (int)threadIdx.y) / (S4 * S5)) * 64ll) + (reduce_k_0_3 * 32ll)) + (int)threadIdx.x) * S4) * S5) + ((((((int)blockIdx.x * 4ll) + i_j_k_fused_5) * 8ll) + (int)threadIdx.y) % (S4 * S5)))];
        float var_local_12 = var[((reduce_k_0_3 * 32ll) + (int)threadIdx.x)];
        var_7_rf[i_j_k_fused_5] = (var_7_rf[i_j_k_fused_5] + (var_1_local_1 + var_local_12));
      };
    };
  };
  for (int32_t i_j_k_fused_5 = 0ll; i_j_k_fused_5 < 4ll; i_j_k_fused_5 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_fused_5) * 8ll) + (int)threadIdx.y) < (S4 * S5))) {
      var_7[i_j_k_fused_5] = cinn_block_reduce_sum_fp32(var_7_rf[i_j_k_fused_5], shm32__fp32_reduce, true);
    };
  };
  for (int32_t i_j_append_var_1_fused_0 = 0ll; i_j_append_var_1_fused_0 < 4ll; i_j_append_var_1_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_append_var_1_fused_0) * 8ll) + (int)threadIdx.y) < (S4 * S5))) {
      var_16_loopalign_1_rf__reduce_init[i_j_append_var_1_fused_0] = 0.00000000f;
      for (int32_t reduce_k_0_4 = 0; reduce_k_0_4 < 2; reduce_k_0_4 += 1) {
        float var_1_local_2 = var_1[(((((((int)blockIdx.x * 4ll) + i_j_append_var_1_fused_0) * 8ll) + (int)threadIdx.y) % (S4 * S5)) + ((((reduce_k_0_4 * 32ll) + (int)threadIdx.x) * S4) * S5))];
        float var_local_13 = var[((reduce_k_0_4 * 32ll) + (int)threadIdx.x)];
        var_16_loopalign_1_rf[i_j_append_var_1_fused_0] = (var_16_loopalign_1_rf[i_j_append_var_1_fused_0] + (((var_1_local_2 + var_local_13) - (var_7[i_j_append_var_1_fused_0] / 64.0000000f)) * ((var_1_local_2 + var_local_13) - (var_7[i_j_append_var_1_fused_0] / 64.0000000f))));
      };
    };
  };
  for (int32_t i_j_append_var_1_fused_0 = 0ll; i_j_append_var_1_fused_0 < 4ll; i_j_append_var_1_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_append_var_1_fused_0) * 8ll) + (int)threadIdx.y) < (S4 * S5))) {
      var_16_loopalign_1[i_j_append_var_1_fused_0] = cinn_block_reduce_sum_fp32(var_16_loopalign_1_rf[i_j_append_var_1_fused_0], shm32__fp32_reduce, true);
    };
  };
  for (int32_t i_j_append_var_2_fused_2 = 0ll; i_j_append_var_2_fused_2 < 4ll; i_j_append_var_2_fused_2 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_append_var_2_fused_2) * 8ll) + (int)threadIdx.y) < (S4 * S5))) {
      for (int32_t k_3 = 0; k_3 < 2; k_3 += 1) {
        float var_1_local_3 = var_1[((((((((((((int)blockIdx.x * 4ll) + i_j_append_var_2_fused_2) * 8ll) + (int)threadIdx.y) / (S4 * S5)) * 64ll) + (k_3 * 32ll)) + (int)threadIdx.x) * S4) * S5) + ((((((int)blockIdx.x * 4ll) + i_j_append_var_2_fused_2) * 8ll) + (int)threadIdx.y) % (S4 * S5)))];
        float var_local_14 = var[((k_3 * 32) + (int)threadIdx.x)];
        float var_30_local = var_30[((k_3 * 32) + (int)threadIdx.x)];
        float var_34_local = var_34[((k_3 * 32) + (int)threadIdx.x)];
        var_36[((((((((int)blockIdx.x * 4ll) + i_j_append_var_2_fused_2) * 8ll) + (int)threadIdx.y) * 64ll) + (k_3 * 32ll)) + (int)threadIdx.x)] = (((((var_1_local_3 + var_local_14) - (var_7[i_j_append_var_2_fused_2] / 64.0000000f)) * cinn_nvgpu_rsqrt_fp32(((var_16_loopalign_1[i_j_append_var_2_fused_2] / 64.0000000f) + 9.99999975e-06f))) * var_30_local) + var_34_local);
      };
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_transpose_gs_reshape__COND__FPA__FPA__FPA__FPA_S0MULS1_BPA_MUL64ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S0MULS1_BPA_MUL64ll_BPA_LE1023ll_BPA__BPA___kernel(const float* __restrict__ var, const int32_t* __restrict__ var_1, const int32_t* __restrict__ var_2, float* __restrict__ var_4, int32_t S0, int32_t S1)
{
  __builtin_assume(((int)blockIdx.x < ((S0 * S1) * 64)));
  var_4[(int)blockIdx.x] = var[((((((int)blockIdx.x / 524288) * 8192) + ((int)blockIdx.x & 8191)) * 64) + (((int)blockIdx.x & 524287) / 8192))];
}__global__
void __launch_bounds__(1024) fn_transpose_gs_reshape__COND__FPA__FPA__FPA_S0MULS1_BPA_MUL64ll_BPA_GE1024ll_BPA___kernel(const float* __restrict__ var, const int32_t* __restrict__ var_1, const int32_t* __restrict__ var_2, float* __restrict__ var_4, int32_t S0, int32_t S1)
{
  __builtin_assume(((int)blockIdx.x < ((((S0 * S1) * 64) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_6 = 0; i_j_k_a_fused_6 < 4; i_j_k_a_fused_6 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) < ((S0 * S1) * 64))) {
      float var_local_15 = var[((((((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) / 524288) * 8192) + ((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) & 8191)) * 64) + (((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) & 524287) / 8192))];
      var_4[(((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x)] = var_local_15;
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_slice__COND__FPA__FPA__FPA__FPA_S2MUL64ll_BPA_GE1ll_BPA_AND_FPA__FPA_S2MUL64ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S2MUL128ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (S2 * 64)));
  var_0[(((((((int)blockIdx.x % (S2 * 64)) / (S2 * 32)) + (((int)blockIdx.x / (S2 * 64)) * 2)) * S2) * 32) + ((int)blockIdx.x % (S2 * 32)))] = var[((((((((int)blockIdx.x % (S2 * 64)) / (S2 * 32)) + (((int)blockIdx.x / (S2 * 64)) * 2)) + 2) * S2) * 32) + ((int)blockIdx.x % (S2 * 32)))];
}__global__
void __launch_bounds__(1) fn_slice__COND__FPA__FPA__FPA__FPA_S2MUL64ll_BPA_GE1ll_BPA_AND_FPA__FPA_S2MUL64ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S2MUL128ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (S2 * 64ll)));
  var_0[(((((((int)blockIdx.x % (S2 * 64ll)) / (S2 * 32ll)) + (((int)blockIdx.x / (S2 * 64ll)) * 2ll)) * S2) * 32ll) + ((int)blockIdx.x % (S2 * 32ll)))] = var[((((((((int)blockIdx.x % (S2 * 64ll)) / (S2 * 32ll)) + (((int)blockIdx.x / (S2 * 64ll)) * 2ll)) + 2ll) * S2) * 32ll) + ((int)blockIdx.x % (S2 * 32ll)))];
}__global__
void __launch_bounds__(1024) fn_slice__COND__FPA__FPA__FPA_S2MUL64ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S2MUL128ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((S2 * 64) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_0 = 0; i_j_k_a_fused_0 < 4; i_j_k_a_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) < (S2 * 64))) {
      float var_local_11 = var[((((((((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) % (S2 * 64)) / (S2 * 32)) + (((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) / (S2 * 64)) * 2)) + 2) * S2) * 32) + ((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) % (S2 * 32)))];
      var_0[(((((((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) % (S2 * 64)) / (S2 * 32)) + (((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) / (S2 * 64)) * 2)) * S2) * 32) + ((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) % (S2 * 32)))] = var_local_11;
    };
  };
}__global__
void __launch_bounds__(1024) fn_slice__COND__FPA__FPA__FPA_S2MUL64ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S2MUL128ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((S2 * 64ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_fused_0 = 0ll; i_j_k_a_fused_0 < 4ll; i_j_k_a_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) < (S2 * 64ll))) {
      float var_local_11 = var[((((((((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) % (S2 * 64ll)) / (S2 * 32ll)) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) / (S2 * 64ll)) * 2ll)) + 2ll) * S2) * 32ll) + ((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) % (S2 * 32ll)))];
      var_0[(((((((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) % (S2 * 64ll)) / (S2 * 32ll)) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) / (S2 * 64ll)) * 2ll)) * S2) * 32ll) + ((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) % (S2 * 32ll)))] = var_local_11;
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA__FPA_S3MUL128ll_BPA_GE1ll_BPA_AND_FPA__FPA_S3MUL128ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S3MUL128ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int32_t S3)
{
  __builtin_assume(((int)blockIdx.x < (S3 * 128)));
  var_5[(((((((int)blockIdx.x % (S3 * 64)) / (S3 * 32)) + (((int)blockIdx.x / (S3 * 64)) * 2)) * S3) * 32) + ((int)blockIdx.x % (S3 * 32)))] = (var[(((((((int)blockIdx.x % (S3 * 64)) / (S3 * 32)) * 32) + (((int)blockIdx.x / (S3 * 64)) * 64)) + ((int)blockIdx.x & 31)) + ((((int)blockIdx.x % (S3 * 32)) / 32) * 128))] + var_1[((((((int)blockIdx.x % (S3 * 64)) / (S3 * 32)) * 32) + (((int)blockIdx.x / (S3 * 64)) * 64)) + ((int)blockIdx.x & 31))]);
  if ((((int)blockIdx.x / (S3 * 64)) == 0)) {
    var_7[(((((((int)blockIdx.x % (S3 * 64)) / (S3 * 32)) * 32) + ((int)blockIdx.x & 31)) * S3) + (((int)blockIdx.x % (S3 * 32)) / 32))] = (var[((((((int)blockIdx.x % (S3 * 64)) / (S3 * 32)) * 32) + ((int)blockIdx.x & 31)) + ((((int)blockIdx.x % (S3 * 32)) / 32) * 128))] + var_1[(((((int)blockIdx.x % (S3 * 64)) / (S3 * 32)) * 32) + ((int)blockIdx.x & 31))]);
  };
}__global__
void __launch_bounds__(1) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA__FPA_S3MUL128ll_BPA_GE1ll_BPA_AND_FPA__FPA_S3MUL128ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S3MUL128ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int64_t S3)
{
  __builtin_assume(((int)blockIdx.x < (S3 * 128ll)));
  var_5[(((((((int)blockIdx.x % (S3 * 64ll)) / (S3 * 32ll)) + (((int)blockIdx.x / (S3 * 64ll)) * 2ll)) * S3) * 32ll) + ((int)blockIdx.x % (S3 * 32ll)))] = (var[(((((((int)blockIdx.x % (S3 * 64ll)) / (S3 * 32ll)) * 32ll) + (((int)blockIdx.x / (S3 * 64ll)) * 64ll)) + ((int)blockIdx.x & 31)) + ((((int)blockIdx.x % (S3 * 32ll)) / 32ll) * 128ll))] + var_1[((((((int)blockIdx.x % (S3 * 64ll)) / (S3 * 32ll)) * 32ll) + (((int)blockIdx.x / (S3 * 64ll)) * 64ll)) + ((int)blockIdx.x & 31))]);
  if ((((int)blockIdx.x / (S3 * 64ll)) == 0ll)) {
    var_7[(((((((int)blockIdx.x % (S3 * 64ll)) / (S3 * 32ll)) * 32ll) + ((int)blockIdx.x & 31)) * S3) + (((int)blockIdx.x % (S3 * 32ll)) / 32ll))] = (var[((((((int)blockIdx.x % (S3 * 64ll)) / (S3 * 32ll)) * 32ll) + ((int)blockIdx.x & 31)) + ((((int)blockIdx.x % (S3 * 32ll)) / 32ll) * 128ll))] + var_1[(((((int)blockIdx.x % (S3 * 64ll)) / (S3 * 32ll)) * 32ll) + ((int)blockIdx.x & 31))]);
  };
}__global__
void __launch_bounds__(1024) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA_S3MUL128ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S3MUL128ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int32_t S3)
{
  __builtin_assume(((int)blockIdx.x < (((S3 * 128) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_b_fused_0 = 0; i_j_k_a_b_fused_0 < 4; i_j_k_a_b_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) < (S3 * 128))) {
      float var_local_16 = var[(((((((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) % (S3 * 64)) / (S3 * 32)) * 32) + (((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) / (S3 * 64)) * 64)) + ((int)threadIdx.x & 31)) + ((((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) % (S3 * 32)) / 32) * 128))];
      float var_1_local_4 = var_1[((((((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) % (S3 * 64)) / (S3 * 32)) * 32) + (((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) / (S3 * 64)) * 64)) + ((int)threadIdx.x & 31))];
      var_5[(((((((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) % (S3 * 64)) / (S3 * 32)) + (((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) / (S3 * 64)) * 2)) * S3) * 32) + ((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) % (S3 * 32)))] = (var_local_16 + var_1_local_4);
    };
  };
  for (int32_t append_var_0_i_j_a_k_fused_0 = 0; append_var_0_i_j_a_k_fused_0 < 4; append_var_0_i_j_a_k_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) < (S3 * 128))) {
      if ((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) / (S3 * 64)) == 0)) {
        var_7[(((((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S3 * 64)) / (S3 * 32)) * 32) + ((int)threadIdx.x & 31)) * S3) + (((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S3 * 32)) / 32))] = (var[((((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S3 * 64)) / (S3 * 32)) * 32) + ((int)threadIdx.x & 31)) + ((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S3 * 32)) / 32) * 128))] + var_1[(((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S3 * 64)) / (S3 * 32)) * 32) + ((int)threadIdx.x & 31))]);
      };
    };
  };
}__global__
void __launch_bounds__(1024) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA_S3MUL128ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S3MUL128ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int64_t S3)
{
  __builtin_assume(((int)blockIdx.x < (((S3 * 128ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_b_fused_0 = 0ll; i_j_k_a_b_fused_0 < 4ll; i_j_k_a_b_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) < (S3 * 128ll))) {
      float var_local_16 = var[(((((((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) % (S3 * 64ll)) / (S3 * 32ll)) * 32ll) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) / (S3 * 64ll)) * 64ll)) + ((int)threadIdx.x & 31)) + ((((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) % (S3 * 32ll)) / 32ll) * 128ll))];
      float var_1_local_4 = var_1[((((((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) % (S3 * 64ll)) / (S3 * 32ll)) * 32ll) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) / (S3 * 64ll)) * 64ll)) + ((int)threadIdx.x & 31))];
      var_5[(((((((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) % (S3 * 64ll)) / (S3 * 32ll)) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) / (S3 * 64ll)) * 2ll)) * S3) * 32ll) + ((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) % (S3 * 32ll)))] = (var_local_16 + var_1_local_4);
    };
  };
  for (int32_t append_var_0_i_j_a_k_fused_0 = 0ll; append_var_0_i_j_a_k_fused_0 < 4ll; append_var_0_i_j_a_k_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) < (S3 * 128ll))) {
      if ((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) / (S3 * 64ll)) == 0ll)) {
        var_7[(((((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S3 * 64ll)) / (S3 * 32ll)) * 32ll) + ((int)threadIdx.x & 31)) * S3) + (((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S3 * 32ll)) / 32ll))] = (var[((((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S3 * 64ll)) / (S3 * 32ll)) * 32ll) + ((int)threadIdx.x & 31)) + ((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S3 * 32ll)) / 32ll) * 128ll))] + var_1[(((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S3 * 64ll)) / (S3 * 32ll)) * 32ll) + ((int)threadIdx.x & 31))]);
      };
    };
  };
}

}
