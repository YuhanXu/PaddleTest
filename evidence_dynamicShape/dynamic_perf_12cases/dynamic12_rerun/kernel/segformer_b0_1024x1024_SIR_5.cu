
extern "C" {

__global__
void __launch_bounds__(256) fn_transpose_reshape__COND_true__kernel(const float* __restrict__ var, float* __restrict__ var_1)
{
  __builtin_assume(((int)blockIdx.x < 2048));
  __builtin_assume(((int)threadIdx.x < 256));
  for (int32_t i_j_k_fused_0 = 0; i_j_k_fused_0 < 4; i_j_k_fused_0 += 1) {
    float var_local = var[((((int)blockIdx.x * 1024) + (i_j_k_fused_0 * 256)) + (int)threadIdx.x)];
    var_1[((((int)blockIdx.x * 1024) + (i_j_k_fused_0 * 256)) + (int)threadIdx.x)] = var_local;
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_bc_add_reshape_transpose__COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_4)
{
  __builtin_assume(((int)blockIdx.x < 2048));
  __builtin_assume(((int)threadIdx.x < 256));
  for (int32_t i_j_k_a_fused_3 = 0; i_j_k_a_fused_3 < 4; i_j_k_a_fused_3 += 1) {
    float var_1_local = var_1[((((int)blockIdx.x * 1024) + (i_j_k_a_fused_3 * 256)) + (int)threadIdx.x)];
    float var_local_0 = var[((int)threadIdx.x & 31)];
    var_4[((((int)blockIdx.x * 1024) + (i_j_k_a_fused_3 * 256)) + (int)threadIdx.x)] = (var_1_local + var_local_0);
  };
}

}

extern "C" {

__global__
void __launch_bounds__(128) fn_bc_add__COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_2)
{
  __builtin_assume(((int)blockIdx.x < 4096));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 4));
  for (int32_t i_j_fused_k_fused_0 = 0; i_j_fused_k_fused_0 < 4; i_j_fused_k_fused_0 += 1) {
    float var_1_local_0 = var_1[(((((int)blockIdx.x * 512) + (i_j_fused_k_fused_0 * 128)) + ((int)threadIdx.y * 32)) + (int)threadIdx.x)];
    float var_local_1 = var[((int)threadIdx.x & 31)];
    var_2[(((((int)blockIdx.x * 512) + (i_j_fused_k_fused_0 * 128)) + ((int)threadIdx.y * 32)) + (int)threadIdx.x)] = (var_1_local_0 + var_local_1);
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_slice__COND__FPA__FPA__FPA__FPA_S5MUL32ll_BPA_GE1ll_BPA_AND_FPA__FPA_S5MUL32ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S5MUL64ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int32_t S5)
{
  __builtin_assume(((int)blockIdx.x < (S5 * 32)));
  var_0[(int)blockIdx.x] = var[((S5 * 32) + (int)blockIdx.x)];
}__global__
void __launch_bounds__(1) fn_slice__COND__FPA__FPA__FPA__FPA_S5MUL32ll_BPA_GE1ll_BPA_AND_FPA__FPA_S5MUL32ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S5MUL64ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int64_t S5)
{
  __builtin_assume(((int)blockIdx.x < (S5 * 32ll)));
  var_0[(int)blockIdx.x] = var[((S5 * 32ll) + (int)blockIdx.x)];
}__global__
void __launch_bounds__(1024) fn_slice__COND__FPA__FPA__FPA_S5MUL32ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S5MUL64ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int32_t S5)
{
  __builtin_assume(((int)blockIdx.x < (((S5 * 32) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_0 = 0; i_j_k_a_fused_0 < 4; i_j_k_a_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) < (S5 * 32))) {
      float var_local_2 = var[((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) + (S5 * 32))];
      var_0[(((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x)] = var_local_2;
    };
  };
}__global__
void __launch_bounds__(1024) fn_slice__COND__FPA__FPA__FPA_S5MUL32ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S5MUL64ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_0, int64_t S5)
{
  __builtin_assume(((int)blockIdx.x < (((S5 * 32ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_fused_0 = 0ll; i_j_k_a_fused_0 < 4ll; i_j_k_a_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) < (S5 * 32ll))) {
      float var_local_2 = var[((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) + (S5 * 32ll))];
      var_0[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x)] = var_local_2;
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_sum_full_div_bc_sub_mul_sum_full_div_full_add_rsqrt_bc_mul_bc_mul_bc_add__COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_14, const float* __restrict__ var_17, float* __restrict__ var_19)
{
  __builtin_assume(((int)blockIdx.x < 2048));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 8));
  float _var_0_rf_temp_buffer [ 4 ];
  float _var_0_temp_buffer [ 4 ];
  float _var_6_rf_temp_buffer [ 4 ];
  float _var_6_temp_buffer [ 4 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_0 = _var_0_temp_buffer;
  float* var_0_rf = _var_0_rf_temp_buffer;
  float* var_0_rf__reduce_init = _var_0_rf_temp_buffer;
  float* var_6 = _var_6_temp_buffer;
  float* var_6_rf = _var_6_rf_temp_buffer;
  float* var_6_rf__reduce_init = _var_6_rf_temp_buffer;
  for (int32_t i_j_k_fused_3 = 0; i_j_k_fused_3 < 4; i_j_k_fused_3 += 1) {
    float var_local_4 = var[((((((int)blockIdx.x * 32) + (i_j_k_fused_3 * 8)) + (int)threadIdx.y) * 32) + (int)threadIdx.x)];
    var_0_rf__reduce_init[i_j_k_fused_3] = 0.00000000f;
    var_0_rf[i_j_k_fused_3] = (var_0_rf[i_j_k_fused_3] + var_local_4);
  };
  for (int32_t i_j_k_fused_3 = 0; i_j_k_fused_3 < 4; i_j_k_fused_3 += 1) {
    var_0[i_j_k_fused_3] = cinn_block_reduce_sum_fp32(var_0_rf[i_j_k_fused_3], shm32__fp32_reduce, true);
  };
  for (int32_t i_j_append_var_3_fused_0 = 0; i_j_append_var_3_fused_0 < 4; i_j_append_var_3_fused_0 += 1) {
    float var_local_5 = var[((((((int)blockIdx.x * 32) + (i_j_append_var_3_fused_0 * 8)) + (int)threadIdx.y) * 32) + (int)threadIdx.x)];
    var_6_rf__reduce_init[i_j_append_var_3_fused_0] = 0.00000000f;
    var_6_rf[i_j_append_var_3_fused_0] = (var_6_rf[i_j_append_var_3_fused_0] + ((var_local_5 - (var_0[i_j_append_var_3_fused_0] / 32.0000000f)) * (var_local_5 - (var_0[i_j_append_var_3_fused_0] / 32.0000000f))));
  };
  for (int32_t i_j_append_var_3_fused_0 = 0; i_j_append_var_3_fused_0 < 4; i_j_append_var_3_fused_0 += 1) {
    var_6[i_j_append_var_3_fused_0] = cinn_block_reduce_sum_fp32(var_6_rf[i_j_append_var_3_fused_0], shm32__fp32_reduce, true);
  };
  for (int32_t i_j_append_var_4_fused_0 = 0; i_j_append_var_4_fused_0 < 4; i_j_append_var_4_fused_0 += 1) {
    float var_local_6 = var[((((((int)blockIdx.x * 32) + (i_j_append_var_4_fused_0 * 8)) + (int)threadIdx.y) * 32) + (int)threadIdx.x)];
    float var_14_local = var_14[(int)threadIdx.x];
    float var_17_local = var_17[(int)threadIdx.x];
    var_19[((((((int)blockIdx.x * 32) + (i_j_append_var_4_fused_0 * 8)) + (int)threadIdx.y) * 32) + (int)threadIdx.x)] = ((((var_local_6 - (var_0[i_j_append_var_4_fused_0] / 32.0000000f)) * cinn_nvgpu_rsqrt_fp32(((var_6[i_j_append_var_4_fused_0] / 32.0000000f) + 9.99999997e-07f))) * var_14_local) + var_17_local);
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA__FPA_S6MUL64ll_BPA_GE1ll_BPA_AND_FPA__FPA_S6MUL64ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S6MUL64ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int32_t S6)
{
  __builtin_assume(((int)blockIdx.x < (S6 * 64)));
  var_5[(int)blockIdx.x] = (var[(((((int)blockIdx.x / (S6 * 32)) * 32) + ((int)blockIdx.x & 31)) + ((((int)blockIdx.x % (S6 * 32)) / 32) * 64))] + var_1[((((int)blockIdx.x / (S6 * 32)) * 32) + ((int)blockIdx.x & 31))]);
  if ((((int)blockIdx.x / (S6 * 32)) == 0)) {
    var_7[((((int)blockIdx.x % (S6 * 32)) / 32) + (((int)blockIdx.x & 31) * S6))] = (var[(((((int)blockIdx.x % (S6 * 32)) / 32) * 64) + ((int)blockIdx.x & 31))] + var_1[((int)blockIdx.x & 31)]);
  };
}__global__
void __launch_bounds__(1) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA__FPA_S6MUL64ll_BPA_GE1ll_BPA_AND_FPA__FPA_S6MUL64ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S6MUL64ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int64_t S6)
{
  __builtin_assume(((int)blockIdx.x < (S6 * 64ll)));
  var_5[(int)blockIdx.x] = (var[(((((int)blockIdx.x / (S6 * 32ll)) * 32ll) + ((int)blockIdx.x & 31)) + ((((int)blockIdx.x % (S6 * 32ll)) / 32ll) * 64ll))] + var_1[((((int)blockIdx.x / (S6 * 32ll)) * 32ll) + ((int)blockIdx.x & 31))]);
  if ((((int)blockIdx.x / (S6 * 32ll)) == 0ll)) {
    var_7[((((int)blockIdx.x % (S6 * 32ll)) / 32ll) + (((int)blockIdx.x & 31) * S6))] = (var[(((((int)blockIdx.x % (S6 * 32ll)) / 32ll) * 64ll) + ((int)blockIdx.x & 31))] + var_1[((int)blockIdx.x & 31)]);
  };
}__global__
void __launch_bounds__(1024) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA_S6MUL64ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S6MUL64ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int32_t S6)
{
  __builtin_assume(((int)blockIdx.x < (((S6 * 64) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_b_fused_0 = 0; i_j_k_a_b_fused_0 < 4; i_j_k_a_b_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) < (S6 * 64))) {
      float var_local_3 = var[(((((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) / (S6 * 32)) * 32) + ((int)threadIdx.x & 31)) + ((((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) % (S6 * 32)) / 32) * 64))];
      float var_1_local_1 = var_1[((((((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x) / (S6 * 32)) * 32) + ((int)threadIdx.x & 31))];
      var_5[(((((int)blockIdx.x * 4) + i_j_k_a_b_fused_0) * 1024) + (int)threadIdx.x)] = (var_local_3 + var_1_local_1);
    };
  };
  for (int32_t append_var_0_i_j_a_k_fused_0 = 0; append_var_0_i_j_a_k_fused_0 < 4; append_var_0_i_j_a_k_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) < (S6 * 64))) {
      if ((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) / (S6 * 32)) == 0)) {
        var_7[((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S6 * 32)) / 32) + (((int)threadIdx.x & 31) * S6))] = (var[(((((((((int)blockIdx.x * 4) + append_var_0_i_j_a_k_fused_0) * 1024) + (int)threadIdx.x) % (S6 * 32)) / 32) * 64) + ((int)threadIdx.x & 31))] + var_1[((int)threadIdx.x & 31)]);
      };
    };
  };
}__global__
void __launch_bounds__(1024) fn_gs_bc_add_reshape_transpose_slice_transpose__COND__FPA__FPA__FPA_S6MUL64ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S6MUL64ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_7, float* __restrict__ var_5, int64_t S6)
{
  __builtin_assume(((int)blockIdx.x < (((S6 * 64ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_b_fused_0 = 0ll; i_j_k_a_b_fused_0 < 4ll; i_j_k_a_b_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) < (S6 * 64ll))) {
      float var_local_3 = var[(((((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) / (S6 * 32ll)) * 32ll) + ((int)threadIdx.x & 31)) + ((((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) % (S6 * 32ll)) / 32ll) * 64ll))];
      float var_1_local_1 = var_1[((((((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x) / (S6 * 32ll)) * 32ll) + ((int)threadIdx.x & 31))];
      var_5[(((((int)blockIdx.x * 4ll) + i_j_k_a_b_fused_0) * 1024ll) + (int)threadIdx.x)] = (var_local_3 + var_1_local_1);
    };
  };
  for (int32_t append_var_0_i_j_a_k_fused_0 = 0ll; append_var_0_i_j_a_k_fused_0 < 4ll; append_var_0_i_j_a_k_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) < (S6 * 64ll))) {
      if ((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) / (S6 * 32ll)) == 0ll)) {
        var_7[((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S6 * 32ll)) / 32ll) + (((int)threadIdx.x & 31) * S6))] = (var[(((((((((int)blockIdx.x * 4ll) + append_var_0_i_j_a_k_fused_0) * 1024ll) + (int)threadIdx.x) % (S6 * 32ll)) / 32ll) * 64ll) + ((int)threadIdx.x & 31))] + var_1[((int)threadIdx.x & 31)]);
      };
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA__FPA_S2GE1ll_BPA_AND_FPA_S2LE2048ll_BPA__BPA_AND_FPA__FPA_S2MUL65536ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < 65536));
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
  for (int32_t reduce_k_0_0 = 0; reduce_k_0_0 < ((S2 / 256) + 1); reduce_k_0_0 += 1) {
    int64_t reduce_k_0_0_strided = (reduce_k_0_0 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_0_strided))) < S2), 256);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_0_strided))) < S2)) {
      float var_local_7 = var[(((int)threadIdx.x + reduce_k_0_0_strided) + (S2 * (int)blockIdx.x))];
      var_1_rf[0] = max(var_1_rf[0], (0.176776692f * var_local_7));
    };
  };
  var_1[0] = cinn_block_reduce_max_fp32(var_1_rf[0], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_2 = 0; reduce_k_0_2 < ((S2 / 256) + 1); reduce_k_0_2 += 1) {
    int64_t reduce_k_0_2_strided = (reduce_k_0_2 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_2_strided))) < S2), 256);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_2_strided))) < S2)) {
      float var_local_8 = var[(((((int)blockIdx.x & 65535) * S2) + (int)threadIdx.x) + reduce_k_0_2_strided)];
      var_6_loopalign_1_rf[0] = (var_6_loopalign_1_rf[0] + cinn_nvgpu_exp_fp32(((0.176776692f * var_local_8) - var_1[0])));
    };
  };
  var_6_loopalign_1[0] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf[0], shm32__fp32_reduce, false);
  for (int32_t a = 0; a < ((S2 / 256) + 1); a += 1) {
    int64_t a_strided = (a * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(a_strided, ((((int32_t)(a_strided)) + (int)threadIdx.x) < S2), 256);
    if (((((int32_t)(a_strided)) + (int)threadIdx.x) < S2)) {
      float var_local_9 = var[((a_strided + (int)threadIdx.x) + (S2 * (int)blockIdx.x))];
      var_9[((a_strided + (int)threadIdx.x) + (S2 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.176776692f * var_local_9) - var_1[0])) / var_6_loopalign_1[0]);
    };
  };
}__global__
void __launch_bounds__(256) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA__FPA_S2GE1ll_BPA_AND_FPA_S2LE2048ll_BPA__BPA_AND_FPA__FPA_S2MUL65536ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < 65536ll));
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
  for (int32_t reduce_k_0_0 = 0ll; reduce_k_0_0 < ((S2 / 256ll) + 1ll); reduce_k_0_0 += 1) {
    int64_t reduce_k_0_0_strided = (reduce_k_0_0 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_strided, (((int)threadIdx.x + reduce_k_0_0_strided) < S2), 256ll);
    if ((((int)threadIdx.x + reduce_k_0_0_strided) < S2)) {
      float var_local_7 = var[(((int)threadIdx.x + reduce_k_0_0_strided) + (S2 * (int)blockIdx.x))];
      var_1_rf[0ll] = max(var_1_rf[0ll], (0.176776692f * var_local_7));
    };
  };
  var_1[0ll] = cinn_block_reduce_max_fp32(var_1_rf[0ll], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf__reduce_init[0ll] = 0.00000000f;
  for (int32_t reduce_k_0_2 = 0ll; reduce_k_0_2 < ((S2 / 256ll) + 1ll); reduce_k_0_2 += 1) {
    int64_t reduce_k_0_2_strided = (reduce_k_0_2 * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_strided, (((int)threadIdx.x + reduce_k_0_2_strided) < S2), 256ll);
    if ((((int)threadIdx.x + reduce_k_0_2_strided) < S2)) {
      float var_local_8 = var[(((((int)blockIdx.x & 65535) * S2) + (int)threadIdx.x) + reduce_k_0_2_strided)];
      var_6_loopalign_1_rf[0ll] = (var_6_loopalign_1_rf[0ll] + cinn_nvgpu_exp_fp32(((0.176776692f * var_local_8) - var_1[0ll])));
    };
  };
  var_6_loopalign_1[0ll] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf[0ll], shm32__fp32_reduce, false);
  for (int32_t a = 0ll; a < ((S2 / 256ll) + 1ll); a += 1) {
    int64_t a_strided = (a * 256ll);
    CINN_ENTAIL_LOOP_CONDITION(a_strided, ((a_strided + (int)threadIdx.x) < S2), 256ll);
    if (((a_strided + (int)threadIdx.x) < S2)) {
      float var_local_9 = var[((a_strided + (int)threadIdx.x) + (S2 * (int)blockIdx.x))];
      var_9[((a_strided + (int)threadIdx.x) + (S2 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.176776692f * var_local_9) - var_1[0ll])) / var_6_loopalign_1[0ll]);
    };
  };
}__global__
void __launch_bounds__(1024) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA_S2GE2049ll_BPA_AND_FPA__FPA_S2MUL65536ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < 65536));
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
  for (int32_t reduce_k_0_0_1 = 0; reduce_k_0_0_1 < ((S2 / 1024) + 1); reduce_k_0_0_1 += 1) {
    int64_t reduce_k_0_0_1_strided = (reduce_k_0_0_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_1_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_0_1_strided))) < S2), 1024);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_0_1_strided))) < S2)) {
      float var_local_10 = var[(((int)threadIdx.x + reduce_k_0_0_1_strided) + (S2 * (int)blockIdx.x))];
      var_1_rf_0[0] = max(var_1_rf_0[0], (0.176776692f * var_local_10));
    };
  };
  var_1[0] = cinn_block_reduce_max_fp32(var_1_rf_0[0], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf_0__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_2_1 = 0; reduce_k_0_2_1 < ((S2 / 1024) + 1); reduce_k_0_2_1 += 1) {
    int64_t reduce_k_0_2_1_strided = (reduce_k_0_2_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_1_strided, (((int)threadIdx.x + ((int32_t)(reduce_k_0_2_1_strided))) < S2), 1024);
    if ((((int)threadIdx.x + ((int32_t)(reduce_k_0_2_1_strided))) < S2)) {
      float var_local_11 = var[(((((int)blockIdx.x & 65535) * S2) + (int)threadIdx.x) + reduce_k_0_2_1_strided)];
      var_6_loopalign_1_rf_0[0] = (var_6_loopalign_1_rf_0[0] + cinn_nvgpu_exp_fp32(((0.176776692f * var_local_11) - var_1[0])));
    };
  };
  var_6_loopalign_1[0] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf_0[0], shm32__fp32_reduce, false);
  for (int32_t a_5 = 0; a_5 < ((S2 / 1024) + 1); a_5 += 1) {
    int64_t a_5_strided = (a_5 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(a_5_strided, ((((int32_t)(a_5_strided)) + (int)threadIdx.x) < S2), 1024);
    if (((((int32_t)(a_5_strided)) + (int)threadIdx.x) < S2)) {
      float var_local_12 = var[((a_5_strided + (int)threadIdx.x) + (S2 * (int)blockIdx.x))];
      var_9[((a_5_strided + (int)threadIdx.x) + (S2 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.176776692f * var_local_12) - var_1[0])) / var_6_loopalign_1[0]);
    };
  };
}__global__
void __launch_bounds__(1024) fn_scale_r_max_gs_bc_sub_exp_sum_gs_bc_div__COND__FPA__FPA_S2GE2049ll_BPA_AND_FPA__FPA_S2MUL65536ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, float* __restrict__ var_9, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < 65536ll));
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
  for (int32_t reduce_k_0_0_1 = 0ll; reduce_k_0_0_1 < ((S2 / 1024ll) + 1ll); reduce_k_0_0_1 += 1) {
    int64_t reduce_k_0_0_1_strided = (reduce_k_0_0_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_0_1_strided, (((int)threadIdx.x + reduce_k_0_0_1_strided) < S2), 1024ll);
    if ((((int)threadIdx.x + reduce_k_0_0_1_strided) < S2)) {
      float var_local_10 = var[(((int)threadIdx.x + reduce_k_0_0_1_strided) + (S2 * (int)blockIdx.x))];
      var_1_rf_0[0ll] = max(var_1_rf_0[0ll], (0.176776692f * var_local_10));
    };
  };
  var_1[0ll] = cinn_block_reduce_max_fp32(var_1_rf_0[0ll], shm32__fp32_reduce, false);
  var_6_loopalign_1_rf_0__reduce_init[0ll] = 0.00000000f;
  for (int32_t reduce_k_0_2_1 = 0ll; reduce_k_0_2_1 < ((S2 / 1024ll) + 1ll); reduce_k_0_2_1 += 1) {
    int64_t reduce_k_0_2_1_strided = (reduce_k_0_2_1 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(reduce_k_0_2_1_strided, (((int)threadIdx.x + reduce_k_0_2_1_strided) < S2), 1024ll);
    if ((((int)threadIdx.x + reduce_k_0_2_1_strided) < S2)) {
      float var_local_11 = var[(((((int)blockIdx.x & 65535) * S2) + (int)threadIdx.x) + reduce_k_0_2_1_strided)];
      var_6_loopalign_1_rf_0[0ll] = (var_6_loopalign_1_rf_0[0ll] + cinn_nvgpu_exp_fp32(((0.176776692f * var_local_11) - var_1[0ll])));
    };
  };
  var_6_loopalign_1[0ll] = cinn_block_reduce_sum_fp32(var_6_loopalign_1_rf_0[0ll], shm32__fp32_reduce, false);
  for (int32_t a_5 = 0ll; a_5 < ((S2 / 1024ll) + 1ll); a_5 += 1) {
    int64_t a_5_strided = (a_5 * 1024ll);
    CINN_ENTAIL_LOOP_CONDITION(a_5_strided, ((a_5_strided + (int)threadIdx.x) < S2), 1024ll);
    if (((a_5_strided + (int)threadIdx.x) < S2)) {
      float var_local_12 = var[((a_5_strided + (int)threadIdx.x) + (S2 * (int)blockIdx.x))];
      var_9[((a_5_strided + (int)threadIdx.x) + (S2 * (int)blockIdx.x))] = (cinn_nvgpu_exp_fp32(((0.176776692f * var_local_12) - var_1[0ll])) / var_6_loopalign_1[0ll]);
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_reshape_gs_bc_add_reshape_transpose_sum_full_gs_bc_div_gs_bc_sub_mul_sum_full_gs_bc_div_full_gs_bc_add_rsqrt_gs_bc_mul_gs_bc_mul_gs_bc_add__COND__FPA__FPA__FPA_S3MULS4_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S3MULS4_BPA_MUL32ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_30, const float* __restrict__ var_34, float* __restrict__ var_36, int32_t S3, int32_t S4)
{
  __builtin_assume(((int)blockIdx.x < (((S3 * S4) / 64) + 1)));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 8));
  float _var_16_loopalign_1_rf_temp_buffer [ 8ll ];
  float _var_16_loopalign_1_temp_buffer [ 8ll ];
  float _var_7_rf_temp_buffer [ 8ll ];
  float _var_7_temp_buffer [ 8ll ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_16_loopalign_1 = _var_16_loopalign_1_temp_buffer;
  float* var_16_loopalign_1_rf = _var_16_loopalign_1_rf_temp_buffer;
  float* var_16_loopalign_1_rf__reduce_init = _var_16_loopalign_1_rf_temp_buffer;
  float* var_7 = _var_7_temp_buffer;
  float* var_7_rf = _var_7_rf_temp_buffer;
  float* var_7_rf__reduce_init = _var_7_rf_temp_buffer;
  for (int32_t i_j_k_fused_6 = 0; i_j_k_fused_6 < 8; i_j_k_fused_6 += 1) {
    if (((((((int)blockIdx.x * 8) + i_j_k_fused_6) * 8) + (int)threadIdx.y) < (S3 * S4))) {
      var_7_rf__reduce_init[i_j_k_fused_6] = 0.00000000f;
      for (int32_t reduce_k_0_3 = 0; reduce_k_0_3 < 1; reduce_k_0_3 += 1) {
        float var_1_local_2 = var_1[(((((((((((int)blockIdx.x * 8) + i_j_k_fused_6) * 8) + (int)threadIdx.y) / (S3 * S4)) * 32) + (int)threadIdx.x) * S3) * S4) + ((((((int)blockIdx.x * 8) + i_j_k_fused_6) * 8) + (int)threadIdx.y) % (S3 * S4)))];
        float var_local_13 = var[(int)threadIdx.x];
        var_7_rf[i_j_k_fused_6] = (var_7_rf[i_j_k_fused_6] + (var_1_local_2 + var_local_13));
      };
    };
  };
  for (int32_t i_j_k_fused_6 = 0; i_j_k_fused_6 < 8; i_j_k_fused_6 += 1) {
    if (((((((int)blockIdx.x * 8) + i_j_k_fused_6) * 8) + (int)threadIdx.y) < (S3 * S4))) {
      var_7[i_j_k_fused_6] = cinn_block_reduce_sum_fp32(var_7_rf[i_j_k_fused_6], shm32__fp32_reduce, true);
    };
  };
  for (int32_t i_j_append_var_1_fused_0 = 0; i_j_append_var_1_fused_0 < 8; i_j_append_var_1_fused_0 += 1) {
    if (((((((int)blockIdx.x * 8) + i_j_append_var_1_fused_0) * 8) + (int)threadIdx.y) < (S3 * S4))) {
      var_16_loopalign_1_rf__reduce_init[i_j_append_var_1_fused_0] = 0.00000000f;
      for (int32_t reduce_k_0_4 = 0; reduce_k_0_4 < 1; reduce_k_0_4 += 1) {
        float var_1_local_3 = var_1[(((((((int)blockIdx.x * 8) + i_j_append_var_1_fused_0) * 8) + (int)threadIdx.y) % (S3 * S4)) + ((S3 * (int)threadIdx.x) * S4))];
        float var_local_14 = var[(int)threadIdx.x];
        var_16_loopalign_1_rf[i_j_append_var_1_fused_0] = (var_16_loopalign_1_rf[i_j_append_var_1_fused_0] + (((var_1_local_3 + var_local_14) - (var_7[i_j_append_var_1_fused_0] / 32.0000000f)) * ((var_1_local_3 + var_local_14) - (var_7[i_j_append_var_1_fused_0] / 32.0000000f))));
      };
    };
  };
  for (int32_t i_j_append_var_1_fused_0 = 0; i_j_append_var_1_fused_0 < 8; i_j_append_var_1_fused_0 += 1) {
    if (((((((int)blockIdx.x * 8) + i_j_append_var_1_fused_0) * 8) + (int)threadIdx.y) < (S3 * S4))) {
      var_16_loopalign_1[i_j_append_var_1_fused_0] = cinn_block_reduce_sum_fp32(var_16_loopalign_1_rf[i_j_append_var_1_fused_0], shm32__fp32_reduce, true);
    };
  };
  for (int32_t i_j_append_var_2_fused_3 = 0; i_j_append_var_2_fused_3 < 8; i_j_append_var_2_fused_3 += 1) {
    if (((((((int)blockIdx.x * 8) + i_j_append_var_2_fused_3) * 8) + (int)threadIdx.y) < (S3 * S4))) {
      float var_1_local_4 = var_1[(((((((((((int)blockIdx.x * 8) + i_j_append_var_2_fused_3) * 8) + (int)threadIdx.y) / (S3 * S4)) * 32) + (int)threadIdx.x) * S3) * S4) + ((((((int)blockIdx.x * 8) + i_j_append_var_2_fused_3) * 8) + (int)threadIdx.y) % (S3 * S4)))];
      float var_local_15 = var[(int)threadIdx.x];
      float var_30_local = var_30[(int)threadIdx.x];
      float var_34_local = var_34[(int)threadIdx.x];
      var_36[(((((((int)blockIdx.x * 8) + i_j_append_var_2_fused_3) * 8) + (int)threadIdx.y) * 32) + (int)threadIdx.x)] = (((((var_1_local_4 + var_local_15) - (var_7[i_j_append_var_2_fused_3] / 32.0000000f)) * cinn_nvgpu_rsqrt_fp32(((var_16_loopalign_1[i_j_append_var_2_fused_3] / 32.0000000f) + 9.99999975e-06f))) * var_30_local) + var_34_local);
    };
  };
}__global__
void __launch_bounds__(256) fn_reshape_gs_bc_add_reshape_transpose_sum_full_gs_bc_div_gs_bc_sub_mul_sum_full_gs_bc_div_full_gs_bc_add_rsqrt_gs_bc_mul_gs_bc_mul_gs_bc_add__COND__FPA__FPA__FPA_S3MULS4_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S3MULS4_BPA_MUL32ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_30, const float* __restrict__ var_34, float* __restrict__ var_36, int64_t S3, int64_t S4)
{
  __builtin_assume(((int)blockIdx.x < (((S3 * S4) / 64ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 8ll));
  float _var_16_loopalign_1_rf_temp_buffer [ 8ll ];
  float _var_16_loopalign_1_temp_buffer [ 8ll ];
  float _var_7_rf_temp_buffer [ 8ll ];
  float _var_7_temp_buffer [ 8ll ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_16_loopalign_1 = _var_16_loopalign_1_temp_buffer;
  float* var_16_loopalign_1_rf = _var_16_loopalign_1_rf_temp_buffer;
  float* var_16_loopalign_1_rf__reduce_init = _var_16_loopalign_1_rf_temp_buffer;
  float* var_7 = _var_7_temp_buffer;
  float* var_7_rf = _var_7_rf_temp_buffer;
  float* var_7_rf__reduce_init = _var_7_rf_temp_buffer;
  for (int32_t i_j_k_fused_6 = 0ll; i_j_k_fused_6 < 8ll; i_j_k_fused_6 += 1) {
    if (((((((int)blockIdx.x * 8ll) + i_j_k_fused_6) * 8ll) + (int)threadIdx.y) < (S3 * S4))) {
      var_7_rf__reduce_init[i_j_k_fused_6] = 0.00000000f;
      for (int32_t reduce_k_0_3 = 0; reduce_k_0_3 < 1; reduce_k_0_3 += 1) {
        float var_1_local_2 = var_1[(((((((((((int)blockIdx.x * 8ll) + i_j_k_fused_6) * 8ll) + (int)threadIdx.y) / (S3 * S4)) * 32ll) + (int)threadIdx.x) * S3) * S4) + ((((((int)blockIdx.x * 8ll) + i_j_k_fused_6) * 8ll) + (int)threadIdx.y) % (S3 * S4)))];
        float var_local_13 = var[(int)threadIdx.x];
        var_7_rf[i_j_k_fused_6] = (var_7_rf[i_j_k_fused_6] + (var_1_local_2 + var_local_13));
      };
    };
  };
  for (int32_t i_j_k_fused_6 = 0ll; i_j_k_fused_6 < 8ll; i_j_k_fused_6 += 1) {
    if (((((((int)blockIdx.x * 8ll) + i_j_k_fused_6) * 8ll) + (int)threadIdx.y) < (S3 * S4))) {
      var_7[i_j_k_fused_6] = cinn_block_reduce_sum_fp32(var_7_rf[i_j_k_fused_6], shm32__fp32_reduce, true);
    };
  };
  for (int32_t i_j_append_var_1_fused_0 = 0ll; i_j_append_var_1_fused_0 < 8ll; i_j_append_var_1_fused_0 += 1) {
    if (((((((int)blockIdx.x * 8ll) + i_j_append_var_1_fused_0) * 8ll) + (int)threadIdx.y) < (S3 * S4))) {
      var_16_loopalign_1_rf__reduce_init[i_j_append_var_1_fused_0] = 0.00000000f;
      for (int32_t reduce_k_0_4 = 0; reduce_k_0_4 < 1; reduce_k_0_4 += 1) {
        float var_1_local_3 = var_1[(((((((int)blockIdx.x * 8ll) + i_j_append_var_1_fused_0) * 8ll) + (int)threadIdx.y) % (S3 * S4)) + ((S3 * (int)threadIdx.x) * S4))];
        float var_local_14 = var[(int)threadIdx.x];
        var_16_loopalign_1_rf[i_j_append_var_1_fused_0] = (var_16_loopalign_1_rf[i_j_append_var_1_fused_0] + (((var_1_local_3 + var_local_14) - (var_7[i_j_append_var_1_fused_0] / 32.0000000f)) * ((var_1_local_3 + var_local_14) - (var_7[i_j_append_var_1_fused_0] / 32.0000000f))));
      };
    };
  };
  for (int32_t i_j_append_var_1_fused_0 = 0ll; i_j_append_var_1_fused_0 < 8ll; i_j_append_var_1_fused_0 += 1) {
    if (((((((int)blockIdx.x * 8ll) + i_j_append_var_1_fused_0) * 8ll) + (int)threadIdx.y) < (S3 * S4))) {
      var_16_loopalign_1[i_j_append_var_1_fused_0] = cinn_block_reduce_sum_fp32(var_16_loopalign_1_rf[i_j_append_var_1_fused_0], shm32__fp32_reduce, true);
    };
  };
  for (int32_t i_j_append_var_2_fused_3 = 0ll; i_j_append_var_2_fused_3 < 8ll; i_j_append_var_2_fused_3 += 1) {
    if (((((((int)blockIdx.x * 8ll) + i_j_append_var_2_fused_3) * 8ll) + (int)threadIdx.y) < (S3 * S4))) {
      float var_1_local_4 = var_1[(((((((((((int)blockIdx.x * 8ll) + i_j_append_var_2_fused_3) * 8ll) + (int)threadIdx.y) / (S3 * S4)) * 32ll) + (int)threadIdx.x) * S3) * S4) + ((((((int)blockIdx.x * 8ll) + i_j_append_var_2_fused_3) * 8ll) + (int)threadIdx.y) % (S3 * S4)))];
      float var_local_15 = var[(int)threadIdx.x];
      float var_30_local = var_30[(int)threadIdx.x];
      float var_34_local = var_34[(int)threadIdx.x];
      var_36[(((((((int)blockIdx.x * 8ll) + i_j_append_var_2_fused_3) * 8ll) + (int)threadIdx.y) * 32ll) + (int)threadIdx.x)] = (((((var_1_local_4 + var_local_15) - (var_7[i_j_append_var_2_fused_3] / 32.0000000f)) * cinn_nvgpu_rsqrt_fp32(((var_16_loopalign_1[i_j_append_var_2_fused_3] / 32.0000000f) + 9.99999975e-06f))) * var_30_local) + var_34_local);
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_transpose_gs_reshape__COND__FPA__FPA__FPA__FPA_S0MULS1_BPA_MUL32ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S0MULS1_BPA_MUL32ll_BPA_LE1023ll_BPA__BPA___kernel(const float* __restrict__ var, const int32_t* __restrict__ var_1, const int32_t* __restrict__ var_2, float* __restrict__ var_4, int32_t S0, int32_t S1)
{
  __builtin_assume(((int)blockIdx.x < ((S0 * S1) * 32)));
  var_4[(int)blockIdx.x] = var[((((((int)blockIdx.x / 2097152) * 65536) + ((int)blockIdx.x & 65535)) * 32) + (((int)blockIdx.x & 2097151) / 65536))];
}__global__
void __launch_bounds__(1024) fn_transpose_gs_reshape__COND__FPA__FPA__FPA_S0MULS1_BPA_MUL32ll_BPA_GE1024ll_BPA___kernel(const float* __restrict__ var, const int32_t* __restrict__ var_1, const int32_t* __restrict__ var_2, float* __restrict__ var_4, int32_t S0, int32_t S1)
{
  __builtin_assume(((int)blockIdx.x < ((((S0 * S1) * 32) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_6 = 0; i_j_k_a_fused_6 < 4; i_j_k_a_fused_6 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) < ((S0 * S1) * 32))) {
      float var_local_16 = var[((((((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) / 2097152) * 65536) + ((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) & 65535)) * 32) + (((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) & 2097151) / 65536))];
      var_4[(((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x)] = var_local_16;
    };
  };
}

}
