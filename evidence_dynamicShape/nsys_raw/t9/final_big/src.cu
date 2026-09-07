
extern "C" {

__global__
void __launch_bounds__(32) fn_reshape_gs_bc_add_full_gs_bc_max__11008249055510313070_COND__FPA__FPA__FPA__FPA_S0MUL18ll_BPA_GE1ll_BPA_AND_FPA__FPA_S0MUL18ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S0MUL18ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int32_t S0)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * 18) / 32) + 1)));
  __builtin_assume(((int)threadIdx.x < 32));
  if (((((int)blockIdx.x * 32) + (int)threadIdx.x) < (S0 * 18))) {
    var_8[(((int)blockIdx.x * 32) + (int)threadIdx.x)] = max((var_1[(((int)blockIdx.x * 32) + (int)threadIdx.x)] + var[((((int)blockIdx.x * 32) + (int)threadIdx.x) % 18)]), 0.00000000f);
  };
}__global__
void __launch_bounds__(32) fn_reshape_gs_bc_add_full_gs_bc_max__11008249055510313070_COND__FPA__FPA__FPA__FPA_S0MUL18ll_BPA_GE1ll_BPA_AND_FPA__FPA_S0MUL18ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S0MUL18ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int64_t S0)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * 18ll) / 32ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 32ll));
  if (((((int)blockIdx.x * 32ll) + (int)threadIdx.x) < (S0 * 18ll))) {
    var_8[(((int)blockIdx.x * 32ll) + (int)threadIdx.x)] = max((var_1[(((int)blockIdx.x * 32ll) + (int)threadIdx.x)] + var[((((int)blockIdx.x * 32ll) + (int)threadIdx.x) % 18ll)]), 0.00000000f);
  };
}__global__
void __launch_bounds__(256) fn_reshape_gs_bc_add_full_gs_bc_max__11008249055510313070_COND__FPA__FPA__FPA_S0MUL18ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S0MUL18ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int32_t S0)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * 18) / 512) + 1)));
  __builtin_assume(((int)threadIdx.x < 256));
  for (int32_t i_j_k_a_fused_2 = 0; i_j_k_a_fused_2 < 2; i_j_k_a_fused_2 += 1) {
    if (((((((int)blockIdx.x * 2) + i_j_k_a_fused_2) * 256) + (int)threadIdx.x) < (S0 * 18))) {
      float var_1_local = var_1[(((((int)blockIdx.x * 2) + i_j_k_a_fused_2) * 256) + (int)threadIdx.x)];
      float var_local = var[((((((int)blockIdx.x * 2) + i_j_k_a_fused_2) * 256) + (int)threadIdx.x) % 18)];
      var_8[(((((int)blockIdx.x * 2) + i_j_k_a_fused_2) * 256) + (int)threadIdx.x)] = max((var_1_local + var_local), 0.00000000f);
    };
  };
}__global__
void __launch_bounds__(256) fn_reshape_gs_bc_add_full_gs_bc_max__11008249055510313070_COND__FPA__FPA__FPA_S0MUL18ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S0MUL18ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int64_t S0)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * 18ll) / 512ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 256ll));
  for (int32_t i_j_k_a_fused_2 = 0ll; i_j_k_a_fused_2 < 2ll; i_j_k_a_fused_2 += 1) {
    if (((((((int)blockIdx.x * 2ll) + i_j_k_a_fused_2) * 256ll) + (int)threadIdx.x) < (S0 * 18ll))) {
      float var_1_local = var_1[(((((int)blockIdx.x * 2ll) + i_j_k_a_fused_2) * 256ll) + (int)threadIdx.x)];
      float var_local = var[((((((int)blockIdx.x * 2ll) + i_j_k_a_fused_2) * 256ll) + (int)threadIdx.x) % 18ll)];
      var_8[(((((int)blockIdx.x * 2ll) + i_j_k_a_fused_2) * 256ll) + (int)threadIdx.x)] = max((var_1_local + var_local), 0.00000000f);
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(32) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__17594470531925998404_COND__FPA__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA__FPA_S0MUL72ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int32_t S0, int32_t S1, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((((S0 * S1) * S2) * 72) / 32) + 1)));
  __builtin_assume(((int)threadIdx.x < 32));
  if (((((int)blockIdx.x * 32) + (int)threadIdx.x) < (((S0 * S1) * S2) * 72))) {
    var_24[(((int)blockIdx.x * 32) + (int)threadIdx.x)] = (var_9[(((int)blockIdx.x * 32) + (int)threadIdx.x)] * max(min((((var_1[((((((int)blockIdx.x * 32) + (int)threadIdx.x) % ((S1 * S2) * 72)) / (S1 * S2)) + (((((int)blockIdx.x * 32) + (int)threadIdx.x) / ((S1 * S2) * 72)) * 72))] + var[(((((int)blockIdx.x * 32) + (int)threadIdx.x) % ((S1 * S2) * 72)) / (S1 * S2))]) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
  };
}__global__
void __launch_bounds__(32) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__17594470531925998404_COND__FPA__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE1023ll_BPA__BPA_ANDNOT_FPA__FPA__FPA_S0MUL72ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int64_t S0, int64_t S1, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((((S0 * S1) * S2) * 72ll) / 32ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 32ll));
  if (((((int)blockIdx.x * 32ll) + (int)threadIdx.x) < (((S0 * S1) * S2) * 72ll))) {
    var_24[(((int)blockIdx.x * 32ll) + (int)threadIdx.x)] = (var_9[(((int)blockIdx.x * 32ll) + (int)threadIdx.x)] * max(min((((var_1[((((((int)blockIdx.x * 32ll) + (int)threadIdx.x) % ((S1 * S2) * 72ll)) / (S1 * S2)) + (((((int)blockIdx.x * 32ll) + (int)threadIdx.x) / ((S1 * S2) * 72ll)) * 72ll))] + var[(((((int)blockIdx.x * 32ll) + (int)threadIdx.x) % ((S1 * S2) * 72ll)) / (S1 * S2))]) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
  };
}__global__
void __launch_bounds__(256) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__17594470531925998404_COND__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_GE1024ll_BPA_AND_FPA__FPA__FPA_S0MUL72ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int32_t S0, int32_t S1, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((((S0 * S1) * S2) * 72) / 512) + 1)));
  __builtin_assume(((int)threadIdx.x < 256));
  for (int32_t i_j_k_a_fused_7 = 0; i_j_k_a_fused_7 < 2; i_j_k_a_fused_7 += 1) {
    if (((((((int)blockIdx.x * 2) + i_j_k_a_fused_7) * 256) + (int)threadIdx.x) < (((S0 * S1) * S2) * 72))) {
      float var_9_local = var_9[(((((int)blockIdx.x * 2) + i_j_k_a_fused_7) * 256) + (int)threadIdx.x)];
      float var_1_local_0 = var_1[((((((((int)blockIdx.x * 2) + i_j_k_a_fused_7) * 256) + (int)threadIdx.x) % ((S1 * S2) * 72)) / (S1 * S2)) + (((((((int)blockIdx.x * 2) + i_j_k_a_fused_7) * 256) + (int)threadIdx.x) / ((S1 * S2) * 72)) * 72))];
      float var_local_0 = var[(((((((int)blockIdx.x * 2) + i_j_k_a_fused_7) * 256) + (int)threadIdx.x) % ((S1 * S2) * 72)) / (S1 * S2))];
      var_24[(((((int)blockIdx.x * 2) + i_j_k_a_fused_7) * 256) + (int)threadIdx.x)] = (var_9_local * max(min((((var_1_local_0 + var_local_0) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
    };
  };
}__global__
void __launch_bounds__(256) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__17594470531925998404_COND__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_GE1024ll_BPA_ANDNOT_FPA__FPA__FPA_S0MUL72ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int64_t S0, int64_t S1, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((((S0 * S1) * S2) * 72ll) / 512ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 256ll));
  for (int32_t i_j_k_a_fused_7 = 0ll; i_j_k_a_fused_7 < 2ll; i_j_k_a_fused_7 += 1) {
    if (((((((int)blockIdx.x * 2ll) + i_j_k_a_fused_7) * 256ll) + (int)threadIdx.x) < (((S0 * S1) * S2) * 72ll))) {
      float var_9_local = var_9[(((((int)blockIdx.x * 2ll) + i_j_k_a_fused_7) * 256ll) + (int)threadIdx.x)];
      float var_1_local_0 = var_1[((((((((int)blockIdx.x * 2ll) + i_j_k_a_fused_7) * 256ll) + (int)threadIdx.x) % ((S1 * S2) * 72ll)) / (S1 * S2)) + (((((((int)blockIdx.x * 2ll) + i_j_k_a_fused_7) * 256ll) + (int)threadIdx.x) / ((S1 * S2) * 72ll)) * 72ll))];
      float var_local_0 = var[(((((((int)blockIdx.x * 2ll) + i_j_k_a_fused_7) * 256ll) + (int)threadIdx.x) % ((S1 * S2) * 72ll)) / (S1 * S2))];
      var_24[(((((int)blockIdx.x * 2ll) + i_j_k_a_fused_7) * 256ll) + (int)threadIdx.x)] = (var_9_local * max(min((((var_1_local_0 + var_local_0) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
    };
  };
}

}
