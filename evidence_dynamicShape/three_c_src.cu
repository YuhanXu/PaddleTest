
extern "C" {

__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_gs_bc_max__11008249055510313070_COND__FPA__FPA__FPA__FPA_S0MUL18ll_BPA_GE1ll_BPA_AND_FPA__FPA_S0MUL18ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S0MUL18ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int32_t S0)
{
  __builtin_assume(((int)blockIdx.x < (S0 * 18)));
  var_8[(int)blockIdx.x] = max((var_1[(int)blockIdx.x] + var[((int)blockIdx.x % 18)]), 0.00000000f);
}__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_gs_bc_max__11008249055510313070_COND__FPA__FPA__FPA__FPA_S0MUL18ll_BPA_GE1ll_BPA_AND_FPA__FPA_S0MUL18ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S0MUL18ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int64_t S0)
{
  __builtin_assume(((int)blockIdx.x < (S0 * 18ll)));
  var_8[(int)blockIdx.x] = max((var_1[(int)blockIdx.x] + var[((int)blockIdx.x % 18ll)]), 0.00000000f);
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_gs_bc_max__11008249055510313070_COND__FPA__FPA__FPA_S0MUL18ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S0MUL18ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int32_t S0)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * 18) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_0 = 0; i_j_k_a_fused_0 < 4; i_j_k_a_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) < (S0 * 18))) {
      float var_1_local = var_1[(((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x)];
      float var_local = var[((((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x) % 18)];
      var_8[(((((int)blockIdx.x * 4) + i_j_k_a_fused_0) * 1024) + (int)threadIdx.x)] = max((var_1_local + var_local), 0.00000000f);
    };
  };
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_gs_bc_max__11008249055510313070_COND__FPA__FPA__FPA_S0MUL18ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S0MUL18ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int64_t S0)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * 18ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_fused_0 = 0ll; i_j_k_a_fused_0 < 4ll; i_j_k_a_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) < (S0 * 18ll))) {
      float var_1_local = var_1[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x)];
      float var_local = var[((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x) % 18ll)];
      var_8[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_0) * 1024ll) + (int)threadIdx.x)] = max((var_1_local + var_local), 0.00000000f);
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__17594470531925998404_COND__FPA__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA__FPA_S0MUL72ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int32_t S0, int32_t S1, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * S1) * S2) * 72)));
  var_24[(int)blockIdx.x] = (var_9[(int)blockIdx.x] * max(min((((var_1[((((int)blockIdx.x % ((S1 * S2) * 72)) / (S1 * S2)) + (((int)blockIdx.x / ((S1 * S2) * 72)) * 72))] + var[(((int)blockIdx.x % ((S1 * S2) * 72)) / (S1 * S2))]) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
}__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__17594470531925998404_COND__FPA__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE1023ll_BPA__BPA_ANDNOT_FPA__FPA__FPA_S0MUL72ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int64_t S0, int64_t S1, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * S1) * S2) * 72ll)));
  var_24[(int)blockIdx.x] = (var_9[(int)blockIdx.x] * max(min((((var_1[((((int)blockIdx.x % ((S1 * S2) * 72ll)) / (S1 * S2)) + (((int)blockIdx.x / ((S1 * S2) * 72ll)) * 72ll))] + var[(((int)blockIdx.x % ((S1 * S2) * 72ll)) / (S1 * S2))]) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__17594470531925998404_COND__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_GE1024ll_BPA_AND_FPA__FPA__FPA_S0MUL72ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int32_t S0, int32_t S1, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((((S0 * S1) * S2) * 72) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_3 = 0; i_j_k_a_fused_3 < 4; i_j_k_a_fused_3 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_3) * 1024) + (int)threadIdx.x) < (((S0 * S1) * S2) * 72))) {
      float var_9_local = var_9[(((((int)blockIdx.x * 4) + i_j_k_a_fused_3) * 1024) + (int)threadIdx.x)];
      float var_1_local_0 = var_1[((((((((int)blockIdx.x * 4) + i_j_k_a_fused_3) * 1024) + (int)threadIdx.x) % ((S1 * S2) * 72)) / (S1 * S2)) + (((((((int)blockIdx.x * 4) + i_j_k_a_fused_3) * 1024) + (int)threadIdx.x) / ((S1 * S2) * 72)) * 72))];
      float var_local_0 = var[(((((((int)blockIdx.x * 4) + i_j_k_a_fused_3) * 1024) + (int)threadIdx.x) % ((S1 * S2) * 72)) / (S1 * S2))];
      var_24[(((((int)blockIdx.x * 4) + i_j_k_a_fused_3) * 1024) + (int)threadIdx.x)] = (var_9_local * max(min((((var_1_local_0 + var_local_0) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
    };
  };
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__17594470531925998404_COND__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_GE1024ll_BPA_ANDNOT_FPA__FPA__FPA_S0MUL72ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL72ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int64_t S0, int64_t S1, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((((S0 * S1) * S2) * 72ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_fused_3 = 0ll; i_j_k_a_fused_3 < 4ll; i_j_k_a_fused_3 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_3) * 1024ll) + (int)threadIdx.x) < (((S0 * S1) * S2) * 72ll))) {
      float var_9_local = var_9[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_3) * 1024ll) + (int)threadIdx.x)];
      float var_1_local_0 = var_1[((((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_3) * 1024ll) + (int)threadIdx.x) % ((S1 * S2) * 72ll)) / (S1 * S2)) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_3) * 1024ll) + (int)threadIdx.x) / ((S1 * S2) * 72ll)) * 72ll))];
      float var_local_0 = var[(((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_3) * 1024ll) + (int)threadIdx.x) % ((S1 * S2) * 72ll)) / (S1 * S2))];
      var_24[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_3) * 1024ll) + (int)threadIdx.x)] = (var_9_local * max(min((((var_1_local_0 + var_local_0) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_gs_bc_max__3090994648363325756_COND__FPA__FPA__FPA__FPA_S0MUL14ll_BPA_GE1ll_BPA_AND_FPA__FPA_S0MUL14ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S0MUL14ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int32_t S0)
{
  __builtin_assume(((int)blockIdx.x < (S0 * 14)));
  var_8[(int)blockIdx.x] = max((var_1[(int)blockIdx.x] + var[((int)blockIdx.x % 14)]), 0.00000000f);
}__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_gs_bc_max__3090994648363325756_COND__FPA__FPA__FPA__FPA_S0MUL14ll_BPA_GE1ll_BPA_AND_FPA__FPA_S0MUL14ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S0MUL14ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int64_t S0)
{
  __builtin_assume(((int)blockIdx.x < (S0 * 14ll)));
  var_8[(int)blockIdx.x] = max((var_1[(int)blockIdx.x] + var[((int)blockIdx.x % 14ll)]), 0.00000000f);
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_gs_bc_max__3090994648363325756_COND__FPA__FPA__FPA_S0MUL14ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S0MUL14ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int32_t S0)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * 14) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_6 = 0; i_j_k_a_fused_6 < 4; i_j_k_a_fused_6 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) < (S0 * 14))) {
      float var_1_local_1 = var_1[(((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x)];
      float var_local_1 = var[((((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x) % 14)];
      var_8[(((((int)blockIdx.x * 4) + i_j_k_a_fused_6) * 1024) + (int)threadIdx.x)] = max((var_1_local_1 + var_local_1), 0.00000000f);
    };
  };
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_gs_bc_max__3090994648363325756_COND__FPA__FPA__FPA_S0MUL14ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S0MUL14ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int64_t S0)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * 14ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_fused_6 = 0ll; i_j_k_a_fused_6 < 4ll; i_j_k_a_fused_6 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_6) * 1024ll) + (int)threadIdx.x) < (S0 * 14ll))) {
      float var_1_local_1 = var_1[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_6) * 1024ll) + (int)threadIdx.x)];
      float var_local_1 = var[((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_6) * 1024ll) + (int)threadIdx.x) % 14ll)];
      var_8[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_6) * 1024ll) + (int)threadIdx.x)] = max((var_1_local_1 + var_local_1), 0.00000000f);
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__4704868700554850640_COND__FPA__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL56ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL56ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA__FPA_S0MUL56ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL56ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int32_t S0, int32_t S1, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * S1) * S2) * 56)));
  var_24[(int)blockIdx.x] = (var_9[(int)blockIdx.x] * max(min((((var_1[((((int)blockIdx.x % ((S1 * S2) * 56)) / (S1 * S2)) + (((int)blockIdx.x / ((S1 * S2) * 56)) * 56))] + var[(((int)blockIdx.x % ((S1 * S2) * 56)) / (S1 * S2))]) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
}__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__4704868700554850640_COND__FPA__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL56ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL56ll_BPA_LE1023ll_BPA__BPA_ANDNOT_FPA__FPA__FPA_S0MUL56ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL56ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int64_t S0, int64_t S1, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * S1) * S2) * 56ll)));
  var_24[(int)blockIdx.x] = (var_9[(int)blockIdx.x] * max(min((((var_1[((((int)blockIdx.x % ((S1 * S2) * 56ll)) / (S1 * S2)) + (((int)blockIdx.x / ((S1 * S2) * 56ll)) * 56ll))] + var[(((int)blockIdx.x % ((S1 * S2) * 56ll)) / (S1 * S2))]) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__4704868700554850640_COND__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL56ll_BPA_GE1024ll_BPA_AND_FPA__FPA__FPA_S0MUL56ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL56ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int32_t S0, int32_t S1, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((((S0 * S1) * S2) * 56) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_9 = 0; i_j_k_a_fused_9 < 4; i_j_k_a_fused_9 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_9) * 1024) + (int)threadIdx.x) < (((S0 * S1) * S2) * 56))) {
      float var_9_local_0 = var_9[(((((int)blockIdx.x * 4) + i_j_k_a_fused_9) * 1024) + (int)threadIdx.x)];
      float var_1_local_2 = var_1[((((((((int)blockIdx.x * 4) + i_j_k_a_fused_9) * 1024) + (int)threadIdx.x) % ((S1 * S2) * 56)) / (S1 * S2)) + (((((((int)blockIdx.x * 4) + i_j_k_a_fused_9) * 1024) + (int)threadIdx.x) / ((S1 * S2) * 56)) * 56))];
      float var_local_2 = var[(((((((int)blockIdx.x * 4) + i_j_k_a_fused_9) * 1024) + (int)threadIdx.x) % ((S1 * S2) * 56)) / (S1 * S2))];
      var_24[(((((int)blockIdx.x * 4) + i_j_k_a_fused_9) * 1024) + (int)threadIdx.x)] = (var_9_local_0 * max(min((((var_1_local_2 + var_local_2) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
    };
  };
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__4704868700554850640_COND__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL56ll_BPA_GE1024ll_BPA_ANDNOT_FPA__FPA__FPA_S0MUL56ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL56ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int64_t S0, int64_t S1, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((((S0 * S1) * S2) * 56ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_fused_9 = 0ll; i_j_k_a_fused_9 < 4ll; i_j_k_a_fused_9 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_9) * 1024ll) + (int)threadIdx.x) < (((S0 * S1) * S2) * 56ll))) {
      float var_9_local_0 = var_9[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_9) * 1024ll) + (int)threadIdx.x)];
      float var_1_local_2 = var_1[((((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_9) * 1024ll) + (int)threadIdx.x) % ((S1 * S2) * 56ll)) / (S1 * S2)) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_9) * 1024ll) + (int)threadIdx.x) / ((S1 * S2) * 56ll)) * 56ll))];
      float var_local_2 = var[(((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_9) * 1024ll) + (int)threadIdx.x) % ((S1 * S2) * 56ll)) / (S1 * S2))];
      var_24[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_9) * 1024ll) + (int)threadIdx.x)] = (var_9_local_0 * max(min((((var_1_local_2 + var_local_2) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_gs_bc_max__4826633601978809597_COND__FPA__FPA__FPA__FPA_S0MUL11ll_BPA_GE1ll_BPA_AND_FPA__FPA_S0MUL11ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S0MUL11ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int32_t S0)
{
  __builtin_assume(((int)blockIdx.x < (S0 * 11)));
  var_8[(int)blockIdx.x] = max((var_1[(int)blockIdx.x] + var[((int)blockIdx.x % 11)]), 0.00000000f);
}__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_gs_bc_max__4826633601978809597_COND__FPA__FPA__FPA__FPA_S0MUL11ll_BPA_GE1ll_BPA_AND_FPA__FPA_S0MUL11ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA_S0MUL11ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int64_t S0)
{
  __builtin_assume(((int)blockIdx.x < (S0 * 11ll)));
  var_8[(int)blockIdx.x] = max((var_1[(int)blockIdx.x] + var[((int)blockIdx.x % 11ll)]), 0.00000000f);
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_gs_bc_max__4826633601978809597_COND__FPA__FPA__FPA_S0MUL11ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S0MUL11ll_BPA_LE2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int32_t S0)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * 11) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_12 = 0; i_j_k_a_fused_12 < 4; i_j_k_a_fused_12 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_12) * 1024) + (int)threadIdx.x) < (S0 * 11))) {
      float var_1_local_3 = var_1[(((((int)blockIdx.x * 4) + i_j_k_a_fused_12) * 1024) + (int)threadIdx.x)];
      float var_local_3 = var[((((((int)blockIdx.x * 4) + i_j_k_a_fused_12) * 1024) + (int)threadIdx.x) % 11)];
      var_8[(((((int)blockIdx.x * 4) + i_j_k_a_fused_12) * 1024) + (int)threadIdx.x)] = max((var_1_local_3 + var_local_3), 0.00000000f);
    };
  };
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_gs_bc_max__4826633601978809597_COND__FPA__FPA__FPA_S0MUL11ll_BPA_GE1024ll_BPA_AND_FPA__FPA_S0MUL11ll_BPA_GT2147483647ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_8, int64_t S0)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * 11ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_fused_12 = 0ll; i_j_k_a_fused_12 < 4ll; i_j_k_a_fused_12 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_12) * 1024ll) + (int)threadIdx.x) < (S0 * 11ll))) {
      float var_1_local_3 = var_1[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_12) * 1024ll) + (int)threadIdx.x)];
      float var_local_3 = var[((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_12) * 1024ll) + (int)threadIdx.x) % 11ll)];
      var_8[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_12) * 1024ll) + (int)threadIdx.x)] = max((var_1_local_3 + var_local_3), 0.00000000f);
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__715298721637821971_COND__FPA__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL44ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL44ll_BPA_LE1023ll_BPA__BPA_AND_FPA__FPA__FPA_S0MUL44ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL44ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int32_t S0, int32_t S1, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * S1) * S2) * 44)));
  var_24[(int)blockIdx.x] = (var_9[(int)blockIdx.x] * max(min((((var_1[((((int)blockIdx.x % ((S1 * S2) * 44)) / (S1 * S2)) + (((int)blockIdx.x / ((S1 * S2) * 44)) * 44))] + var[(((int)blockIdx.x % ((S1 * S2) * 44)) / (S1 * S2))]) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
}__global__
void __launch_bounds__(1) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__715298721637821971_COND__FPA__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL44ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL44ll_BPA_LE1023ll_BPA__BPA_ANDNOT_FPA__FPA__FPA_S0MUL44ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL44ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int64_t S0, int64_t S1, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((S0 * S1) * S2) * 44ll)));
  var_24[(int)blockIdx.x] = (var_9[(int)blockIdx.x] * max(min((((var_1[((((int)blockIdx.x % ((S1 * S2) * 44ll)) / (S1 * S2)) + (((int)blockIdx.x / ((S1 * S2) * 44ll)) * 44ll))] + var[(((int)blockIdx.x % ((S1 * S2) * 44ll)) / (S1 * S2))]) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__715298721637821971_COND__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL44ll_BPA_GE1024ll_BPA_AND_FPA__FPA__FPA_S0MUL44ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL44ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int32_t S0, int32_t S1, int32_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((((S0 * S1) * S2) * 44) / 4096) + 1)));
  __builtin_assume(((int)threadIdx.x < 1024));
  for (int32_t i_j_k_a_fused_15 = 0; i_j_k_a_fused_15 < 4; i_j_k_a_fused_15 += 1) {
    if (((((((int)blockIdx.x * 4) + i_j_k_a_fused_15) * 1024) + (int)threadIdx.x) < (((S0 * S1) * S2) * 44))) {
      float var_9_local_1 = var_9[(((((int)blockIdx.x * 4) + i_j_k_a_fused_15) * 1024) + (int)threadIdx.x)];
      float var_1_local_4 = var_1[((((((((int)blockIdx.x * 4) + i_j_k_a_fused_15) * 1024) + (int)threadIdx.x) % ((S1 * S2) * 44)) / (S1 * S2)) + (((((((int)blockIdx.x * 4) + i_j_k_a_fused_15) * 1024) + (int)threadIdx.x) / ((S1 * S2) * 44)) * 44))];
      float var_local_4 = var[(((((((int)blockIdx.x * 4) + i_j_k_a_fused_15) * 1024) + (int)threadIdx.x) % ((S1 * S2) * 44)) / (S1 * S2))];
      var_24[(((((int)blockIdx.x * 4) + i_j_k_a_fused_15) * 1024) + (int)threadIdx.x)] = (var_9_local_1 * max(min((((var_1_local_4 + var_local_4) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
    };
  };
}__global__
void __launch_bounds__(1024) fn_reshape_gs_bc_add_full_full_full_full_gs_bc_mul_gs_bc_add_gs_bc_min_gs_bc_max_gs_bc_mul__715298721637821971_COND__FPA__FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL44ll_BPA_GE1024ll_BPA_ANDNOT_FPA__FPA__FPA_S0MUL44ll_BPA_LE2147483647ll_BPA_AND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MULS2_BPA_MUL44ll_BPA_LE2147483647ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_9, float* __restrict__ var_24, int64_t S0, int64_t S1, int64_t S2)
{
  __builtin_assume(((int)blockIdx.x < (((((S0 * S1) * S2) * 44ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  for (int32_t i_j_k_a_fused_15 = 0ll; i_j_k_a_fused_15 < 4ll; i_j_k_a_fused_15 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_15) * 1024ll) + (int)threadIdx.x) < (((S0 * S1) * S2) * 44ll))) {
      float var_9_local_1 = var_9[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_15) * 1024ll) + (int)threadIdx.x)];
      float var_1_local_4 = var_1[((((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_15) * 1024ll) + (int)threadIdx.x) % ((S1 * S2) * 44ll)) / (S1 * S2)) + (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_15) * 1024ll) + (int)threadIdx.x) / ((S1 * S2) * 44ll)) * 44ll))];
      float var_local_4 = var[(((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_15) * 1024ll) + (int)threadIdx.x) % ((S1 * S2) * 44ll)) / (S1 * S2))];
      var_24[(((((int)blockIdx.x * 4ll) + i_j_k_a_fused_15) * 1024ll) + (int)threadIdx.x)] = (var_9_local_1 * max(min((((var_1_local_4 + var_local_4) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
    };
  };
}

}
