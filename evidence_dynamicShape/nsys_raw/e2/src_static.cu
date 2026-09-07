
extern "C" {

__global__
void __launch_bounds__(32) fn_reshape_add_full_max__10342299779483483015_COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_1, float* __restrict__ var_4)
{
  __builtin_assume(((int)threadIdx.x < 32));
  if (((int)blockIdx.x == 0)) {
    if (((int)threadIdx.x < 18)) {
      var_4[(int)threadIdx.x] = max((var_1[(int)threadIdx.x] + var[(int)threadIdx.x]), 0.00000000f);
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_reshape_add_full_mul_full_add_full_min_full_max_bc_mul__2053255021606788863_COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_1, const float* __restrict__ var_12, float* __restrict__ var_13)
{
  __builtin_assume(((int)blockIdx.x < 8));
  __builtin_assume(((int)blockIdx.y < 72));
  __builtin_assume(((int)threadIdx.x < 256));
  for (int32_t i_k_a_fused_fused_0 = 0; i_k_a_fused_fused_0 < 4; i_k_a_fused_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4) + i_k_a_fused_fused_0) * 256) + (int)threadIdx.x) < 7744)) {
      float var_12_local = var_12[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (i_k_a_fused_fused_0 * 256)) + ((int)blockIdx.y * 7744))];
      float var_1_local = var_1[(int)blockIdx.y];
      float var_local = var[(int)blockIdx.y];
      var_13[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (i_k_a_fused_fused_0 * 256)) + ((int)blockIdx.y * 7744))] = (var_12_local * max(min((((var_1_local + var_local) * 0.166666701f) + 0.500000000f), 1.00000000f), 0.00000000f));
    };
  };
}

}
