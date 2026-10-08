
extern "C" {

__global__
void __launch_bounds__(256) fn_full_full_full_arange_reshape_bc_bc_equal_cast_full_full_full_arange_gather_reshape_bc_mul__COND_true__kernel(const int32_t* __restrict__ var_3, const float* __restrict__ var_14, float* __restrict__ var_17)
{
  __builtin_assume(((int)blockIdx.x < 322));
  __builtin_assume(((int)threadIdx.x < 256));
  if (((((int)blockIdx.x * 256) + (int)threadIdx.x) < 82320)) {
    var_17[(((int)blockIdx.x * 256) + (int)threadIdx.x)] = (((float)((((int64_t)(var_3[((((int)blockIdx.x * 256) + (int)threadIdx.x) / 20)])) == (0ll + (1ll * (0ll + (1ll * ((((int)blockIdx.x * 256ll) + (int)threadIdx.x) % 20ll)))))))) * var_14[((((int)blockIdx.x * 256) + (int)threadIdx.x) / 20)]);
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_reshape_reshape_gather_reshape__COND_true__kernel(const float* __restrict__ var, const int64_t* __restrict__ var_1, float* __restrict__ var_4)
{
  __builtin_assume(((int)blockIdx.x < 65));
  __builtin_assume(((int)threadIdx.x < 256));
  if (((((int)blockIdx.x * 256) + (int)threadIdx.x) < 16464)) {
    var_4[(((int)blockIdx.x * 256) + (int)threadIdx.x)] = var[((((int32_t)(var_1[((((int)blockIdx.x * 256) + (int)threadIdx.x) / 4)])) * 4) + ((int)threadIdx.x & 3))];
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_argmax_gs_bc_add__COND_true__kernel(const float* __restrict__ var, const int64_t* __restrict__ var_1, int64_t* __restrict__ var_4, int32_t S0)
{
  __builtin_assume(((int)blockIdx.x < 17));
  __builtin_assume(((int)threadIdx.y < 256));
  argidx_fp32_i64 _var_0_rf_temp_buffer [ 1 ];
  argidx_fp32_i64 _var_0_temp_buffer [ 1 ];
  argidx_fp32_i64* var_0 = _var_0_temp_buffer;
  argidx_fp32_i64* var_0_rf = _var_0_rf_temp_buffer;
  argidx_fp32_i64* var_0_rf__reduce_init = _var_0_rf_temp_buffer;
  if (((((int)blockIdx.x * 256) + (int)threadIdx.y) < 4116)) {
    var_0_rf__reduce_init[0] = argidx_fp32_i64(-3.40282347e+38f, 0ll);
    for (int32_t reduce_k_0_3 = 0; reduce_k_0_3 < 1; reduce_k_0_3 += 1) {
      float var_local = var[(((int)blockIdx.x * 256) + (int)threadIdx.y)];
      var_0_rf[0] = max(var_0_rf[0], argidx_fp32_i64(var_local, 0));
    };
    var_0[0] = var_0_rf[0];
    if (((int)threadIdx.x == 0)) {
      var_4[(((int)blockIdx.x * 256) + (int)threadIdx.y)] = (var_1[0] + ((int64_t)(var_0[0])));
    };
  };
}

}

extern "C" {

__global__
void __launch_bounds__(256) fn_gs_reshape_gather_reshape_full_greater_than_full_select_scale_bc_div_bc_mul_r_max__COND_true__kernel(const int32_t* __restrict__ var, const int64_t* __restrict__ var_1, const float* __restrict__ var_6, const float* __restrict__ var_10, const float* __restrict__ var_13, const float* __restrict__ var_15, int32_t* __restrict__ var_9, float* __restrict__ var_18, int32_t S1)
{
  __builtin_assume(((int)blockIdx.x < 17));
  __builtin_assume(((int)threadIdx.y < 256));
  float _var_18_rf_temp_buffer [ 1 ];
  float* var_18_rf = _var_18_rf_temp_buffer;
  float* var_18_rf__reduce_init = _var_18_rf_temp_buffer;
  if (((((int)blockIdx.x * 256) + (int)threadIdx.y) < 4116)) {
    if (((int)threadIdx.x == 0)) {
      var_9[(((int)blockIdx.x * 256) + (int)threadIdx.y)] = (((var_6[(((int)blockIdx.x * 256) + (int)threadIdx.y)] > 0.00000000f)) ? ((int32_t)(((int64_t)(var[0])))) : 20);
    };
    var_18_rf__reduce_init[0] = -3.40282347e+38f;
    for (int32_t reduce_k_0_2 = 0; reduce_k_0_2 < 1; reduce_k_0_2 += 1) {
      float var_13_local = var_13[(((int)blockIdx.x * 256) + (int)threadIdx.y)];
      float var_10_local = var_10[0];
      float var_15_local = var_15[0];
      var_18_rf[0] = max(var_18_rf[0], ((var_13_local / (var_10_local + 9.99999972e-10f)) * var_15_local));
    };
    var_18[(((int)blockIdx.x * 256) + (int)threadIdx.y)] = var_18_rf[0];
  };
}

}

extern "C" {

__global__
void __launch_bounds__(1024) fn_mul_r_max_mul_r_max__COND_true__kernel(const float* __restrict__ var, const float* __restrict__ var_0, const float* __restrict__ var_3, float* __restrict__ var_2, float* __restrict__ var_5, float* __restrict__ var_1)
{
  __builtin_assume(((int)threadIdx.x < 1024));
  float _var_2_rf_temp_buffer [ 1 ];
  float _var_5_rf_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  float* var_2_rf = _var_2_rf_temp_buffer;
  float* var_2_rf__reduce_init = _var_2_rf_temp_buffer;
  float* var_5_rf = _var_5_rf_temp_buffer;
  float* var_5_rf__reduce_init = _var_5_rf_temp_buffer;
  var_5_rf__reduce_init[0] = -3.40282347e+38f;
  var_2_rf__reduce_init[0] = -3.40282347e+38f;
  for (int32_t reduce_k_0_0 = 0; reduce_k_0_0 < 5; reduce_k_0_0 += 1) {
    if ((((reduce_k_0_0 * 1024) + (int)threadIdx.x) < 4116)) {
      float var_3_local = var_3[((reduce_k_0_0 * 1024) + (int)threadIdx.x)];
      float var_0_local = var_0[((reduce_k_0_0 * 1024) + (int)threadIdx.x)];
      float var_local_0 = var[((reduce_k_0_0 * 1024) + (int)threadIdx.x)];
      var_5_rf[0] = max(var_5_rf[0], (var_3_local * var_0_local));
      var_1[((reduce_k_0_0 * 1024) + (int)threadIdx.x)] = (var_local_0 * var_0_local);
      var_2_rf[0] = max(var_2_rf[0], (var_local_0 * var_0_local));
    };
  };
  var_2[0] = cinn_block_reduce_max_fp32(var_2_rf[0], shm32__fp32_reduce, false);
  var_5[0] = cinn_block_reduce_max_fp32(var_5_rf[0], shm32__fp32_reduce, false);
}

}
