# CINN Kernel 3 (backward): fn_add_scale_..._equal_cast_full_gs_bc_div_mul_greater_than_cast_mul_add_less_than_cast_mul_add_..._concat
# IoU 交集路径反向: 通过 max/min 的梯度路由 (equal/greater_than/less_than) + concat (split 反向)
# 仅包含 IoU 交集/并集计算部分, var_0 stop_gradient=False
# 反向中 max/min 梯度需要 equal/greater_than/less_than 条件选择, 最终 concat 回 grad_var_0
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,    # (shape: [24], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [24], dtype: paddle.float32, stop_gradient: True)
    ):
        out = paddle.tensor.manipulation.split(var_0, num_or_sections=4, axis=-1)
        var_2 = out[0]
        var_3 = out[1]
        var_4 = out[2]
        var_5 = out[3]
        out = paddle.tensor.manipulation.split(var_1, num_or_sections=4, axis=-1)
        var_6 = out[0]
        var_7 = out[1]
        var_8 = out[2]
        var_9 = out[3]
        # pred bbox max
        var_22 = paddle.tensor.math.maximum(var_2, var_4)
        var_23 = paddle.tensor.math.maximum(var_3, var_5)
        # intersection box
        var_24 = paddle.tensor.math.maximum(var_2, var_6)
        var_25 = paddle.tensor.math.maximum(var_3, var_7)
        var_26 = paddle.tensor.math.minimum(var_22, var_8)
        var_27 = paddle.tensor.math.minimum(var_23, var_9)
        var_28 = paddle.tensor.math.minimum(var_2, var_6)
        var_29 = paddle.tensor.math.minimum(var_3, var_7)
        var_30 = paddle.tensor.math.maximum(var_22, var_8)
        var_31 = paddle.tensor.math.maximum(var_23, var_9)
        # intersection area with clamping
        var_32 = var_26.__sub__(var_24)
        var_33 = var_27.__sub__(var_25)
        var_34 = var_32.__mul__(var_33)
        var_35 = paddle.tensor.logic.greater_than(var_26, var_24)
        var_36 = var_34.__mul__(paddle.cast(var_35, paddle.float32))
        var_37 = paddle.tensor.logic.greater_than(var_27, var_25)
        var_38 = var_36.__mul__(paddle.cast(var_37, paddle.float32))
        # union area
        var_39 = var_22.__sub__(var_2)
        var_40 = var_23.__sub__(var_3)
        var_41 = var_39.__mul__(var_40)
        var_42 = var_8.__sub__(var_6)
        var_43 = var_9.__sub__(var_7)
        var_44 = var_42.__mul__(var_43)
        var_45 = var_41.__add__(var_44)
        var_46 = var_45.__sub__(var_38)
        var_47 = var_46.__add__(1e-10)
        # IoU
        var_48 = var_38.__truediv__(var_47)
        # loss = mean(1 - IoU)
        var_loss = var_48.__rsub__(1)
        var_result = paddle.tensor.stat.mean(var_loss)
        return var_result


def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1,), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1,), dtype=paddle.float32, stop_gradient=True),
    )
    return inputspec


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[24], dtype=paddle.float32),
        paddle.rand(shape=[24], dtype=paddle.float32),
    )
    inputs[0].stop_gradient = False
    inputs[1].stop_gradient = True
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[24]).astype('float32'),
        np.random.random(size=[24]).astype('float32'),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_tensor_inputs()
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.base.core._set_prim_all_enabled(with_prim)
            if with_cinn:
                assert with_prim, "with_cinn=True but with_prim=False is unsupported"
                net = paddle.jit.to_static(net, backend="CINN", full_graph=True)
            else:
                net = paddle.jit.to_static(net, backend=None, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(self.net, to_static=True, with_prim=True, with_cinn=True)
        for st, cinn in zip(paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
