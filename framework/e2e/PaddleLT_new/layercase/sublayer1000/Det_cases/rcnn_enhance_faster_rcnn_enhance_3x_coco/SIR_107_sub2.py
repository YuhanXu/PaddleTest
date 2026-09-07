# CINN Kernel 5: fn_add_scale_add_scale_sub_sub_max_max_sub_sub_mul_div_atan
# 预测 bbox 处理: center_x, center_y, width, height, max, area, atan(w/h)
# 融合类型: Elementwise(atan, mul, max) + Broadcast(add, sub, div) + Injective(scale=÷2)
# 对应原始 SIR_107.py 中 var_0 split 后的 pred bbox 处理部分
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,    # (shape: [24], dtype: paddle.float32, stop_gradient: True)
    ):
        out = paddle.tensor.manipulation.split(var_0, num_or_sections=4, axis=-1)
        var_2 = out[0]
        var_3 = out[1]
        var_4 = out[2]
        var_5 = out[3]
        var_10 = var_2.__add__(var_4)
        var_11 = var_10.__truediv__(2)
        var_12 = var_3.__add__(var_5)
        var_13 = var_12.__truediv__(2)
        var_14 = var_4.__sub__(var_2)
        var_15 = var_5.__sub__(var_3)
        var_22 = paddle.tensor.math.maximum(var_2, var_4)
        var_23 = paddle.tensor.math.maximum(var_3, var_5)
        var_39 = var_22.__sub__(var_2)
        var_40 = var_23.__sub__(var_3)
        var_41 = var_39.__mul__(var_40)
        var_67 = var_14.__truediv__(var_15)
        var_69 = paddle.tensor.ops.atan(var_67)
        return var_11, var_13, var_14, var_15, var_22, var_23, var_41, var_69


def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1,), dtype=paddle.float32, stop_gradient=True),
    )
    return inputspec


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[24], dtype=paddle.float32),
    )
    inputs[0].stop_gradient = True
    return inputs


def create_numpy_inputs():
    inputs = (
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
