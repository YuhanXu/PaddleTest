# CINN Kernel 6: fn_add_scale_add_scale_sub_sub_mul_div_atan
# 目标 bbox 处理: center_x, center_y, width, height, area, atan(w/h)
# 融合类型: Elementwise(atan, mul) + Broadcast(add, sub, div) + Injective(scale=÷2)
# 对应原始 SIR_107.py 中 var_1 split 后的 target bbox 处理部分
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1,    # (shape: [24], dtype: paddle.float32, stop_gradient: True)
    ):
        out = paddle.tensor.manipulation.split(var_1, num_or_sections=4, axis=-1)
        var_6 = out[0]
        var_7 = out[1]
        var_8 = out[2]
        var_9 = out[3]
        var_16 = var_6.__add__(var_8)
        var_17 = var_16.__truediv__(2)
        var_18 = var_7.__add__(var_9)
        var_19 = var_18.__truediv__(2)
        var_20 = var_8.__sub__(var_6)
        var_21 = var_9.__sub__(var_7)
        var_44 = var_20.__mul__(var_21)
        var_66 = var_20.__truediv__(var_21)
        var_68 = paddle.tensor.ops.atan(var_66)
        return var_17, var_19, var_20, var_21, var_44, var_68


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
