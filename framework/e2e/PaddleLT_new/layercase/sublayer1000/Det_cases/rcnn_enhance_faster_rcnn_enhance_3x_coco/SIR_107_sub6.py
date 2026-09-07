# CINN Kernel 4 (backward): fn_add_scale_add_scale_div_div_scale_mul_div_mul_mul_scale_..._concat
# 中心距离 + 宽高比路径反向: 较简单的梯度路径 (无 max/min 梯度路由) + concat (split 反向)
# 仅包含中心距离和宽高比计算部分, var_0 stop_gradient=False
# 反向梯度通过 atan, div, sub, mul 等运算传播, 最终 concat 回 grad_var_0
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
        # pred center
        var_10 = var_2.__add__(var_4)
        var_11 = var_10.__truediv__(2)
        var_12 = var_3.__add__(var_5)
        var_13 = var_12.__truediv__(2)
        # target center
        var_16 = var_6.__add__(var_8)
        var_17 = var_16.__truediv__(2)
        var_18 = var_7.__add__(var_9)
        var_19 = var_18.__truediv__(2)
        # center distance squared
        var_49 = var_11.__sub__(var_17)
        var_50 = var_11.__sub__(var_17)
        var_51 = var_49.__mul__(var_50)
        var_52 = var_13.__sub__(var_19)
        var_53 = var_13.__sub__(var_19)
        var_54 = var_52.__mul__(var_53)
        var_55 = var_51.__add__(var_54)
        # enclosing box diagonal squared
        var_22 = paddle.tensor.math.maximum(var_2, var_4)
        var_23 = paddle.tensor.math.maximum(var_3, var_5)
        var_28 = paddle.tensor.math.minimum(var_2, var_6)
        var_29 = paddle.tensor.math.minimum(var_3, var_7)
        var_30 = paddle.tensor.math.maximum(var_22, var_8)
        var_31 = paddle.tensor.math.maximum(var_23, var_9)
        var_56 = var_30.__sub__(var_28)
        var_57 = var_30.__sub__(var_28)
        var_58 = var_56.__mul__(var_57)
        var_59 = var_31.__sub__(var_29)
        var_60 = var_31.__sub__(var_29)
        var_61 = var_59.__mul__(var_60)
        var_62 = var_58.__add__(var_61)
        # center distance ratio
        var_63 = var_55.__add__(1e-10)
        var_64 = var_62.__add__(1e-10)
        var_65 = var_63.__truediv__(var_64)
        # aspect ratio
        var_14 = var_4.__sub__(var_2)
        var_15 = var_5.__sub__(var_3)
        var_20 = var_8.__sub__(var_6)
        var_21 = var_9.__sub__(var_7)
        var_66 = var_20.__truediv__(var_21)
        var_67 = var_14.__truediv__(var_15)
        var_68 = paddle.tensor.ops.atan(var_66)
        var_69 = paddle.tensor.ops.atan(var_67)
        var_70 = var_68.__sub__(var_69)
        var_71 = var_70.__rmul__(0.4052847345693511)
        var_72 = var_71.__mul__(var_70)
        # loss = mean(center_dist_ratio + v)
        var_result = var_65.__add__(var_72)
        var_loss = paddle.tensor.stat.mean(var_result)
        return var_loss


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
