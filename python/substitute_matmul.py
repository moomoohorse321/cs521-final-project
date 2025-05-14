# import iree.compiler.tf
# import iree.runtime
# import tensorflow as tf
# import numpy as np
# from matplotlib import pyplot as plt
# from iree.tf.support import module_utils
# from iree import runtime as ireert
# from iree.compiler import compile_str
# from substitute import FuncSubstitute, get_approx_kernel

import numpy as np

A = 4
B = 3
C = 3

# create a 4 x 3 tensor manually
SAMPLE_LHS = np.array([[1, 2, 3], [1, 1, 1], [0,2,0], [2, 0, 1]], dtype=np.float32)
SAMPLE_RHS = np.array([[1, 0, 0], [0, 1, 0], [2, 3, 0]], dtype=np.float32)


def create_matmul_module():
    class MatMulModule(tf.Module):
        def __init__(self):
            super(MatMulModule, self).__init__()

        @tf.function(input_signature=[tf.TensorSpec(shape=(A, B), dtype=tf.float32),
                                        tf.TensorSpec(shape=(B, C), dtype=tf.float32)])
        def basic_matmul(self, lhs, rhs):
            res = np.zeros((A, C), dtype=np.float32)
            for i in range(A):
                for j in range(C):
                    for k in range(B):
                        res[i][j] += lhs[i][k] * rhs[k][j]
            return res

    return MatMulModule()


def test_exact_matmul():
    module = create_matmul_module()
    compiled_module = iree.compiler.tf.compile_module(module.basic_matmul, target_backends=["dylib"])
    compiled_module = module_utils.load_module(compiled_module)
    compiled_module.basic_matmul(SAMPLE_LHS, SAMPLE_RHS)


# TODO remove
def basic_matmul(lhs, rhs):
    res = np.zeros((A, C), dtype=np.float32)
    for i in range(A):
        for j in range(C):
            for k in range(B):
                res[i][j] += lhs[i][k] * rhs[k][j]
    return res

if __name__ == "__main__":
    # test_exact_matmul()

    # test basic matmul
    res = basic_matmul(SAMPLE_LHS, SAMPLE_RHS)
    predicted_res = np.matmul(SAMPLE_LHS, SAMPLE_RHS)
    assert np.allclose(res, predicted_res), "Basic matmul failed"
    print("Basic matmul passed")