import os
import shutil
import iree.compiler.tf
from iree.tf.support import module_utils
import iree.runtime as ireert
from iree.compiler import compile_str

import tensorflow as tf
import numpy as np

A = 4
B = 3
C = 3

# create a 4 x 3 tensor manually
SAMPLE_LHS = np.array([[1, 2, 3], [1, 1, 1], [0,2,0], [2, 0, 1]], dtype=np.float32)
SAMPLE_RHS = np.array([[1, 0, 0], [0, 1, 0], [2, 3, 0]], dtype=np.float32)


SAVE_DIR = "mlirbs_temp"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def create_matmul_module():
    class MatMulModule(tf.Module):
        def __init__(self):
            super(MatMulModule, self).__init__()

        @tf.function(input_signature=[tf.TensorSpec(shape=(A, B), dtype=tf.float32),
                                        tf.TensorSpec(shape=(B, C), dtype=tf.float32)])
        def basic_matmul(self, lhs, rhs):
            res = tf.zeros((A, C), dtype=tf.float32)

            # using tensorflow automatic diff compatible operations
            for i in range(A):
                for j in range(C):
                    for k in range(B):
                        indices = tf.constant([[i, j]])
                        updates = tf.expand_dims(lhs[i, k] * rhs[k, j], 0)
                        res = tf.tensor_scatter_nd_add(res, indices, updates)
                        
                        if i == 0 and j == 0:
                            tf.print(f"lhs[{i}, {k}] * rhs[{k}, {j}] = {lhs[i, k]} * {rhs[k, j]}")
            return res
        
        @tf.function(input_signature=[tf.TensorSpec(shape=(A, B), dtype=tf.float32),
                                        tf.TensorSpec(shape=(B, C), dtype=tf.float32), 
                                        tf.TensorSpec(shape=(), dtype=tf.int32)]) 
        def approx_matmul(self, lhs, rhs, stride=1):
            res = tf.zeros((A, C), dtype=tf.float32)

            # using tensorflow automatic diff compatible operations
            for i in range(A):
                for j in range(C):
                    for k in range(0, B, stride):
                        indices = tf.constant([[i, j]])
                        updates = tf.expand_dims(lhs[i, k] * rhs[k, j], 0)
                        res = tf.tensor_scatter_nd_add(res, indices, updates)
                        
                        if i == 0 and j == 0:
                            tf.print(f"lhs[{i}, {k}] * rhs[{k}, {j}] = {lhs[i, k]} * {rhs[k, j]}")
            return res

    return MatMulModule()

if __name__ == "__main__":
    print("-------------------------------------------------")
    
    exact_matmul_module = create_matmul_module()

    func_path = "mlirbs_temp/matmul_module"
    tf.saved_model.save(exact_matmul_module, func_path)
    print(f"Saved exact matmul module to {func_path}")

    exact_matmul_module = tf.saved_model.load(func_path)
    print("Loaded exact matmul module from saved file.")

    # Test the exact module
    print("Testing approx matmul module...")
    approx_res = exact_matmul_module.approx_matmul(SAMPLE_LHS, SAMPLE_RHS, 2)
    print("Exact matmul result:")
    print(approx_res)

    expected_result = np.matmul(SAMPLE_LHS, SAMPLE_RHS)
    print("Expected result:")
    print(expected_result)

    # Find mean square error
    mse = np.mean((approx_res - expected_result) ** 2)
    print("Mean square error:")
    print(mse)
    print("-------------------------------------------------")