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

            for i in range(A):
                for j in range(C):
                    for k in range(B):
                        res += lhs[i, k] * rhs[k, j]
                        if i == 0 and j == 0:
                            print(f"lhs[{i}, {k}] * rhs[{k}, {j}] = {lhs[i, k]} * {rhs[k, j]}")
            return res

    return MatMulModule()


def compile_exact_matmul():
    module = create_matmul_module()

    temp_save_path = "matmul_module"
    mlir_path = "mlirbs_temp/matmul_module.mlirbc"


    print("Saving module to SavedModel...")
    tf.saved_model.save(module, temp_save_path)

    # TODO: without saving and loading like this, get the error
    # AttributeError: '_SignatureMap' object has no attribute 'name'
    mlir_bc = iree.compiler.tf.compile_saved_model(
        temp_save_path,
        target_backends=["llvm-cpu"],
        input_type="savedmodel",
        import_only=True
    )
    print("Compiled og matmul module.")
    shutil.rmtree(temp_save_path)

    with open(mlir_path, "wb") as f:
        f.write(mlir_bc)
    print(f"Wrote MLIR bytecode to {mlir_path}")

    
    # return mlir_bc
    return mlir_path



# def load_and_run_matmul(mlir_bc):
#     # flatbuffer_blob = compile_str(mlir_bc, target_backends=["llvm-cpu"], input_type="stablehlo")
#     # print("Flatbuffer blob:")
#     # print(flatbuffer_blob)
#     print(mlir_bc)

#     config = ireert.Config("local-task")
#     ctx = ireert.SystemContext(config=config)
#     vm_module = ireert.VmModule.from_flatbuffer(ireert.VmInstance(config), mlir_bc)
#     ctx.add_vm_module(vm_module)

    
#     result = ctx.modules.module.basic_matmul(SAMPLE_LHS, SAMPLE_RHS)

#     result_np = result.to_host()
#     return result_np

def load_and_run_matmul_from_file(mlir_path):
    # Read bytecode from file
    with open(mlir_path, "rb") as f:
        bytecode = f.read()
    
    # Create VM instance and context
    vm_instance = ireert.VmInstance()
    config = ireert.Config("local-task")
    context = ireert.SystemContext(config=config)
    
    # Load the compiled module from the bytecode
    vm_module = ireert.VmModule.from_flatbuffer(vm_instance, bytecode)
    context.add_vm_module(vm_module)
    
    # Call the function directly through the module namespace
    result = context.modules.module.basic_matmul(SAMPLE_LHS, SAMPLE_RHS)
    
    # Convert result back to numpy (if needed)
    if hasattr(result, 'to_host'):
        result_np = result.to_host()
    else:
        result_np = result  # It might already be a numpy array
    
    return result_np


if __name__ == "__main__":
    print("-------------------------------------------------")
    mlir_path = compile_exact_matmul()
    # mlir_bc = compile_exact_matmul()

    # result = load_and_run_matmul(mlir_bc)
    result = load_and_run_matmul_from_file(mlir_path)
    print("Result of matmul:")
    print(result)

    expected = tf.matmul(SAMPLE_LHS, SAMPLE_RHS).numpy()
    print("Expected result:")
    print(expected)
    print("Difference:")
    print(np.abs(result - expected).max())