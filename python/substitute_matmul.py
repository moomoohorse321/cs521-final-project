import os
import iree.compiler.tf
import tensorflow as tf
from iree.tf.support import module_utils
from iree import runtime as ireert
from iree.compiler import compile_str

import tensorflow as tf
import numpy as np

A = 4
B = 3
C = 3

# create a 4 x 3 tensor manually
SAMPLE_LHS = np.array([[1, 2, 3], [1, 1, 1], [0,2,0], [2, 0, 1]], dtype=np.float32)
SAMPLE_RHS = np.array([[1, 0, 0], [0, 1, 0], [2, 3, 0]], dtype=np.float32)


SAVE_DIR = "mlirbs_temp/"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
os.environ["IREE_SAVE_TEMPS"] = "/tmp/iree_debug"


def load_mlir_from_file(mlir_path):
    backend_choice = "iree_llvmcpu (CPU)"  # @param [ "iree_vmvx (CPU)", "iree_llvmcpu (CPU)", "iree_vulkan (GPU/SwiftShader)" ]
    backend_choice = backend_choice.split(" ")[0]
    backend = module_utils.BackendInfo(backend_choice)
    print("Backend choice:", backend_choice)

    with open(mlir_path, "r") as f:
        mlir_module = f.read()

    flatbuffer_blob = compile_str(
        mlir_module, target_backends=["llvm-cpu"], input_type="stablehlo",
        # extra_args=["--mlir-print-ir-after-all", "--iree-flow-trace-dispatch-tensors"]
    )

    config = ireert.Config(backend.driver)
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, flatbuffer_blob)
    ctx.add_vm_module(vm_module)

    return ctx.modules.module

def create_matmul_module():
    class MatMulModule(tf.Module):
        def __init__(self):
            # TODO take ABC as inputs instead of constants
            super(MatMulModule, self).__init__()

        @tf.function(input_signature=[tf.TensorSpec(shape=(A, B), dtype=tf.float32),
                                        tf.TensorSpec(shape=(B, C), dtype=tf.float32)])
        def basic_matmul(self, lhs, rhs):
            res = tf.zeros((1, C), dtype=tf.float32)

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
    
    
    module = create_matmul_module()

    func_path = SAVE_DIR + "matmul_module"
    tf.saved_model.save(module, func_path)
    print(f"Saved exact matmul module to {func_path}")


    # Compiling to IREE
    mlir_bc = iree.compiler.tf.compile_module(
        module,
        target_backends=["iree_llvmcpu"],
        exported_names=["basic_matmul"],
        import_only=True,
    )

    mlirbc_path = SAVE_DIR + "exact.mlirbc"
    mlir_path = SAVE_DIR + "exact.mlir"
    with open(mlirbc_path, "wb") as f:
        f.write(mlir_bc)
    print(f"Wrote MLIR bytecode to {mlirbc_path}")

    os.system(f"iree-ir-tool copy {mlirbc_path} -o {mlir_path}")


    loaded_module = load_mlir_from_file(mlir_path)
    print("Loaded MLIR module from file.")


    # exact_matmul_module = tf.saved_model.load(func_path)
    # print("Loaded exact matmul module from saved file.")

    # # Test the exact module
    # print("Testing approx matmul module...")
    # approx_res = exact_matmul_module.approx_matmul(SAMPLE_LHS, SAMPLE_RHS, 2)
    # print("Exact matmul result:")
    # print(approx_res)

    # expected_result = np.matmul(SAMPLE_LHS, SAMPLE_RHS)
    # print("Expected result:")
    # print(expected_result)

    # # Find mean square error
    # mse = np.mean((approx_res - expected_result) ** 2)
    # print("Mean square error:")
    # print(mse)
    print("-------------------------------------------------")

