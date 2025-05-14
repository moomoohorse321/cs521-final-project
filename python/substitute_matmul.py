import iree.compiler.tf
import tensorflow as tf
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
            res = tf.zeros((A, C), dtype=tf.float32)

            for i in range(A):
                for j in range(C):
                    for k in range(B):
                        res += lhs[i, k] * rhs[k, j]
            return res

    return MatMulModule()


def test_exact_matmul():
    module = create_matmul_module()
    save_path = "mlirbs_temp/matmul_module.mlirbc"

    print("Saving module to SavedModel...")
    tf.saved_model.save(module, save_path)

    # TODO: without saving and loading like this, get the error
    # AttributeError: '_SignatureMap' object has no attribute 'name'
    print("Compiling og matmul module...")
    mlir_bc = iree.compiler.tf.compile_saved_model(
        save_path,
        target_backends=["llvm-cpu"],
        input_type="savedmodel",
        import_only=True
    )
    print("Compiling og matmul module done.")
    print(mlir_bc)



if __name__ == "__main__":
    print("-------------------------------------------------")
    test_exact_matmul()