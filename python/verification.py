import iree.compiler.tf
import iree.runtime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from iree.tf.support import module_utils
from iree import runtime as ireert
from iree.compiler import compile_str
from substitute import FuncSubstitute, get_approx_kernel
from approxMLIR import ToolBox
import os

from common import load_data, test_load, test_comparison, train_exact_module, proj_dir, OUT_DIR

# Configuration for our MNIST model
NUM_CLASSES = 10
NUM_ROWS, NUM_COLS = 28, 28
BATCH_SIZE = 32
INPUT_SHAPE = [1, NUM_ROWS, NUM_COLS, 1]  # Static shape with batch size of 1
OUTPUT_SHAPE = [NUM_CLASSES]  # Static shape for output (batch size of 1)
FEATURES_SHAPE = [NUM_ROWS, NUM_COLS, 1]  # Single image shape (without batch)

IMG_DIR = "../imgs/verification/"

# Create a CNN model for MNIST digit recognition (this is our "exact" model)
def create_mnist_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=FEATURES_SHAPE),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Wrap the model in a tf.Module to compile it with IREE
def create_mnist_module(batch_size=BATCH_SIZE):
    class MNISTModule(tf.Module):
        def __init__(self):
            super(MNISTModule, self).__init__()
            self.model = create_mnist_model()
            
            # Compile the model
            self.model.compile(
                optimizer='adam',
                loss=tf.keras.losses.KLDivergence(),
                metrics=['accuracy']
            )

        @tf.function(input_signature=[tf.TensorSpec(FEATURES_SHAPE, tf.float32)])
        def predict(self, x):
            """Exact prediction function for MNIST. (non-batched)"""
            batched_x = tf.expand_dims(x, 0)  # Add batch dimension
            batched_res = self.model(batched_x, training=False)
            return tf.squeeze(batched_res, 0)  # Remove batch dimension for output
        
        @tf.function(input_signature=[
            tf.TensorSpec([batch_size] + FEATURES_SHAPE, tf.float32),
            tf.TensorSpec([batch_size, NUM_CLASSES], tf.float32)  # One-hot encoded labels
        ])
        def learn(self, x, y):
            """Train the model on batched data."""
            with tf.GradientTape() as tape:
                predictions = self.model(x, training=True)
                loss = tf.keras.losses.KLDivergence()(y, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            return loss
        
    return MNISTModule()


def init_mlir_files():
    # Load MNIST data
    exact_module_path = OUT_DIR + "mnist_exact_model"
    (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot) = load_data()
    
    # Create and train the exact MNIST module
    exact_module = create_mnist_module(BATCH_SIZE)
    print("Training the exact model...")
    exact_module = train_exact_module(exact_module, (x_train, y_train, y_train_onehot), epochs=5) 
    print("Exact model training complete.")
    
    # # save it to .pth
    tf.saved_model.save(exact_module, exact_module_path)
    print(f"Exact model saved to {exact_module_path}")
    
    # load it back
    exact_module = tf.saved_model.load(exact_module_path)
    print("Exact model loaded from saved file.")
    
    # Test constructing an approximate kernel
    print("Creating approximation kernel...")
    test_kernel = get_approx_kernel(FEATURES_SHAPE, OUTPUT_SHAPE, BATCH_SIZE)
    print("Successfully created approximation kernel")
    
    # Create the function substitution handler
    func_sub = FuncSubstitute(
        exact_module=exact_module,
        approx_kernel=test_kernel,
        input_shape=FEATURES_SHAPE,
        batch_size=BATCH_SIZE
    )
    
    mlir_path1 = func_sub.compile_exact()
    
    func_sub.test_comparison = test_comparison.__get__(func_sub)  # Bind the method to the instance
    
    # Training parameters
    num_samples = 5000  # Number of samples to use for training the approximation
    epochs = 50        # Number of training epochs 
    
    # Train the approximate kernel
    print(f"Training approximation kernel with up to {num_samples} samples for {epochs} epochs...")
    print(f"There are {len(x_train)} samples in total.")
    func_sub.train_approx(use_provided=True, user_data=x_train[:num_samples], epochs=epochs)
    print("Approximation training complete.")
    
    # Compile the approximate kernel to MLIR
    mlir_path2 = func_sub.compile_approx()
    
    replace_exec_path = "../bin/replace"
    merge_exec_path = "../bin/merge"
    opt_exec_path = "../bin/approxMLIR-opt"

    mlir_path1 = OUT_DIR + "approx.mlir"
    mlir_path2 = OUT_DIR + "exact.mlir"
    output_path = OUT_DIR + "merged.mlir"
    auziliary_path = OUT_DIR + "auxiliary.mlir"
    ext_path = OUT_DIR + "ext.mlir"

    # mlir path is *.mlirbc
    os.system(f"iree-ir-tool copy {mlir_path1}bc -o {mlir_path1}")
    os.system(f"iree-ir-tool copy {mlir_path2}bc -o {mlir_path2}")
    toolbox = ToolBox(replace_exec_path, merge_exec_path, opt_exec_path)
    toolbox.write2file_auxiliary_mlir_str(auziliary_path)
    toolbox.link_mlir_modules(auziliary_path, mlir_path1, ext_path, keep_temp_files=True)
    toolbox.link_mlir_modules(ext_path, mlir_path2, output_path, keep_temp_files=True)
    toolbox.optimize_mlir(output_path, "output.mlir")
    
    os.system(f"mv output.mlir {OUT_DIR + 'output.mlir'}")
    print("Successfully merged and optimized MLIR files to ", {OUT_DIR + 'output.mlir'})
    
    

if __name__ == "__main__":
    print("-------------------------------------------------")
    # Declare and create the directory for saving images
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    init_mlir_files()
    test_load(img_dir=IMG_DIR)
    print("-------------------------------------------------")