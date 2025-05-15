import iree.compiler.tf
import iree.runtime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from iree.tf.support import module_utils
from iree import runtime as ireert
from iree.compiler import compile_str
from substitute import FuncSubstitute, get_approx_kernel
import os

from common import load_data, test_load, create_mnist_module

# Configuration for our MNIST model
NUM_CLASSES = 10
NUM_ROWS, NUM_COLS = 28, 28
BATCH_SIZE = 32
INPUT_SHAPE = [1, NUM_ROWS, NUM_COLS, 1]  # Static shape with batch size of 1
OUTPUT_SHAPE = [NUM_CLASSES]  # Static shape for output (batch size of 1)
FEATURES_SHAPE = [NUM_ROWS, NUM_COLS, 1]  # Single image shape (without batch)



def train_exact_module(model, data, epochs=5):
    """Train the trainable model on MNIST data with real-time line updates."""
    (x_train, y_train, y_train_onehot) = data
    
    # Set up training loop
    steps_per_epoch = len(x_train) // BATCH_SIZE
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            # Get batch
            batch_start = step * BATCH_SIZE
            batch_end = batch_start + BATCH_SIZE
            x_batch = x_train[batch_start:batch_end]
            y_batch = y_train_onehot[batch_start:batch_end] 
            
            # Perform one training step
            step_loss = model.learn(x_batch, y_batch)
            epoch_loss += step_loss
            
            # Update progress line (overwrites previous line)
            if step % 10 == 0:
                print(f"\rEpoch {epoch+1}/{epochs}, Step {step}/{steps_per_epoch}", end="")
        
        # Print epoch summary (overwrites previous line)
        print(f"\rEpoch {epoch+1}/{epochs} complete, Average Loss: {epoch_loss / steps_per_epoch}", end="")
    print("\nTraining complete.")
    
    return model

def test_comparison(self, test_images, test_labels, num_samples=10, use_mlir_approx=True):
    """
    Compare the exact and approximate models on test images.
    
    Args:
        test_images: Test images
        test_labels: Test labels
        num_samples: Number of samples to test
    """
    # Select random samples for testing
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    
    correct_exact = 0
    correct_approx = 0
    avg_kl_div = 0.0
    
    plt.figure(figsize=(15, num_samples * 2))
    
    for i, idx in enumerate(indices):
        # Get test image and label
        test_image = test_images[idx]
        true_label = test_labels[idx]
        
        # Get exact prediction
        exact_result = self.exact_module.predict(test_image)
        exact_pred = np.argmax(exact_result.numpy())
        
        # Get approximate prediction
        if use_mlir_approx:
            approx_result = self.approx_kernel.approx_predict(test_image).to_host()
        else:
            approx_result = self.approx_kernel.approx_predict(test_image).numpy()
        approx_pred = np.argmax(approx_result)
        
        # Update counters
        if exact_pred == true_label:
            correct_exact += 1
        if approx_pred == true_label:
            correct_approx += 1
        
        # Calculate KL divergence
        kl_div = tf.keras.losses.KLDivergence()(exact_result, approx_result)
        avg_kl_div += kl_div.numpy()
        
        # Display image and predictions
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(test_image.reshape(NUM_ROWS, NUM_COLS), cmap='gray')
        plt.title(f"True: {true_label}")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.bar(range(NUM_CLASSES), exact_result.numpy())
        plt.title(f"Exact: {exact_pred}" + (" ✓" if exact_pred == true_label else " ✗"))
        
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.bar(range(NUM_CLASSES), approx_result)
        plt.title(f"Approx: {approx_pred}" + (" ✓" if approx_pred == true_label else " ✗"))
    
    plt.tight_layout()
    plt.savefig("figure1.png")
    plt.show()
    
    # Print results
    print(f"\nExact Model Accuracy: {correct_exact / num_samples:.2f}")
    print(f"Approximate Model Accuracy: {correct_approx / num_samples:.2f}")
    print(f"Average KL Divergence: {avg_kl_div / num_samples:.6f}")
    
    # Test on full test set
    exact_correct = 0
    approx_correct = 0
    
    for i in range(len(test_images)):
        # Get test image and label
        test_image = test_images[i]
        true_label = test_labels[i]
        
        # Get exact prediction
        exact_result = self.exact_module.predict(test_image)
        exact_pred = np.argmax(exact_result.numpy())
        
        # Get approximate prediction
        if use_mlir_approx:
            approx_result = self.approx_kernel.approx_predict(test_image).to_host()
        else:
            approx_result = self.approx_kernel.approx_predict(test_image).numpy()
        approx_pred = np.argmax(approx_result)
        
        # Update counters
        if exact_pred == true_label:
            exact_correct += 1
        if approx_pred == true_label:
            approx_correct += 1
            
    print(f"\nFull Test Set - Exact Model Accuracy: {exact_correct / len(test_images):.4f}")
    print(f"Full Test Set - Approximate Model Accuracy: {approx_correct / len(test_images):.4f}")

def test():
    # Load MNIST data
    exact_module_path = "mnist_exact_model"
    (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot) = load_data()
    
    # # Create and train the exact MNIST module
    # exact_module = create_mnist_module(BATCH_SIZE)
    # print("Training the exact model...")
    # exact_module = train_exact_module(exact_module, (x_train, y_train, y_train_onehot), epochs=5)
    # print("Exact model training complete.")
    
    # # # save it to .pth
    # tf.saved_model.save(exact_module, exact_module_path)
    # print(f"Exact model saved to {exact_module_path}")
    
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
    
    func_sub.compile_exact()
    
    func_sub.test_comparison = test_comparison.__get__(func_sub)  # Bind the method to the instance
    
    # Training parameters
    num_samples = 5000  # Number of samples to use for training the approximation
    epochs = 50         # Number of training epochs
    
    # Train the approximate kernel
    print(f"Training approximation kernel with up to {num_samples} samples for {epochs} epochs...")
    print(f"There are {len(x_train)} samples in total.")
    func_sub.train_approx(use_provided=True, user_data=x_train[:num_samples], epochs=epochs)
    print("Approximation training complete.")
    
    # Compile the approximate kernel to MLIR
    mlir_path = func_sub.compile_approx()
    
    # Test on sample images
    print("\nComparing exact and approximate models on test samples...")
    func_sub.test_comparison(x_test, y_test, num_samples=10, use_mlir_approx=False)
    
    print(f"\nApproximate function substitution test complete.")
    print(f"MLIR bytecode saved to: {mlir_path}")
    
    return exact_module, func_sub

    

if __name__ == "__main__":
    test_load()