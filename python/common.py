import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from substitute import FuncSubstitute, get_approx_kernel

# TODO sort correctly to not be copied w common and other scripts# Configuration for our MNIST model
NUM_CLASSES = 10
NUM_ROWS, NUM_COLS = 28, 28
BATCH_SIZE = 32
INPUT_SHAPE = [1, NUM_ROWS, NUM_COLS, 1]  # Static shape with batch size of 1
OUTPUT_SHAPE = [NUM_CLASSES]  # Static shape for output (batch size of 1)
FEATURES_SHAPE = [NUM_ROWS, NUM_COLS, 1]  # Single image shape (without batch)


parent_dir_path = os.path.dirname(os.path.abspath(__file__))


def load_data():
    """Load MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape into grayscale images
    x_train = np.reshape(x_train, (-1, NUM_ROWS, NUM_COLS, 1))
    x_test = np.reshape(x_test, (-1, NUM_ROWS, NUM_COLS, 1))

    # Rescale pixel values to [0, 1]
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    # Convert labels to one-hot encoding for the classifier
    y_train_onehot = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot)


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


def test_load(load_mlir_path = os.path.join(parent_dir_path, "../bin", "output.mlir")):
    (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot) = load_data()
    
    exact_module_path = "mnist_exact_model"
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
        approx_kernel=FuncSubstitute.load_mlir_from_file(load_mlir_path),
        input_shape=FEATURES_SHAPE,
        batch_size=BATCH_SIZE
    )
    
    func_sub.test_comparison = test_comparison.__get__(func_sub)
    
    func_sub.test_comparison(
        x_test, y_test, num_samples=10
    )