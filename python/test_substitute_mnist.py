import tensorflow as tf
from substitute import FuncSubstitute, get_approx_kernel

from common import (
    load_data,
    test_comparison,
    OUT_DIR,
)

# Configuration for our MNIST model
NUM_CLASSES = 10
NUM_ROWS, NUM_COLS = 28, 28
BATCH_SIZE = 32
INPUT_SHAPE = [1, NUM_ROWS, NUM_COLS, 1]  # Static shape with batch size of 1
OUTPUT_SHAPE = [NUM_CLASSES]  # Static shape for output (batch size of 1)
FEATURES_SHAPE = [NUM_ROWS, NUM_COLS, 1]  # Single image shape (without batch)


def test():
    # Load MNIST data
    exact_module_path = OUT_DIR + "mnist_exact_model"
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
        batch_size=BATCH_SIZE,
    )

    func_sub.compile_exact()

    func_sub.test_comparison = test_comparison.__get__(
        func_sub
    )  # Bind the method to the instance

    # Training parameters
    num_samples = 5000  # Number of samples to use for training the approximation
    epochs = 50  # Number of training epochs

    # Train the approximate kernel
    print(
        f"Training approximation kernel with up to {num_samples} samples for {epochs} epochs..."
    )
    print(f"There are {len(x_train)} samples in total.")
    func_sub.train_approx(
        use_provided=True, user_data=x_train[:num_samples], epochs=epochs
    )
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
    test()
