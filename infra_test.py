import tensorflow as tf
import numpy as np
import os # For environment variable to suppress TF oneDNN logs if desired

# Optional: Suppress TensorFlow oneDNN informational messages
# These messages (like "oneDNN custom operations are on") are not errors.
# Set this before importing TensorFlow if you want to suppress them for cleaner output.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# For more verbose TF logging (e.g., device placement), you can use:
# tf.debugging.set_log_device_placement(True)


# Import Sionna after TensorFlow and basic checks
try:
    import sionna as sn
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    print("Warning: Sionna library not found or could not be imported.")


print("--- TensorFlow and Sionna Version Check ---")
print(f"TensorFlow version: {tf.__version__}")

if SIONNA_AVAILABLE:
    try:
        print(f"Sionna version: {sn.__version__}")
    except AttributeError:
        # Fallback for older Sionna or unusual installations if __version__ is missing
        try:
            import pkg_resources
            print(f"Sionna version (from pkg_resources): {pkg_resources.get_distribution('sionna').version}")
        except Exception:
            print("Sionna version: Could not be determined (consider checking pip list).")
else:
    print("Sionna: Not available.")


print("\n--- TensorFlow CUDA and GPU Availability Basic Check ---")
print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
# is_gpu_available() also checks if a GPU can be successfully initialized.
# cuda_only=True ensures it's an NVIDIA GPU.
is_tf_gpu_available = tf.test.is_gpu_available(cuda_only=True)
print(f"TensorFlow GPU available (and initializable): {is_tf_gpu_available}")

print("\n--- Detailed GPU Device Listing and Test ---")
# List physical GPUs first. This doesn't initialize them.
physical_gpus = tf.config.list_physical_devices('GPU')

if physical_gpus:
    print(f"Found {len(physical_gpus)} Physical GPU(s):")
    for i, gpu in enumerate(physical_gpus):
        print(f"  Physical GPU {i}: Name - {gpu.name}, Type - {gpu.device_type}")
    
    print("\nAttempting to configure Physical GPUs for TensorFlow...")
    try:
        # It's crucial to set memory growth before any significant GPU operations
        # to prevent TensorFlow from allocating all memory on the first GPU it uses.
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # After setting memory growth, list logical GPUs that TensorFlow will actually use.
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Successfully configured {len(logical_gpus)} Logical GPU(s) with memory growth.")

        if logical_gpus:
            print("\n--- Performing Test Operation on Each Configured Logical GPU ---")
            all_gpu_tests_passed = True
            for i, logical_gpu in enumerate(logical_gpus):
                print(f"Testing on Logical GPU {i}: {logical_gpu.name}...")
                try:
                    with tf.device(logical_gpu.name): # Explicitly place operations on this GPU
                        # Create some tensors on the GPU
                        a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
                        b = tf.constant([[1.0, 1.0], [0.0, 1.0]], dtype=tf.float32)
                        # Perform a computation
                        c = tf.matmul(a, b)
                        # Ensure computation happened by accessing the result
                        result_numpy = c.numpy() 
                        print(f"  Matrix multiplication on {logical_gpu.name} result:\n  {result_numpy}")
                        
                        # Check if the result tensor is indeed on the specified GPU
                        if "GPU" in c.device.upper():
                             print(f"  Result tensor c is on device: {c.device} (Correctly on GPU)")
                             print(f"  Test on {logical_gpu.name} PASSED.")
                        else:
                             print(f"  Warning: Result tensor c is on device: {c.device} (Expected GPU, got CPU?)")
                             print(f"  Test on {logical_gpu.name} partially PASSED (computation done, but device might be unexpected).")
                             all_gpu_tests_passed = False # Or treat as a failure depending on strictness

                except RuntimeError as e:
                    print(f"  RuntimeError during test on {logical_gpu.name}: {e}")
                    print(f"  Test on {logical_gpu.name} FAILED.")
                    all_gpu_tests_passed = False
                except Exception as e_gen:
                    print(f"  An unexpected error occurred during test on {logical_gpu.name}: {e_gen}")
                    print(f"  Test on {logical_gpu.name} FAILED.")
                    all_gpu_tests_passed = False
            
            if all_gpu_tests_passed and logical_gpus:
                print("\nAll GPU operation tests PASSED.")
            elif logical_gpus: # Some tests might have failed or had warnings
                print("\nSome GPU operation tests had issues or FAILED. Please review logs.")
            else: # Should not happen if physical_gpus is true and set_memory_growth worked
                print("No logical GPUs were available for testing after configuration.")
        else: # No logical_gpus after physical_gpus detected
            print("Physical GPUs were detected, but no Logical GPUs could be configured by TensorFlow.")
            print("This might indicate an issue with CUDA drivers, toolkit, or TensorFlow's GPU support.")

    except RuntimeError as e:
        # This usually means memory growth was set too late, or another fundamental GPU setup issue.
        print(f"RuntimeError during GPU configuration (e.g., setting memory growth): {e}")
        print("TensorFlow may not be able to use the GPUs correctly.")
    except Exception as e_setup:
        print(f"An unexpected error occurred during GPU setup: {e_setup}")

else: # No physical_gpus
    print("No Physical GPU(s) detected by TensorFlow.")

print("\n--- Setup Test Script Finished ---")