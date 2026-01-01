import os
import sys
import subprocess
import pkgutil
import importlib

def test_jax_cuda_loading():
    print("=== JAX CUDA Loading Test ===")
    
    # Check if jax is installed
    try:
        import jax
        print("✓ JAX imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import JAX: {e}")
        return
        
    # Check if jaxlib is installed
    try:
        import jaxlib
        print("✓ jaxlib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import jaxlib: {e}")
        return
    
    # Check CUDA path environment variables
    cuda_path = os.environ.get('CUDA_PATH')
    cuda_home = os.environ.get('CUDA_HOME')
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    
    print("\n=== Environment Variables ===")
    print(f"CUDA_PATH: {cuda_path if cuda_path else 'not set'}")
    print(f"CUDA_HOME: {cuda_home if cuda_home else 'not set'}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # Try to get available devices
    print("\n=== Available Devices ===")
    try:
        devices = jax.devices()
        print(f"Available devices: {devices}")
        
        # Check if GPU is available
        gpu_devices = [d for d in devices if d.device_kind == 'gpu']
        if gpu_devices:
            print(f"✓ Found {len(gpu_devices)} GPU devices:")
            for i, device in enumerate(gpu_devices):
                print(f"  GPU {i}: {device}")
        else:
            print("✗ No GPU devices found. This might indicate CUDA drivers are not properly loaded.")
    except Exception as e:
        print(f"✗ Error getting devices: {e}")
    
    # Test CUDA initialization explicitly
    print("\n=== CUDA Plugin Test ===")
    try:
        from jax_plugins import xla_cuda12
        print("✓ CUDA plugin module found")
    except ImportError as e:
        print(f"✗ CUDA plugin import failed: {e}")
        print("  This might be normal if you're not using CUDA 12.x")
    
    # Try a simple computation to verify GPU works
    print("\n=== GPU Computation Test ===")
    try:
        import numpy as np
        # Create a small array and move it to GPU
        x = jax.numpy.array([1.0, 2.0, 3.0])
        y = jax.numpy.array([4.0, 5.0, 6.0])
        
        # Force computation on GPU if available
        result = jax.jit(lambda a, b: a + b)(x, y)
        print(f"✓ Simple GPU computation succeeded: {result}")
        
        # Check where the computation ran
        device = result.device()
        print(f"  Computation ran on device: {device}")
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")

def check_system_cuda():
    print("\n=== System CUDA Check ===")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ NVIDIA drivers detected:")
            print(result.stdout.split('\n')[0])  # Just show first line
        else:
            print("✗ nvidia-smi failed. NVIDIA drivers may not be installed properly.")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"✗ Could not run nvidia-smi: {e}")
        print("  Make sure NVIDIA drivers are installed and nvidia-smi is in PATH")
    
    # Check for libcuda.so (Linux) or nvcuda.dll (Windows)
    print("\n=== CUDA Library Check ===")
    cuda_libs = []
    
    if sys.platform.startswith('linux'):
        cuda_libs = ['libcuda.so', 'libcuda.so.1']
    elif sys.platform.startswith('win'):
        cuda_libs = ['nvcuda.dll']
    
    found_lib = False
    for lib in cuda_libs:
        try:
            from ctypes import CDLL
            CDLL(lib)
            print(f"✓ Found CUDA library: {lib}")
            found_lib = True
            break
        except OSError:
            continue
    
    if not found_lib:
        print("✗ No CUDA libraries found in system path")
        print("  This suggests CUDA drivers are not properly installed or accessible")

if __name__ == "__main__":
    print("Starting JAX CUDA test...")
    test_jax_cuda_loading()
    check_system_cuda()
    print("\n=== Test Complete ===")