import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Set device to GPU if available (CUDA or Apple Silicon MPS)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}\n")

# Set parameters for the signal
N = 2048  # Number of sample points
T = 1.0  # Duration of the signal in seconds
f0 = 1  # Fundamental frequency of the square wave in Hz
harmonics = [1, 3, 5]


# 1. Modified square_wave using PyTorch
def square_wave(t, f0):
    return torch.sign(torch.sin(2.0 * torch.pi * f0 * t))


# 2. Modified square_wave_fourier using PyTorch
def square_wave_fourier(t, f0, N_harmonics):
    result = torch.zeros_like(t)
    for k in range(N_harmonics):
        n = 2 * k + 1
        result += torch.sin(2 * torch.pi * n * f0 * t) / n
    return (4 / torch.pi) * result


# 3. Modified naive_dft using PyTorch (Loop-based, CPU)
def naive_dft(x):
    N = len(x)
    X = torch.zeros(N, dtype=torch.complex128)
    for k in range(N):
        for n in range(N):
            angle = -2j * torch.pi * k * n / N
            X[k] += x[n] * torch.exp(torch.tensor(angle))
    return X


# 4. NEW: Vectorized GPU DFT using Tensor Operations
def naive_dft_gpu(x):
    N = len(x)
    # Move signal to target device and cast to complex128 (ComplexDouble)
    x_device = x.to(device, dtype=torch.complex128)

    # Create n and k indices for broadcasting
    # Cast to float64 so the complex math promotes properly to complex128
    n = torch.arange(N, device=device, dtype=torch.float64)
    k = n.view(-1, 1)

    # Construct the DFT transformation matrix W
    W = torch.exp(-2j * torch.pi * k * n / N)

    # Force W to complex128 to strictly match x_device
    W = W.to(torch.complex128)

    # Compute the DFT via matrix multiplication: W * x
    X = torch.matmul(W, x_device)
    return X


# Create the time vector in PyTorch (equivalent to endpoint=False)
t = torch.arange(N) * (T / N)

# Construct a square wave using 50 harmonics
signal = square_wave_fourier(t, f0, 50)

# --- Timing the 3 Methods ---

# Method 1: PyTorch Built-in FFT
start_time_fft = time.time()
fft_result = torch.fft.fft(signal)
fft_duration = time.time() - start_time_fft

# Method 2: Vectorized GPU DFT
# (Run once to warm up the GPU/cache before timing)
_ = naive_dft_gpu(signal)
torch.cuda.synchronize() if device.type == 'cuda' else None

start_time_gpu = time.time()
gpu_dft_result = naive_dft_gpu(signal)
torch.cuda.synchronize() if device.type == 'cuda' else None
gpu_duration = time.time() - start_time_gpu

# Method 3: Naive DFT (Nested Loops)
start_time_naive = time.time()
naive_result = naive_dft(signal)
naive_duration = time.time() - start_time_naive

print("--- DFT/FFT Performance Comparison ---")
print(f"1. Built-in FFT Execution Time:      {fft_duration:.6f} seconds")
print(f"2. Vectorized GPU DFT Execution Time:{gpu_duration:.6f} seconds")
print(f"3. Naive Loop DFT Execution Time:    {naive_duration:.6f} seconds\n")

# Verification
fft_cpu = fft_result.cpu().numpy()
gpu_cpu = gpu_dft_result.cpu().numpy()
print(f"GPU DFT matches Built-in FFT: {np.allclose(gpu_cpu, fft_cpu, atol=1e-5)}")