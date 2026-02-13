"""
Multi-Platform Device Utilities for PyTorch

Provides unified device detection and management for:
- NVIDIA CUDA
- Apple Silicon MPS
- CPU fallback
"""

import torch
import warnings


def get_device(
    prefer_cuda: bool = True,
    prefer_mps: bool = True,
    verbose: bool = True
) -> torch.device:
    """
    Automatically detect and return the best available device.

    Priority: CUDA > MPS > CPU

    Args:
        prefer_cuda: Whether to prefer CUDA if available
        prefer_mps: Whether to prefer MPS if available (and CUDA not)
        verbose: Print device information

    Returns:
        torch.device: The selected device
    """
    device = torch.device('cpu')
    device_name = "CPU"

    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif prefer_mps and torch.backends.mps.is_available():
        # MPS availability check
        if torch.backends.mps.is_built():
            device = torch.device('mps')
            device_name = "Apple Silicon MPS"
        else:
            if verbose:
                warnings.warn("MPS backend not built, falling back to CPU")
    else:
        if verbose:
            print("[Device] No GPU accelerator found, using CPU")

    if verbose:
        print(f"[Device] Selected: {device_name}")
        print(f"[Device] Device type: {device.type}")

    return device


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        dict: Device information including type, name, and memory
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'mps_built': torch.backends.mps.is_built(),
    }

    if info['cuda_available']:
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB

    return info


def print_device_info():
    """Print formatted device information."""
    info = get_device_info()
    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"CUDA available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"  Device: {info['cuda_device_name']}")
        print(f"  Count: {info['cuda_device_count']}")
        print(f"  Memory: {info['cuda_memory_total']:.1f} GB")
    print(f"MPS available: {info['mps_available']}")
    print(f"MPS built: {info['mps_built']}")
    print("=" * 60)

# MPS-specific workarounds
def safe_outer(a: torch.Tensor, b: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Safe outer product that works across all devices.

    MPS may have issues with torch.outer in some PyTorch versions.

    Args:
        a: First tensor [n]
        b: Second tensor [m]
        device: Target device

    Returns:
        Outer product tensor [n, m]
    """
    try:
        return torch.outer(a, b)
    except RuntimeError as e:
        # Fallback for MPS issues
        if device.type == 'mps':
            # Move to CPU, compute, move back
            result = torch.outer(a.cpu(), b.cpu())
            return result.to(device)
        raise e
