import torch
from . import cuda_thinning_ext


def binary_thinning(tensor: torch.Tensor, mode: int = 0) -> torch.Tensor:
    """
    3D binary thinning using CUDA. If the tensor is on CUDA, the operation is performed in-place.
    If it's on CPU, it will be moved to CUDA for processing and then copied back.

    Args:
        tensor (torch.Tensor): A 3D tensor. All non-zero values are treated as foreground.
        mode (int):
            0 = GPU Subgrid (Fastest, preserves topology, fully GPU)
            1 = CPU Sequential Re-check (Matches ITK exactly, slower)

    Returns:
        torch.Tensor: The thinned binary tensor.
    """
    if tensor.dim() != 3:
        raise ValueError("Tensor must be 3D.")
    if mode not in [0, 1]:
        raise ValueError("Mode must be 0 (GPU Subgrid) or 1 (CPU Sequential).")

    # We must operate on a contiguous ByteTensor (uint8)
    if not tensor.is_cuda:
        work_tensor = tensor.to(device="cuda", dtype=torch.uint8)
    elif tensor.dtype != torch.uint8 or not tensor.is_contiguous():
        work_tensor = (tensor != 0).to(torch.uint8).contiguous()
    else:
        work_tensor = tensor
        tensor[tensor != 0] = 1  # Ensure binary

    cuda_thinning_ext.binary_thinning(work_tensor, mode)

    if not tensor.is_cuda:
        return work_tensor.cpu()

    if work_tensor is not tensor:
        tensor.copy_(work_tensor)

    return tensor
