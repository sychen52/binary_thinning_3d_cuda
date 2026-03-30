import torch
from . import cuda_thinning_ext


def binary_thinning(tensor: torch.Tensor, mode: int = 0) -> torch.Tensor:
    """
    3D binary thinning using CUDA.
    The operation is performed on the tensor provided (in-place for the binary representation).
    If the tensor is on CPU, it will be moved to CUDA for processing and then copied back to the original tensor.

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

    # Ensure it's uint8 and contiguous before passing to CUDA extension
    # We do this check here to keep the C++ extension focused on the algorithm
    if tensor.dtype != torch.uint8 or not tensor.is_contiguous():
        # This creates a new tensor, we won't be able to modify the original in-place
        # if the user passed something like a float tensor or a non-contiguous one.
        work_tensor = (tensor != 0).to(torch.uint8).contiguous()
        cuda_thinning_ext.binary_thinning(work_tensor, mode)
        # If the original was on the same device, we can try to copy back
        if tensor.shape == work_tensor.shape:
            try:
                tensor.copy_(work_tensor)
            except Exception:
                pass  # Might fail if types are incompatible for copy_
        return work_tensor
    else:
        # In-place ensure binary (0 or 1)
        tensor.clamp_(0, 1)
        cuda_thinning_ext.binary_thinning(tensor, mode)
        return tensor
