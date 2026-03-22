import torch
import SimpleITK as sitk
import numpy as np
import time
import os
from binary_thinning_3d import binary_thinning

def process_nifti(input_path):
    print(f"Loading {input_path}...")
    
    img_sitk = sitk.ReadImage(input_path)
    img_array = sitk.GetArrayFromImage(img_sitk)
    
    print(f"Original shape: {img_array.shape}, type: {img_array.dtype}")
    print(f"Original sum (volume): {np.sum(img_array > 0)}")
    
    tensor_subgrid = torch.from_numpy(img_array > 0).to(torch.uint8).cuda()
    tensor_cpu_sync = torch.from_numpy(img_array > 0).to(torch.uint8).cuda()
    
    # 0. GPU Subgrid (8-color)
    print("\n--- 0. Starting CUDA thinning (GPU Subgrid 8-Color) ---")
    torch.cuda.synchronize()
    start_time = time.time()
    binary_thinning(tensor_subgrid, mode=0)
    torch.cuda.synchronize()
    cuda_time_subgrid = time.time() - start_time
    print(f"CUDA GPU Subgrid Thinning finished in {cuda_time_subgrid:.4f} seconds.")
    cuda_binary_array_subgrid = (tensor_subgrid.cpu().numpy() > 0).astype(np.uint8)

    # 1. GPU Hybrid (CPU Sync)
    print("\n--- 1. Starting CUDA thinning (Hybrid CPU-Sync) ---")
    torch.cuda.synchronize()
    start_time = time.time()
    binary_thinning(tensor_cpu_sync, mode=1)
    torch.cuda.synchronize()
    cuda_time_cpu_sync = time.time() - start_time
    print(
        f"CUDA Hybrid CPU-Sync Thinning finished in {cuda_time_cpu_sync:.4f} seconds."
    )
    cuda_binary_array_cpu_sync = (tensor_cpu_sync.cpu().numpy() > 0).astype(np.uint8)

    # 2. CPU (ITK)
    print("\n--- 2. Starting ITK thinning (CPU) ---")
    try:
        import itk
    except ImportError:
        print("itk module not found. Skipping CPU benchmark.")
        itk_binary_array = None
        itk_time = 0.0
    else:
        itk_image = itk.imread(input_path, itk.UC)
        start_time = time.time()
        thinning_filter = itk.BinaryThinningImageFilter3D.New(itk_image)
        thinning_filter.Update()
        itk_time = time.time() - start_time
        print(f"ITK Thinning finished in {itk_time:.4f} seconds.")

        itk_thinned_array = itk.array_from_image(thinning_filter.GetOutput())
        itk_binary_array = (itk_thinned_array > 0).astype(np.uint8)

    print("\n--- Summary ---")
    print(f"Original sum (volume)            : {np.sum(img_array > 0)}")
    print(f"Mode 0 (GPU Subgrid) sum         : {np.sum(cuda_binary_array_subgrid)}")
    print(f"Mode 1 (Hybrid CPU) sum          : {np.sum(cuda_binary_array_cpu_sync)}")
    if itk_binary_array is not None:
        print(f"ITK CPU sum                      : {np.sum(itk_binary_array)}")

    print("\n--- Timing ---")
    print(f"Mode 0 (GPU Subgrid)             : {cuda_time_subgrid:.4f} s")
    print(f"Mode 1 (Hybrid CPU)              : {cuda_time_cpu_sync:.4f} s")
    if itk_binary_array is not None:
        print(f"2. CPU (ITK)                     : {itk_time:.4f} s")
        speedup_subgrid = itk_time / cuda_time_subgrid
        speedup_cpu_sync = itk_time / cuda_time_cpu_sync
        print(f"Speedup vs ITK (Mode 0)          : {speedup_subgrid:.2f}x")
        print(f"Speedup vs ITK (Mode 1)          : {speedup_cpu_sync:.2f}x")

    if itk_binary_array is not None:
        print("\n--- Comparison (GPU vs CPU ITK) ---")
        diff_0 = np.sum(cuda_binary_array_subgrid != itk_binary_array)
        diff_1 = np.sum(cuda_binary_array_cpu_sync != itk_binary_array)
        print(f"Mode 0 differences from ITK      : {diff_0} (due to parallel subgrid deletion order)")
        print(f"Mode 1 differences from ITK      : {diff_1}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'data', '1_CT_HR_label_airways.nii.gz')
    process_nifti(input_file)
