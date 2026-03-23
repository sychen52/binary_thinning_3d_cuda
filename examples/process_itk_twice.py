import itk
import numpy as np
import time


def process_nifti_itk(input_path):
    print(f"Loading {input_path}...")

    # -----------------------------------------
    # 1. ITK Thinning (Run 1)
    # -----------------------------------------
    print("\n--- Starting ITK thinning (Run 1) ---")
    itk_image1 = itk.imread(input_path, itk.UC)
    start_time = time.time()
    itk_thickness_map1 = itk.MedialThicknessImageFilter3D.New(itk_image1)
    itk_thickness_map1.Update()
    itk_time_1 = time.time() - start_time
    print(f"Run 1 finished in {itk_time_1:.4f} seconds.")

    itk_thinned_array_1 = itk.array_from_image(itk_thickness_map1.GetOutput())
    itk_binary_array_1 = (itk_thinned_array_1 > 0).astype(np.uint8)

    # -----------------------------------------
    # 2. ITK Thinning (Run 2)
    # -----------------------------------------
    print("\n--- Starting ITK thinning (Run 2) ---")
    itk_image2 = itk.imread(input_path, itk.UC)
    start_time = time.time()
    itk_thickness_map2 = itk.MedialThicknessImageFilter3D.New(itk_image2)
    itk_thickness_map2.Update()
    itk_time_2 = time.time() - start_time
    print(f"Run 2 finished in {itk_time_2:.4f} seconds.")

    itk_thinned_array_2 = itk.array_from_image(itk_thickness_map2.GetOutput())
    itk_binary_array_2 = (itk_thinned_array_2 > 0).astype(np.uint8)

    print(f"\nRun 1 Thinned sum (volume): {np.sum(itk_binary_array_1)}")
    print(f"Run 2 Thinned sum (volume): {np.sum(itk_binary_array_2)}")

    # -----------------------------------------
    # 3. Comparison
    # -----------------------------------------
    intersection = np.logical_and(itk_binary_array_1, itk_binary_array_2).sum()
    union = np.logical_or(itk_binary_array_1, itk_binary_array_2).sum()

    sum_1 = np.sum(itk_binary_array_1)
    sum_2 = np.sum(itk_binary_array_2)
    dice = 2.0 * intersection / (sum_1 + sum_2) if (sum_1 + sum_2) > 0 else 1.0
    diff = np.sum(itk_binary_array_1 != itk_binary_array_2)

    print("\n--- Comparison (Run 1 vs Run 2) ---")
    print(f"Number of different pixels: {diff}")
    print(f"Intersection: {intersection}")
    print(f"Union: {union}")
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"Are results identical? {'Yes' if diff == 0 else 'No'}")


if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "1_CT_HR_label_airways.nii.gz")

    process_nifti_itk(input_file)
