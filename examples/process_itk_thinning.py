import itk
import numpy as np


def run():
    input_path = "data/1_CT_HR_label_airways.nii.gz"
    itk_image = itk.imread(input_path, itk.UC)

    thinning_filter = itk.BinaryThinningImageFilter3D.New(itk_image)
    thinning_filter.Update()

    itk_thinned = itk.array_from_image(thinning_filter.GetOutput())
    itk_binary = (itk_thinned > 0).astype(np.uint8)

    print(f"ITK BinaryThinningImageFilter3D sum: {np.sum(itk_binary)}")


if __name__ == "__main__":
    run()
