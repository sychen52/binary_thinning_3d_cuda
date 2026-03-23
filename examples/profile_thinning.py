import torch
import SimpleITK as sitk
import time
from binary_thinning_3d import binary_thinning

img_sitk = sitk.ReadImage("data/1_CT_HR_label_airways.nii.gz")
img_array = sitk.GetArrayFromImage(img_sitk)
tensor = torch.from_numpy(img_array > 0).to(torch.uint8).cuda()

torch.cuda.synchronize()
start = time.time()
binary_thinning(tensor, mode=0)
torch.cuda.synchronize()
print(f"Time: {time.time() - start:.4f}s")
