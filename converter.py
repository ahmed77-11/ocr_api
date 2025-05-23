import os
import cv2
import shutil

scans_dir = "C:/Users/mghir/Desktop/layoutMl/archive/scans/scans/"
masks_dir = "C:/Users/mghir/Desktop/layoutMl/archive/ground-truth-pixel/ground-truth-pixel"
output_dir = "my-classification-dataset"

os.makedirs(f"{output_dir}/with_stamp", exist_ok=True)
os.makedirs(f"{output_dir}/without_stamp", exist_ok=True)

for file in os.listdir(scans_dir):
    scan_path = os.path.join(scans_dir, file)
    mask_path = os.path.join(masks_dir, file)

    if not os.path.exists(mask_path):
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if cv2.countNonZero(mask) > 0:
        shutil.copy(scan_path, f"{output_dir}/with_stamp/{file}")
    else:
        shutil.copy(scan_path, f"{output_dir}/without_stamp/{file}")
