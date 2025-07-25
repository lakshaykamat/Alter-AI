import numpy as np
from deepfake.face_swapper import swap_face_from_source_to_target
import cv2
import os

# Paths for demonstration
SOURCE_IMAGE_PATH = 'media/source2.jpg'
TARGET_IMAGE_PATH = 'media/target.jpg'
OUTPUT_IMAGE_PATH = 'output/swapped.jpg'

# Perform face swap using the deepfake module
swap_face_from_source_to_target(
    source_image_path=SOURCE_IMAGE_PATH,
    target_image_path=TARGET_IMAGE_PATH,
    output_image_path=OUTPUT_IMAGE_PATH
)

# Optionally, print confirmation
print(f"Face swap complete. Output saved to: {OUTPUT_IMAGE_PATH}")
