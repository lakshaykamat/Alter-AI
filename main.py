import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Print the versions of InsightFace and NumPy (helps debug compatibility issues)
print(insightface.__version__)
print(np.__version__)

# Initialize the face analysis application
# 'buffalo_l' is a pre-trained model for face detection and recognition
app = FaceAnalysis(name='buffalo_l')
# Prepare the model (ctx_id=0 means use GPU if available, else CPU; det_size sets detection input size)
app.prepare(ctx_id=0, det_size=(640, 640))

# Load a sample test image (InsightFace provides sample images by name)
img = ins_get_image('t1')

# Display the original image using Matplotlib
plt.imshow(img[:, :, ::-1])  # Convert BGR (OpenCV) to RGB for plotting
plt.show()

# Detect faces in the image
faces = app.get(img)

# Reload the image for cropping (to avoid modifications during processing)
img = ins_get_image('t1')

# Create a subplot with 1 row and 6 columns to show cropped faces
fig, axs = plt.subplots(1, 6, figsize=(12, 5))

# Iterate through detected faces and crop each face based on bounding box
for i, face in enumerate(faces):
    bbox = face['bbox']  # Get bounding box of the face [x1, y1, x2, y2]
    bbox = [int(b) for b in bbox]  # Convert to integers for slicing
    axs[i].imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])  # Crop and convert BGR → RGB
    axs[i].axis('off')  # Hide axis for cleaner visualization

# Load the face swapping model
# Note: There's a typo here: should be "inswapper_128.onnx" (not "onxx")
swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                          download=False,  # Do not auto-download
                                          download_zip=False)  # Do not unzip automatically

# Select a source face (3rd detected face in the list)
source_face = faces[2]

# Copy the original image (to avoid modifying the original)
res = img.copy()

# Perform face swapping: Replace every detected face with the source face
for face in faces:
    res = swapper.get(res, face, source_face, paste_back=True)

# Display the final swapped-face image
plt.imshow(res[:, :, ::-1])  # Convert to RGB for display
plt.show()
