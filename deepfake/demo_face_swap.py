import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as get_insightface_sample_image

# Print library versions for debugging
print('InsightFace version:', insightface.__version__)
print('NumPy version:', np.__version__)

# Initialize face detection and recognition model
face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Load a sample image from InsightFace
sample_image = get_insightface_sample_image('t1')

# Detect faces in the sample image
detected_faces = face_analyzer.get(sample_image)

# Load the face swapper model (ensure model is available at this path)
face_swapper_model = insightface.model_zoo.get_model('inswapper_128.onnx',
                                                     download=False,
                                                     download_zip=False)

# Select a source face (e.g., the 3rd detected face)
source_face_for_swapping = detected_faces[2]

# Make a copy of the image for swapping
swapped_image = sample_image.copy()

# Swap all faces in the image with the source face
for detected_face in detected_faces:
    swapped_image = face_swapper_model.get(swapped_image, detected_face, source_face_for_swapping, paste_back=True)

# Save the final swapped image
output_image_path = "swapped_result.jpg"
cv2.imwrite(output_image_path, swapped_image)
print(f"Swapped image saved as: {output_image_path}") 