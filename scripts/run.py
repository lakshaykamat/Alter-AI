import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Print versions for debugging
print(insightface.__version__)
print(np.__version__)

# Initialize face detection & recognition model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Load a sample image
img = ins_get_image('t1')

# Detect faces
faces = app.get(img)

# Load face swapper model (ensure model is available at this path)
swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                          download=False,
                                          download_zip=False)

# Select a source face (e.g., the 3rd detected face)
source_face = faces[2]

# Make a copy of the image for swapping
res = img.copy()

# Swap all faces with the source face
for face in faces:
    res = swapper.get(res, face, source_face, paste_back=True)

# Save the final swapped image
output_path = "swapped_result.jpg"
cv2.imwrite(output_path, res)
print(f"Swapped image saved as: {output_path}")
