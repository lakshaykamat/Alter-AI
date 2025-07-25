import cv2
import insightface
from insightface.app import FaceAnalysis
import os

# Initialize models once (so we don't reload them every time)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

def swap_faces(source_path: str, target_path: str, output_path: str = "swapped_result.jpg") -> str:
    """
    Swap the face from source image onto the face in target image.
    Both images must contain exactly ONE face.
    
    Args:
        source_path (str): Path to the source image (face to be swapped in)
        target_path (str): Path to the target image (face to replace)
        output_path (str): Path to save the swapped image
    
    Returns:
        str: Path to the saved swapped image
    """
    # Load images
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    if source_img is None or target_img is None:
        raise ValueError("Could not read one or both images. Check file paths.")

    # Detect faces
    source_faces = app.get(source_img)
    target_faces = app.get(target_img)

    # Ensure exactly one face per image
    if len(source_faces) != 1 or len(target_faces) != 1:
        raise ValueError("Each image must contain exactly ONE face.")

    # Extract face objects
    source_face = source_faces[0]
    target_face = target_faces[0]

    # Perform the swap
    result_img = swapper.get(target_img.copy(), target_face, source_face, paste_back=True)

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result_img)
    print(f"Swapped face image saved at: {output_path}")
    return output_path

# Example usage:
swap_faces("source2.jpg", "target.jpg", "output/swapped.jpg")
