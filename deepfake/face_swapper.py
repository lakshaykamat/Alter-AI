import cv2
import insightface
from insightface.app import FaceAnalysis
import os

# Initialize the face analysis and swapping models once
face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
face_swapper_model = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

def swap_face_from_source_to_target(
    source_image_path: str,
    target_image_path: str,
    output_image_path: str = "swapped_result.jpg"
) -> str:
    """
    Swap the face from the source image onto the face in the target image.
    Both images must contain exactly ONE face.

    Args:
        source_image_path (str): Path to the source image (face to be swapped in)
        target_image_path (str): Path to the target image (face to replace)
        output_image_path (str): Path to save the swapped image

    Returns:
        str: Path to the saved swapped image
    """
    # Load images
    source_image = cv2.imread(source_image_path)
    target_image = cv2.imread(target_image_path)
    if source_image is None or target_image is None:
        raise ValueError("Could not read one or both images. Check file paths.")

    # Detect faces
    source_face_list = face_analyzer.get(source_image)
    target_face_list = face_analyzer.get(target_image)

    # Ensure exactly one face per image
    if len(source_face_list) != 1 or len(target_face_list) != 1:
        raise ValueError("Each image must contain exactly ONE face.")

    # Extract face objects
    source_face = source_face_list[0]
    target_face = target_face_list[0]

    # Perform the swap
    swapped_image = face_swapper_model.get(target_image.copy(), target_face, source_face, paste_back=True)

    # Save result
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, swapped_image)
    print(f"Swapped face image saved at: {output_image_path}")
    return output_image_path 