# Deep Fake Face Swapper

A simple Python project for face swapping using the [InsightFace](https://github.com/deepinsight/insightface) library. This tool allows you to swap a face from a source image onto a target image using pre-trained models.

## Features
- Detects faces in images using InsightFace
- Swaps a face from a source image onto a target image (both must have exactly one face)
- Saves the swapped image to the `output/` directory

## Project Structure
- `scripts/fun.py` – Core logic for face swapping (see `swap_faces` function)
- `scripts/run.py` – Example script for face detection and swapping
- `main.py` – Another example script for face detection and swapping
- `models/inswapper_128.onnx` – Pre-trained ONNX model for face swapping
- `media/` – Example images (`source.jpg`, `source2.jpg`, `target.jpg`)
- `output/` – Swapped images are saved here

## Requirements
- Python 3.10
- See `pyproject.toml` for dependencies:
  - numpy
  - matplotlib
  - opencv-python
  - onnxruntime
  - insightface

## Installation
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or, if using poetry
   poetry install
   ```
   *(If you use `pyproject.toml`, generate `requirements.txt` with `poetry export` if needed.)*

## Usage
### Swap Faces Between Two Images
Run the core function from `scripts/fun.py`:
```python
from scripts.fun import swap_faces
swap_faces('media/source2.jpg', 'media/target.jpg', 'output/swapped.jpg')
```

### Example Script
You can also run the example script:
```bash
python scripts/run.py
```
This will perform face detection and swapping on a sample image and save the result as `swapped_result.jpg`.

## Notes
- Both source and target images must contain exactly one face for the swap to work.
- The ONNX model (`inswapper_128.onnx`) must be present in the `models/` directory.
- Example images are provided in the `media/` directory.

## License
This project is for educational purposes only.
