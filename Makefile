image-2.1:
	uv pip run --python image-gen/stable_diffusion_2_1_batch.py

image-3.5:
	uv pip run --python image-gen/stable_diffusion_3_5_single.py

face-swapper:
	uv pip run --python deepfake/face_swapper.py

# Sync all project dependencies
sync-all:
	uv pip sync deepfake/pyproject.toml
	uv pip sync image-gen/pyproject.toml 