import os
from PIL import Image
import imageio
import logging

logger = logging.getLogger(__name__)

def ensure_dirs():
    for folder in ["app_static/screenshots", "app_static/gifs", "app_static/pdfs"]:
        os.makedirs(folder, exist_ok=True)
        logger.debug(f"Ensured directory: {folder}")

def generate_gif_from_images(image_paths, output_path):
    images = []
    target_size = None

    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                target_size = img.size
                break
            except Exception as e:
                logger.error(f"Failed to open {img_path}: {e}")

    if not target_size:
        logger.warning("No valid images found to determine target size.")
        return

    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                if img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                images.append(img)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")

    if len(images) >= 2:
        try:
            imageio.mimsave(output_path, images, fps=1)
            logger.info(f"GIF generated: {output_path}")
        except Exception as e:
            logger.error(f"Failed to create GIF: {e}")
    else:
        logger.warning("Not enough images to create a GIF.")
