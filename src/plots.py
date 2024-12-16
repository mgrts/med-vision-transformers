import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from PIL import Image
from pycocotools.coco import COCO

from src.config import (FIGURES_DIR, SEGMENTED_TRAIN_ANNOTATIONS_PATH,
                        SEGMENTED_TRAIN_DATA_DIR)

app = typer.Typer(pretty_exceptions_show_locals=False)


def create_caries_mask(coco, image_info, category_ids):
    """
    Create a mask and bounding box overlays for the CARIES category on a COCO image.
    """
    # Load annotations for the image
    annotation_ids = coco.getAnnIds(imgIds=image_info['id'], catIds=category_ids)
    annotations = coco.loadAnns(annotation_ids)

    # Create a blank mask for overlay
    mask = np.zeros((image_info['height'], image_info['width']))

    plt.imshow(Image.open(image_info['file_name']).convert('RGB'))
    plt.axis('off')

    for annotation in annotations:
        # For segmentation masks
        if 'segmentation' in annotation:
            m = coco.annToMask(annotation)
            mask = np.maximum(mask, m)

        # For bounding boxes
        if 'bbox' in annotation:
            x, y, w, h = annotation['bbox']
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))

    plt.imshow(mask, alpha=0.5, cmap='jet')


@app.command()
def main(
        coco_annotation_file: Path = SEGMENTED_TRAIN_ANNOTATIONS_PATH,
        image_dir: Path = SEGMENTED_TRAIN_DATA_DIR,
        output_dir: Path = FIGURES_DIR,
        target_category: str = 'CARIES',
        max_images: int = 2
):
    """
    Highlight a target category in COCO images and save annotated images.
    """
    logger.info(f"Loading COCO annotations from {coco_annotation_file}")
    coco = COCO(coco_annotation_file)
    category_ids = coco.getCatIds(catNms=[target_category])
    if not category_ids:
        raise ValueError(f'Category "{target_category}" not found in annotations.')

    # Get images with the target category
    image_ids = coco.getImgIds(catIds=category_ids)
    if not image_ids:
        raise ValueError(f'No images found with category "{target_category}".')

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory set to {output_dir}")

    saved_count = 0

    for image_id in image_ids:
        if saved_count >= max_images:
            logger.info(f"Reached maximum limit of {max_images} images. Stopping.")
            break

        image_info = coco.loadImgs(image_id)[0]
        img_path = image_dir / image_info['file_name']
        image_info['file_name'] = str(img_path)  # Set the full image path for loading

        if not img_path.exists():
            logger.warning(f"Image file {img_path} does not exist. Skipping.")
            continue

        logger.info(f"Processing image {img_path}")
        plt.figure(figsize=(10, 10))
        create_caries_mask(coco, image_info, category_ids)

        output_path = output_dir / f'annotated_image_{image_id}.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        logger.success(f"Saved annotated image to {output_path}")
        saved_count += 1

    logger.info(f"Processed and saved {saved_count} images with category '{target_category}'.")


if __name__ == '__main__':
    app()
