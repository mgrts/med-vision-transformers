import os

import matplotlib.pyplot as plt
import torch
import torchvision
import typer
from loguru import logger
from torch import nn
from transformers import AutoModel

from src.config import (BASE_MODEL_NAME, BRATS_TRAIN_DATA_DIR,
                        BRATS_TRAIN_SURVIVAL_INFO_PATH, FIGURES_DIR,
                        IMAGE_SIZE, MASK_RATIO, MODELS_DIR,
                        SEGMENTED_TRAIN_ANNOTATIONS_PATH,
                        SEGMENTED_TRAIN_DATA_DIR, ULTRASOUND_DATA_DIR)
from src.modeling.data_processing import (ImageDataset, ImageDatasetBrats,
                                          ImageDatasetCOCO, create_mask)
from src.modeling.models import MIMTransformer, MultiTaskTransformer
from src.modeling.utils import DEVICE, EVAL_TRANSFORM, get_model_run_id

app = typer.Typer(pretty_exceptions_show_locals=False)


def load_dino_model(model_load_path, model_name='facebook/dino-vits8'):
    """
    Load a pretrained DINO model without the MIM head for attention visualization.
    """
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False, attn_implementation='eager')
    model.load_state_dict(torch.load(model_load_path))  # Load weights from model_load_path
    return model


def load_mim_model(dino_model, mim_model_load_path):
    """
    Create and load a MIM model by combining the DINO model with a MIM head.
    """
    model = MIMTransformer(base_model=dino_model, image_size=IMAGE_SIZE)
    model.load_state_dict(torch.load(mim_model_load_path))  # Load the MIM head weights
    return model


def visualize_attention_maps(pixel_values, model, output_dir):
    """
    Forward the image through the attention model and visualize the attention maps.
    """
    outputs = model(pixel_values, output_attentions=True)
    attentions = outputs.attentions[-1]  # Last layer attention maps
    nh = attentions.shape[1]  # Number of heads

    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # Reshape attention maps
    w_featmap = pixel_values.shape[-2] // model.config.patch_size
    h_featmap = pixel_values.shape[-1] // model.config.patch_size

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=model.config.patch_size, mode='nearest'
    )[0].cpu()

    # Save attention maps
    for j in range(nh):
        fname = os.path.join(output_dir, f'attn-head-{j}.png')
        plt.figure()
        plt.imshow(attentions[j].detach().numpy())
        plt.imsave(fname=fname, arr=attentions[j].detach().numpy(), format='png')
        logger.info(f'{fname} saved.')


@app.command()
def main(model_type: str = 'BRATS_MIM_MSE',
         image_id: int = 0):
    """
    Evaluate the MIM model by visualizing attention maps for a sample image.
    Also save the initial, masked, and restored images.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    run_id = get_model_run_id(model_type)

    model_load_path = MODELS_DIR / run_id / 'base_model.pth'
    mim_model_load_path = MODELS_DIR / run_id / 'model.pth'

    # Load dataset
    if model_type == 'ULTRASOUND':
        dataset = ImageDataset(
            image_dir=ULTRASOUND_DATA_DIR,
            transform=EVAL_TRANSFORM
        )
    elif model_type == 'ULTRASOUND_CLASS':
        dataset = ImageDataset(
            image_dir=ULTRASOUND_DATA_DIR,
            transform=EVAL_TRANSFORM
        )
    elif model_type == 'ULTRASOUND_MULTITASK':
        dataset = ImageDataset(
            image_dir=ULTRASOUND_DATA_DIR,
            transform=EVAL_TRANSFORM
        )
    elif model_type == 'REAL_MULTITASK':
        dataset = ImageDatasetCOCO(
            annotation_file=SEGMENTED_TRAIN_ANNOTATIONS_PATH,
            image_dir=SEGMENTED_TRAIN_DATA_DIR,
            transform=EVAL_TRANSFORM
        )
    elif model_type == 'REAL_CLASS_MSE':
        dataset = ImageDatasetCOCO(
            annotation_file=SEGMENTED_TRAIN_ANNOTATIONS_PATH,
            image_dir=SEGMENTED_TRAIN_DATA_DIR,
            transform=EVAL_TRANSFORM
        )
    elif model_type == 'BRATS_MIM_MSE':
        dataset = ImageDatasetBrats(
            image_dir=BRATS_TRAIN_DATA_DIR,
            info_path=BRATS_TRAIN_SURVIVAL_INFO_PATH,
            transform=EVAL_TRANSFORM
        )
    else:
        dataset = ImageDatasetCOCO(
            annotation_file=SEGMENTED_TRAIN_ANNOTATIONS_PATH,
            image_dir=SEGMENTED_TRAIN_DATA_DIR,
            transform=EVAL_TRANSFORM
        )

    pixel_values = dataset[image_id][0].unsqueeze(0).to(DEVICE)

    base_model = AutoModel.from_pretrained(
        BASE_MODEL_NAME, add_pooling_layer=False, attn_implementation='eager'
    )

    if 'MULTITASK' in model_type:
        mt_model = MultiTaskTransformer(base_model=base_model, image_size=IMAGE_SIZE, num_classes=2)
        mt_model.load_state_dict(torch.load(model_load_path))

        dino_model = mt_model.base_model
        patch_size = dino_model.config.patch_size
        dino_model.to(DEVICE)

        mim_head = mt_model.mim_head
        mim_model = nn.Sequential(dino_model, mim_head)
        mim_model.to(DEVICE)
    else:
        # Load the DINO model without the head (for attention visualization)
        dino_model = load_dino_model(model_load_path)

        # Get patch info from the model
        patch_size = dino_model.config.patch_size
        embed_dim = dino_model.config.hidden_size

        mim_model = load_mim_model(dino_model, mim_model_load_path, embed_dim, patch_size)
        mim_model.to(DEVICE)

    # Generate a mask
    batch_size = pixel_values.size(0)
    mask = create_mask(batch_size, IMAGE_SIZE, patch_size, MASK_RATIO).to(DEVICE)

    # Apply the mask directly to the pixels (no need to extract patches first)
    masked_image = pixel_values.clone()
    masked_image[~mask] = 1

    # Get restored image (output from the MIM model)
    restored_image = mim_model(masked_image)

    # Visualize and save the initial, masked, and restored images
    torchvision.utils.save_image(pixel_values, os.path.join(FIGURES_DIR, 'initial_image.png'))
    torchvision.utils.save_image(masked_image, os.path.join(FIGURES_DIR, 'masked_image.png'))
    torchvision.utils.save_image(restored_image, os.path.join(FIGURES_DIR, 'restored_image.png'))

    # Visualize and save attention maps
    visualize_attention_maps(pixel_values, dino_model, FIGURES_DIR)

    logger.info(f'Evaluation completed. Images and attention maps saved to {FIGURES_DIR}')


if __name__ == '__main__':
    app()
