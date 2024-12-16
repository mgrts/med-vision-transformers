import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import nn
from torch.utils.data import Dataset, random_split

from src.config import (SEGMENTED_TEST_ANNOTATIONS_PATH,
                        SEGMENTED_TEST_DATA_DIR,
                        SEGMENTED_TRAIN_ANNOTATIONS_PATH,
                        SEGMENTED_TRAIN_DATA_DIR,
                        SEGMENTED_VAL_ANNOTATIONS_PATH, SEGMENTED_VAL_DATA_DIR,
                        SYNTHETIC_TEST_DATA_DIR, SYNTHETIC_TEST_LABELS_PATH,
                        SYNTHETIC_TRAIN_DATA_DIR, SYNTHETIC_TRAIN_LABELS_PATH,
                        TARGET_CATEGORIES)


class MultiLabelImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load the labels
        if labels_file.endswith('.csv'):
            self.labels = pd.read_csv(labels_file)
        elif labels_file.endswith('.parquet'):
            self.labels = pd.read_parquet(labels_file)
        else:
            raise ValueError("Labels file must be either a CSV or Parquet file.")

        self.target_categories = self.labels.drop(columns='image_name').columns.to_list()

        # Ensure image paths are valid
        self.image_paths = [os.path.join(image_dir, filename)
                            for filename in self.labels['image_name']
                            if filename.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff'))]

        # Create a dictionary for fast lookup of image labels using image_name as key
        self.labels_dict = self.labels.set_index('image_name').to_dict('index')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        image_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Fetch the corresponding labels (circle, square, triangle) for the given image_name
        labels_dict = self.labels_dict[image_name]
        labels = torch.tensor([labels_dict['circle'], labels_dict['square'], labels_dict['triangle']],
                              dtype=torch.float32)

        return image, labels


class ImageDatasetCOCO(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None, exclude_categories=None, include_categories=None):
        """
        A PyTorch Dataset for loading COCO data with compatibility for ImageDataset.

        Args:
            annotation_file (str): Path to the COCO annotation file (.json).
            image_dir (str or Path): Directory where the images are stored.
            transform (callable, optional): Optional transform to be applied on an image.
            exclude_categories (list, optional): List of categories to exclude.
            include_categories (list, optional): List of categories to include.
        """
        self.coco = COCO(annotation_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_categories = TARGET_CATEGORIES

        # Map TARGET_CATEGORIES to category IDs
        self.category_to_id = {cat['name']: cat['id'] for cat in self.coco.cats.values() if
                               cat['name'] in self.target_categories}

        # Map exclude and include categories to their IDs
        self.exclude_category_ids = {cat['name']: cat['id'] for cat in self.coco.cats.values() if
                                     cat['name'] in exclude_categories} if exclude_categories else {}
        self.include_category_ids = {cat['name']: cat['id'] for cat in self.coco.cats.values() if
                                     cat['name'] in include_categories} if include_categories else {}

        # Filter images based on excluded and included categories
        self.image_ids = self.filter_images()

        # Initialize number of classes (+1 for "background" or "no category")
        self.num_classes = len(self.target_categories) + 1

        # Generate labels for all valid images
        self.labels = self._generate_labels()

    def filter_images(self):
        """Filter images based on excluded and included categories."""
        valid_image_ids = []
        for image_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            # Check if any excluded categories are present
            exclude_present = any(ann['category_id'] in self.exclude_category_ids.values() for ann in anns)
            if exclude_present:
                continue

            # Check if any included categories are present (if include_categories is provided)
            if self.include_category_ids:
                include_present = any(ann['category_id'] in self.include_category_ids.values() for ann in anns)
                if not include_present:
                    continue

            valid_image_ids.append(image_id)
        return valid_image_ids

    def _generate_labels(self):
        """Generate one-hot encoded labels for each valid image."""
        labels = []
        for image_id in self.image_ids:
            label = torch.zeros(self.num_classes, dtype=torch.float32)
            annotation_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
            annotations = self.coco.loadAnns(annotation_ids)

            # Assign 1 for matching categories
            for ann in annotations:
                category_name = self.coco.loadCats(ann['category_id'])[0]['name']
                if category_name in self.target_categories:
                    label_idx = self.target_categories.index(category_name)
                    label[label_idx] = 1

            # Set the "background" label if no categories match
            if label.sum() == 0:
                label[-1] = 1

            labels.append(label)
        return labels

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Retrieve the image and its corresponding label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, label)
        """
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        img_path = self.image_dir / image_info['file_name']

        # Open and transform the image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Retrieve the precomputed label
        label = self.labels[idx]
        return image, label


class ImageDataset(Dataset):
    def __init__(self, image_dir, label_mapping=None, transform=None):
        """
        A PyTorch Dataset for loading images and assigning labels based on file path substrings.

        Args:
            image_dir (str or Path): Directory where the images are stored.
            label_mapping (dict, optional): Dictionary mapping substrings to label values.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = Path(image_dir)
        self.label_mapping = label_mapping
        self.transform = transform
        self.image_paths = self._load_image_paths()
        self.labels = self._generate_labels()
        self.num_classes = len(set(label_mapping.values())) if label_mapping else 0

    def _load_image_paths(self):
        """Recursively gather all image file paths from nested folders."""
        image_paths = list(self.image_dir.rglob("*.jpg")) + list(self.image_dir.rglob("*.png"))
        return image_paths

    def _generate_labels(self):
        """Generate labels for each image path based on the label mapping dictionary, if provided."""
        labels = []
        for img_path in self.image_paths:
            label = -1  # Default label if no substring matches or no mapping is provided
            if self.label_mapping:
                for substr, lbl in self.label_mapping.items():
                    if substr in str(img_path):
                        label = lbl
                        break
            labels.append(label)
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Get the label and convert it to a one-hot encoded tensor
        if self.label_mapping:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            label_one_hot = torch.nn.functional.one_hot(label, num_classes=self.num_classes).float()
        else:
            label_one_hot = torch.tensor(0, dtype=torch.long)

        return image, label_one_hot


def collate_fn(inputs):
    """Custom collate function for batching."""
    images, labels = zip(*inputs)
    return {
        'pixel_values': torch.stack(images, dim=0),
        'labels': torch.stack(labels, dim=0)
    }


def create_mask(batch_size, image_size, patch_size, mask_ratio):
    """
    Generate a random mask for the image at the pixel level, based on the given patch size and mask ratio.
    """
    # Calculate the number of patches based on the image size and patch size
    num_patches_per_dim = image_size // patch_size
    num_patches = num_patches_per_dim ** 2  # Total number of patches in the image

    # Initialize the mask to all ones (unmasked)
    mask = torch.ones(batch_size, num_patches, dtype=torch.bool)

    # Calculate the number of patches to mask based on the mask ratio
    num_masked = int(mask_ratio * num_patches)

    # Randomly select patches to mask for each image in the batch
    for i in range(batch_size):
        mask[i, torch.randperm(num_patches)[:num_masked]] = False

    # Reshape the mask to match the patch grid dimensions (num_patches_per_dim x num_patches_per_dim)
    mask = mask.view(batch_size, num_patches_per_dim, num_patches_per_dim)  # [batch_size, h_patches, w_patches]

    # Expand the mask to have a channel dimension, simulating a 3-channel RGB mask
    mask = mask.unsqueeze(1).expand(-1, 3, -1, -1)  # [batch_size, 3, h_patches, w_patches]

    # Upsample the mask to match the pixel resolution of the image
    mask = nn.functional.interpolate(mask.float(), scale_factor=patch_size).bool()

    return mask


def load_datasets(dateset_type, transform=None):
    if dateset_type == 'real':
        train_dataset = ImageDatasetCOCO(
            annotation_file=SEGMENTED_TRAIN_ANNOTATIONS_PATH,
            image_dir=SEGMENTED_TRAIN_DATA_DIR,
            transform=transform
        )
        val_dataset = ImageDatasetCOCO(
            annotation_file=SEGMENTED_VAL_ANNOTATIONS_PATH,
            image_dir=SEGMENTED_VAL_DATA_DIR,
            transform=transform
        )
        test_dataset = ImageDatasetCOCO(
            annotation_file=SEGMENTED_TEST_ANNOTATIONS_PATH,
            image_dir=SEGMENTED_TEST_DATA_DIR,
            transform=transform
        )
        val_dataset = val_dataset + test_dataset
    elif dateset_type == 'synthetic':
        # Load the dataset and split into train and validation sets
        train_dataset = MultiLabelImageDataset(
            image_dir=SYNTHETIC_TRAIN_DATA_DIR,
            labels_file=SYNTHETIC_TRAIN_LABELS_PATH,
            transform=transform
        )
        val_dataset = MultiLabelImageDataset(
            image_dir=SYNTHETIC_TEST_DATA_DIR,
            labels_file=SYNTHETIC_TEST_LABELS_PATH,
            transform=transform
        )
        val_size = int(0.1 * len(val_dataset))  # 10% of the dataset
        remaining_size = len(val_dataset) - val_size
        val_dataset, _ = random_split(val_dataset, [val_size, remaining_size])
    else:
        raise ValueError(f"Unknown dataset type: {dateset_type}.")

    return train_dataset, val_dataset
