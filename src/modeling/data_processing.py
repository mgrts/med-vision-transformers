from bisect import bisect_right
import os
from pathlib import Path

from PIL import Image
from loguru import logger
import nibabel as nib
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import torch
from torch import nn
from torch.utils.data import ConcatDataset, Dataset, Subset

from src.config import (
    BRATS_SLICE_INDICES,
    BRATS_SURVIVAL_SLICE,
    BRATS_SURVIVAL_THRESHOLD_DAYS,
    BRATS_TUMOR_AREA_THRESHOLD,
    MASK_RATIO,
    TARGET_CATEGORIES,
)


class MultiLabelImageDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load the labels
        if labels_file.endswith(".csv"):
            self.labels = pd.read_csv(labels_file)
        elif labels_file.endswith(".parquet"):
            self.labels = pd.read_parquet(labels_file)
        else:
            raise ValueError("Labels file must be either a CSV or Parquet file.")

        self.target_categories = self.labels.drop(columns="image_name").columns.to_list()

        # Ensure image paths are valid
        self.image_paths = [
            os.path.join(image_dir, filename)
            for filename in self.labels["image_name"]
            if filename.endswith(("jpg", "jpeg", "png", "bmp", "tiff"))
        ]

        # Create a dictionary for fast lookup of image labels using image_name as key
        self.labels_dict = self.labels.set_index("image_name").to_dict("index")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        image_name = os.path.basename(img_path)
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Fetch the corresponding labels (circle, square, triangle) for the given image_name
        labels_dict = self.labels_dict[image_name]
        labels = torch.tensor(
            [labels_dict[c] for c in self.target_categories], dtype=torch.float32
        )

        return image, labels


class ImageDatasetCOCO(Dataset):
    def __init__(
        self,
        annotation_file,
        image_dir,
        transform=None,
        exclude_categories=None,
        include_categories=None,
    ):
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
        self.category_to_id = {
            cat["name"]: cat["id"]
            for cat in self.coco.cats.values()
            if cat["name"] in self.target_categories
        }

        # Map exclude and include categories to their IDs
        self.exclude_category_ids = (
            {
                cat["name"]: cat["id"]
                for cat in self.coco.cats.values()
                if cat["name"] in exclude_categories
            }
            if exclude_categories
            else {}
        )
        self.include_category_ids = (
            {
                cat["name"]: cat["id"]
                for cat in self.coco.cats.values()
                if cat["name"] in include_categories
            }
            if include_categories
            else {}
        )

        # Filter images based on excluded and included categories
        self.image_ids = self.filter_images()

        # One binary output per target category (no separate complementary background class).
        self.num_classes = len(self.target_categories)

        # Generate labels for all valid images
        self.labels = self._generate_labels()

    def filter_images(self):
        """Filter images based on excluded and included categories."""
        valid_image_ids = []
        for image_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            # Check if any excluded categories are present
            exclude_present = any(
                ann["category_id"] in self.exclude_category_ids.values() for ann in anns
            )
            if exclude_present:
                continue

            # Check if any included categories are present (if include_categories is provided)
            if self.include_category_ids:
                include_present = any(
                    ann["category_id"] in self.include_category_ids.values() for ann in anns
                )
                if not include_present:
                    continue

            valid_image_ids.append(image_id)
        return valid_image_ids

    def _generate_labels(self):
        """Generate a multi-hot label per image over the target categories.

        For a single target (CARIES) this is a binary label: [1.] if the category is
        present, [0.] otherwise. Absence is encoded as all-zeros, not a separate class.
        """
        labels = []
        for image_id in self.image_ids:
            label = torch.zeros(self.num_classes, dtype=torch.float32)
            annotation_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
            annotations = self.coco.loadAnns(annotation_ids)

            for ann in annotations:
                category_name = self.coco.loadCats(ann["category_id"])[0]["name"]
                if category_name in self.target_categories:
                    label[self.target_categories.index(category_name)] = 1.0

            labels.append(label)
        return labels

    def get_categories(self, idx):
        """Return the set of annotation category names for the image at index ``idx``."""
        image_id = self.image_ids[idx]
        annotation_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        annotations = self.coco.loadAnns(annotation_ids)
        return {self.coco.loadCats(ann["category_id"])[0]["name"] for ann in annotations}

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
        img_path = self.image_dir / image_info["file_name"]

        # Open and transform the image
        image = Image.open(img_path).convert("RGB")
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
        self.image_paths, self.labels = self._load_paths_and_labels()
        self.num_classes = 1  # single binary output

    def _load_paths_and_labels(self):
        """Gather image paths and binary labels via path-substring matching.

        Files matching no label substring are SKIPPED (with a warning) rather than
        silently coerced to a class, so unlabeled files cannot contaminate the data.
        """
        all_paths = sorted(self.image_dir.rglob("*.jpg")) + sorted(self.image_dir.rglob("*.png"))
        paths, labels, skipped = [], [], 0
        for img_path in all_paths:
            label = None
            if self.label_mapping:
                for substr, lbl in self.label_mapping.items():
                    if substr in str(img_path):
                        label = lbl
                        break
            else:
                label = 0  # no mapping (e.g. inference/visualization): single negative class
            if label is None:
                skipped += 1
                continue
            paths.append(img_path)
            labels.append([float(label)])
        if skipped:
            logger.warning(
                f"ImageDataset: skipped {skipped} file(s) under {self.image_dir} matching no "
                f"label substring in {list(self.label_mapping)}"
            )
        return paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


class ImageDatasetBrats(Dataset):
    def __init__(self, image_dir, info_path, transform=None):
        """
        A PyTorch Dataset for loading images and assigning labels based on file path substrings.

        Args:
            image_dir (str or Path): Directory where the images are stored.
            label_dir (str or Path): Directory where the labels are stored.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = Path(image_dir)
        self.info_path = Path(info_path)
        self.transform = transform
        # NOTE: binary 2-year survival predicted from a SINGLE axial slice — a deliberate
        # simplification; a clinically faithful model would use the full 3D volume.
        self.id_to_label = self._build_label_map()
        self.image_paths = self._load_image_paths()
        self.labels = [[float(self.id_to_label[p.parent.name])] for p in self.image_paths]
        self.num_classes = 1

    def _build_label_map(self):
        """Map Brats20ID -> binary 2-year-survival label, dropping censored rows.

        'ALIVE (... days later)' and other non-numeric Survival_days are right-censored:
        the true survival time is unknown, so these patients cannot be labeled for a
        survived-beyond-2-years target and are excluded rather than mislabeled.
        """
        df = pd.read_csv(self.info_path)
        days = pd.to_numeric(df["Survival_days"], errors="coerce")
        valid = df.loc[days.notna(), ["Brats20ID"]].copy()
        valid["days"] = days[days.notna()]
        dropped = len(df) - len(valid)
        if dropped:
            logger.warning(
                f"ImageDatasetBrats: dropped {dropped} censored/non-numeric Survival_days "
                f"row(s) that cannot be labeled for {BRATS_SURVIVAL_THRESHOLD_DAYS}-day survival"
            )
        labels = (valid["days"] > BRATS_SURVIVAL_THRESHOLD_DAYS).astype(int)
        return dict(zip(valid["Brats20ID"], labels))

    def _load_image_paths(self):
        """Gather t1 scans whose patient has a usable (non-censored) survival label."""
        paths = sorted(self.image_dir.rglob("*_t1.nii"))
        return [p for p in paths if p.parent.name in self.id_to_label]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_complete = nib.load(img_path).get_fdata()
        image = image_complete[:, :, BRATS_SURVIVAL_SLICE]
        denom = (image.max() - image.min()) or 1.0
        image = (image - image.min()) / denom
        image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


class BRATSSliceDataset(Dataset):
    def __init__(self, image_dir, slices_idx=BRATS_SLICE_INDICES, transform=None):
        """
        A PyTorch Dataset for loading selected slices of BRATS MRI scans and
        classifying them based on the presence of tumor tissue using segmentation data.

        Uses only the t1 modality and a fixed area-share threshold to define a positive
        (tumor-present) slice. Exposes ``self.groups`` (a per-slice scan id) so that
        cross-validation can split by patient and avoid slice-level leakage.

        Args:
            image_dir (str or Path): Directory where the BRATS MRI scans and segmentation masks are stored.
            slices_idx (list of int, optional): List of slice indices to include for training.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = Path(image_dir)
        self.slices_idx = slices_idx
        self.transform = transform

        # Pair each scan with its segmentation by patient directory (robust to ordering/counts),
        # rather than relying on positional zip of two independently-sorted file lists.
        scan_by_dir = {p.parent: p for p in self.image_dir.rglob("*_t1.nii")}
        seg_by_dir = {p.parent: p for p in self.image_dir.rglob("*_seg.nii")}
        self.scan_seg_pairs = [
            (scan_by_dir[d], seg_by_dir[d]) for d in sorted(scan_by_dir) if d in seg_by_dir
        ]

        # Generate slice-level labels
        self.slice_info = self._generate_slice_labels()
        self.labels = [[float(info[4])] for info in self.slice_info]
        # Per-slice scan/patient id so CV can group slices of the same scan together.
        self.groups = [info[5] for info in self.slice_info]
        self.num_classes = 1

    def _generate_slice_labels(self):
        """
        Generate labels for selected slices based on segmentation data.

        Returns:
            List[Tuple[Path, Path, int, float, int, int]]: per item
                - Path to the MRI scan file
                - Path to the segmentation mask file
                - Slice index
                - Tumor area share
                - Label (1 tumor present, 0 absent)
                - Scan/patient group id
        """
        slice_info = []

        for scan_idx, (img_path, seg_path) in enumerate(self.scan_seg_pairs):
            segmentation = nib.load(seg_path).get_fdata()
            num_slices = segmentation.shape[2]

            # If slices_idx is not provided, default to selecting middle slices
            selected_slices = (
                self.slices_idx
                if self.slices_idx
                else list(range(num_slices // 4, 3 * num_slices // 4))
            )

            selected_slices = [idx for idx in selected_slices if 0 <= idx < num_slices]
            for slice_idx in selected_slices:
                labeled_area_share = float(np.round(np.mean(segmentation[:, :, slice_idx] > 0), 8))
                slice_label = 1 if labeled_area_share > BRATS_TUMOR_AREA_THRESHOLD else 0
                slice_info.append(
                    (img_path, seg_path, slice_idx, labeled_area_share, slice_label, scan_idx)
                )

        return slice_info

    def _normalize_image(self, image):
        min_val = image.min()
        max_val = image.max()

        if max_val == min_val:
            return np.zeros_like(image)

        return (image - min_val) / (max_val - min_val)

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        img_path, _, slice_idx, _, _, _ = self.slice_info[idx]

        # Load the MRI scan and extract the specific slice
        full_scan = nib.load(img_path).get_fdata()
        slice_data = full_scan[:, :, slice_idx]

        # Convert the slice to an image
        slice_data_norm = self._normalize_image(slice_data)
        slice_img = Image.fromarray((slice_data_norm * 255).astype(np.uint8)).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            slice_img = self.transform(slice_img)

        # Convert the label to a tensor
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return slice_img, label


def resolve_concrete(dataset, idx):
    """Unwrap nested ``Subset``/``ConcatDataset`` wrappers to the concrete dataset and local index.

    Mirrors ``ConcatDataset.__getitem__``'s index arithmetic so callers can recover the
    underlying dataset (e.g. ``ImageDatasetCOCO``) and its per-sample metadata after a
    ``random_split`` or dataset concatenation.
    """
    while isinstance(dataset, (Subset, ConcatDataset)):
        if isinstance(dataset, Subset):
            idx = dataset.indices[idx]
            dataset = dataset.dataset
        else:  # ConcatDataset
            ds_idx = bisect_right(dataset.cumulative_sizes, idx)
            if ds_idx > 0:
                idx -= dataset.cumulative_sizes[ds_idx - 1]
            dataset = dataset.datasets[ds_idx]
    return dataset, idx


def collate_fn(inputs):
    """Custom collate function for batching."""
    images, labels = zip(*inputs)
    return {"pixel_values": torch.stack(images, dim=0), "labels": torch.stack(labels, dim=0)}


def create_mask(batch_size, image_size, patch_size, mask_ratio):
    """
    Generate a random mask for the image at the pixel level, based on the given patch size and mask ratio.
    """
    # Calculate the number of patches based on the image size and patch size
    num_patches_per_dim = image_size // patch_size
    num_patches = num_patches_per_dim**2  # Total number of patches in the image

    # Initialize the mask to all ones (unmasked)
    mask = torch.ones(batch_size, num_patches, dtype=torch.bool)

    # Calculate the number of patches to mask based on the mask ratio
    num_masked = int(mask_ratio * num_patches)

    # Randomly select patches to mask for each image in the batch
    for i in range(batch_size):
        mask[i, torch.randperm(num_patches)[:num_masked]] = False

    # Reshape the mask to match the patch grid dimensions (num_patches_per_dim x num_patches_per_dim)
    mask = mask.view(
        batch_size, num_patches_per_dim, num_patches_per_dim
    )  # [batch_size, h_patches, w_patches]

    # Expand the mask to have a channel dimension, simulating a 3-channel RGB mask
    mask = mask.unsqueeze(1).expand(-1, 3, -1, -1)  # [batch_size, 3, h_patches, w_patches]

    # Upsample the mask to match the pixel resolution of the image
    mask = nn.functional.interpolate(mask.float(), scale_factor=patch_size).bool()

    return mask


def get_masked_images(images, patch_size, mask_ratio=MASK_RATIO):
    """Randomly mask whole patches of a batch for masked-image-modeling.

    Returns ``(masked_images, mask)`` where ``mask`` is a boolean tensor that is True for
    KEPT (visible) pixels and False for masked ones. Masked pixels are filled with 0.0 —
    the post-ImageNet-normalization mean — so the fill matches the encoder's input space.
    Shared by train.py and eval_clf.py so the masking can never silently diverge.
    """
    batch_size, _, height, _ = images.shape
    mask = create_mask(batch_size, height, patch_size, mask_ratio).to(images.device)
    masked_images = images.clone()
    masked_images[~mask] = 0.0
    return masked_images, mask
