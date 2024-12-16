import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f'PROJ_ROOT path is: {PROJ_ROOT}')

DATA_DIR = PROJ_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

TRACKING_URI = PROJ_ROOT / 'mlruns'

SYNTHETIC_DATA_DIR = RAW_DATA_DIR / 'synthetic'
SYNTHETIC_TRAIN_DATA_DIR = SYNTHETIC_DATA_DIR / 'train'
SYNTHETIC_TEST_DATA_DIR = SYNTHETIC_DATA_DIR / 'test'
SYNTHETIC_LABELS_NAME = 'shape_flags.parquet'
SYNTHETIC_TRAIN_LABELS_PATH = SYNTHETIC_TRAIN_DATA_DIR / SYNTHETIC_LABELS_NAME
SYNTHETIC_TEST_LABELS_PATH = SYNTHETIC_TEST_DATA_DIR / SYNTHETIC_LABELS_NAME

SEGMENTED_DATA_DIR = EXTERNAL_DATA_DIR / 'opg111.v1i.coco-segmentation'
SEGMENTED_TRAIN_DATA_DIR = SEGMENTED_DATA_DIR / 'train'
SEGMENTED_VAL_DATA_DIR = SEGMENTED_DATA_DIR / 'valid'
SEGMENTED_TEST_DATA_DIR = SEGMENTED_DATA_DIR / 'test'
SEGMENTED_ANNOTATIONS_NAME = '_annotations.coco.json'
SEGMENTED_TRAIN_ANNOTATIONS_PATH = SEGMENTED_TRAIN_DATA_DIR / SEGMENTED_ANNOTATIONS_NAME
SEGMENTED_VAL_ANNOTATIONS_PATH = SEGMENTED_VAL_DATA_DIR / SEGMENTED_ANNOTATIONS_NAME
SEGMENTED_TEST_ANNOTATIONS_PATH = SEGMENTED_TEST_DATA_DIR / SEGMENTED_ANNOTATIONS_NAME

ULTRASOUND_DATA_DIR = EXTERNAL_DATA_DIR / 'ultrasound'

MODELS_DIR = PROJ_ROOT / 'models'

REPORTS_DIR = PROJ_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

RANDOM_STATE = 214

BASE_MODEL_NAME = 'facebook/dino-vits8'
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
MMD_WEIGHT = 0.01
OOD_CATEGORIES = ['RESTORATION', 'CROWN', 'CROWN AND BRIDGE', 'IMPLANT']
ALTERNATIVE_OOD_CATEGORIES = [
    'ROOT CANAL TREATED TOOTH',
    'ROOT STUMP',
    'SHORTENED RCT',
    'ROOT CANAL BEYOND APEX',
    'ERUPTING TOOTH',
    'BROKEN TOOTH'
]
TARGET_CATEGORIES = ['CARIES']

NUM_EPOCHS_MIM = 30
MASK_RATIO = 0.25

PATIENCE = 5

# ULTRASOUND_LABELS_MAPPING = {
#     'НОРМА': 0,
#     'МИЕЛОПАТИЯ': 1,
#     'МИОПАТИЯ': 2,
#     'ПОЛИНЕВРОПАТИЯ': 3,
# }
ULTRASOUND_LABELS_MAPPING = {
    'НОРМА': 0,
    'ПАТОЛОГИЯ': 1,
}
MIM_WEIGHT = 0.1

RUN_ID_MIM_REAL_L1 = os.getenv('RUN_ID_MIM_REAL_L1')
RUN_ID_MIM_REAL_MSE = os.getenv('RUN_ID_MIM_REAL_MSE')
RUN_ID_MIM_REAL_SMOOTH_L1 = os.getenv('RUN_ID_MIM_REAL_SMOOTH_L1')
RUN_ID_MIM_REAL_HUBER = os.getenv('RUN_ID_MIM_REAL_HUBER')
RUN_ID_MIM_REAL_TUKEY = os.getenv('RUN_ID_MIM_REAL_TUKEY')
RUN_ID_MIM_REAL_SMOOTH_L1_FILTERED = os.getenv('RUN_ID_MIM_REAL_SMOOTH_L1_FILTERED')

RUN_ID_CLASS_REAL_MSE = os.getenv('RUN_ID_CLASS_REAL_MSE')
RUN_ID_CLASS_REAL_L1 = os.getenv('RUN_ID_CLASS_REAL_L1')
RUN_ID_CLASS_REAL_SMOOTH_L1 = os.getenv('RUN_ID_CLASS_REAL_SMOOTH_L1')
RUN_ID_CLASS_REAL_HUBER = os.getenv('RUN_ID_CLASS_REAL_HUBER')
RUN_ID_CLASS_REAL_TUKEY = os.getenv('RUN_ID_CLASS_REAL_TUKEY')

RUN_ID_REAL_MULTITASK = os.getenv('RUN_ID_REAL_MULTITASK')

RUN_ID_MIM_ULTRASOUND_L1 = os.getenv('RUN_ID_MIM_ULTRASOUND_L1')
RUN_ID_MIM_ULTRASOUND_MSE = os.getenv('RUN_ID_MIM_ULTRASOUND_MSE')
RUN_ID_MIM_ULTRASOUND_SMOOTH_L1 = os.getenv('RUN_ID_MIM_ULTRASOUND_SMOOTH_L1')
RUN_ID_MIM_ULTRASOUND_HUBER = os.getenv('RUN_ID_MIM_ULTRASOUND_HUBER')
RUN_ID_MIM_ULTRASOUND_TUKEY = os.getenv('RUN_ID_MIM_ULTRASOUND_TUKEY')

RUN_ID_CLASS_ULTRASOUND_L1 = os.getenv('RUN_ID_CLASS_ULTRASOUND_L1')
RUN_ID_CLASS_ULTRASOUND_MSE = os.getenv('RUN_ID_CLASS_ULTRASOUND_MSE')
RUN_ID_CLASS_ULTRASOUND_SMOOTH_L1 = os.getenv('RUN_ID_CLASS_ULTRASOUND_SMOOTH_L1')
RUN_ID_CLASS_ULTRASOUND_HUBER = os.getenv('RUN_ID_CLASS_ULTRASOUND_HUBER')
RUN_ID_CLASS_ULTRASOUND_TUKEY = os.getenv('RUN_ID_CLASS_ULTRASOUND_TUKEY')
RUN_ID_CLASS_ULTRASOUND = os.getenv('RUN_ID_CLASS_ULTRASOUND')

RUN_ID_ULTRASOUND_MULTITASK = os.getenv('RUN_ID_ULTRASOUND_MULTITASK')

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)
except ModuleNotFoundError:
    pass


# import torch
#
# def gaussian_kernel(x, y, sigma=1.0):
#     # Compute the Gaussian RBF kernel
#     dist = torch.cdist(x, y, p=2)
#     return torch.exp(-dist ** 2 / (2 * sigma ** 2))
#
# def mmd_loss(source, target, sigma=1.0):
#     # Calculate Maximum Mean Discrepancy (MMD)
#     source_kernel = gaussian_kernel(source, source, sigma)
#     target_kernel = gaussian_kernel(target, target, sigma)
#     cross_kernel = gaussian_kernel(source, target, sigma)
#     mmd = source_kernel.mean() + target_kernel.mean() - 2 * cross_kernel.mean()
#     return mmd
#
# class CauchyLoss(nn.Module):
#     def __init__(self, c=1.0):
#         super(CauchyLoss, self).__init__()
#         self.c = c
#
#     def forward(self, predictions, targets):
#         residual = predictions - targets
#         loss = torch.log(1 + (residual ** 2) / self.c ** 2)
#         return loss.mean()
#

#
#
# class HuberLoss(nn.Module):
#     def __init__(self, delta=1.0):
#         super(HuberLoss, self).__init__()
#         self.delta = delta
#
#     def forward(self, predictions, targets):
#         residual = torch.abs(predictions - targets)
#         loss = torch.where(residual < self.delta, 0.5 * residual**2, self.delta * (residual - 0.5 * self.delta))
#         return loss.mean()
#
#
# def entropy_loss(logits):
#     # Convert logits to probabilities
#     probs = torch.softmax(logits, dim=-1)
#
#     # Compute entropy
#     entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
#
#     return torch.mean(entropy)
#
#
# class ConfidencePenaltyLoss(nn.Module):
#     def __init__(self, penalty_weight=0.1):
#         super(ConfidencePenaltyLoss, self).__init__()
#         self.penalty_weight = penalty_weight
#
#     def forward(self, inputs, targets):
#         # Standard binary cross-entropy loss
#         ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets)
#
#         # Apply confidence penalty (entropy regularization)
#         probs = torch.sigmoid(inputs)
#         penalty = self.penalty_weight * torch.mean(probs * torch.log(probs + 1e-8))  # Avoid log(0)
#
#         # Combine losses
#         return ce_loss + penalty
#
#
# def outlier_exposure_loss(predictions, labels, ood_predictions, ood_weight=0.5):
#     # Binary cross entropy loss for in-domain data
#     id_loss = nn.functional.binary_cross_entropy_with_logits(predictions, labels)
#
#     # Confidence penalty for out-of-domain data (encourages low confidence)
#     ood_loss = torch.mean(ood_predictions ** 2)  # Penalize confident OOD predictions
#
#     # Combine both losses
#     total_loss = id_loss + ood_weight * ood_loss
#     return total_loss
#
#
# import torch
# import torch.nn as nn
#
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)  # Prevents nan error in gradient calculation
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return F_loss.mean()
#
#
# def entropy_regularization(predictions):
#     # Apply softmax to get probabilities
#     probs = torch.softmax(predictions, dim=-1)
#     # Compute entropy
#     entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
#     return entropy.mean()
#
# def total_loss(regression_loss, predictions, lambda_reg=0.1):
#     return regression_loss + lambda_reg * entropy_regularization(predictions)
#
#
# import torch
#
