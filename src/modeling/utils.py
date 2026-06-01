import os

import numpy as np
import torch
from scipy import stats
from torch import nn
from torchvision import transforms
from transformers import AutoModel

from src.config import (IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
                        RUN_ID_BRATS_MIM_MSE, RUN_ID_CLASS_REAL_MSE,
                        RUN_ID_CLASS_ULTRASOUND, RUN_ID_MIM_REAL_HUBER,
                        RUN_ID_MIM_REAL_L1, RUN_ID_MIM_REAL_MSE,
                        RUN_ID_MIM_REAL_SMOOTH_L1,
                        RUN_ID_MIM_REAL_SMOOTH_L1_FILTERED,
                        RUN_ID_MIM_REAL_TUKEY, RUN_ID_MIM_ULTRASOUND_MSE,
                        RUN_ID_REAL_MULTITASK, RUN_ID_ULTRASOUND_MULTITASK)

# Auto-detect the compute device, with an optional DEVICE env override (e.g. DEVICE=cpu).
DEVICE = os.environ.get('DEVICE') or (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# Normalize to the ImageNet statistics the DINO backbone was pretrained on.
NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
# Standard 0.875 center-crop ratio (e.g. 256 -> 224); train and eval share this FOV.
_RESIZE = int(round(IMAGE_SIZE / 0.875))

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # vertical flip dropped: anatomically invalid
    transforms.ToTensor(),
    NORMALIZE,
])

# Deterministic variant with the same field-of-view as EVAL_TRANSFORM.
TRAIN_TRANSFORM_SIMPLIFIED = transforms.Compose([
    transforms.Resize(size=_RESIZE),
    transforms.CenterCrop(size=IMAGE_SIZE),
    transforms.ToTensor(),
    NORMALIZE,
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(size=_RESIZE),
    transforms.CenterCrop(size=IMAGE_SIZE),
    transforms.ToTensor(),
    NORMALIZE,
])


def denormalize(tensor):
    """Invert ImageNet normalization for visualization; returns a [0,1]-clamped tensor."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


def confidence_interval(values, confidence=0.95):
    """Two-sided t-based confidence interval for the mean of ``values``.

    Returns ``(mean, lower, upper)``; for fewer than two values returns ``(mean, mean, mean)``.
    Valid only when the values are (approximately) independent — i.e. across disjoint
    cross-validation folds, not overlapping random subsamples.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    se = float(np.std(values, ddof=1)) / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    margin = t_crit * se
    return mean, mean - margin, mean + margin


def load_pretrained_model(base_model_name, state_dict_path):
    """
    Load a pre-trained model from Hugging Face transformers library.
    """
    model = AutoModel.from_pretrained(base_model_name, add_pooling_layer=False, attn_implementation='eager')
    model.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))
    return model


def get_model_patch_info(model, image_size=IMAGE_SIZE):
    """
    Get patch size and number of patches for the model.
    """
    patch_size = model.config.patch_size
    n_patches = int(image_size ** 2 / patch_size ** 2)
    n_patches_per_dim = int(n_patches ** 0.5)
    return patch_size, n_patches, n_patches_per_dim


def get_regression_loss_function(loss_type):
    """
    Return the loss function based on the specified loss_type.
    """
    if loss_type == 'L1':
        return nn.L1Loss()
    elif loss_type == 'MSE':
        return nn.MSELoss()
    elif loss_type == 'SmoothL1':
        return nn.SmoothL1Loss()
    elif loss_type == 'Huber':
        return nn.HuberLoss(delta=1.0)
    elif loss_type == 'Cauchy':
        class CauchyLoss(nn.Module):
            def __init__(self, c=1.0):
                super(CauchyLoss, self).__init__()
                self.c = c

            def forward(self, predictions, targets):
                residual = predictions - targets
                loss = torch.log(1 + (residual ** 2) / self.c ** 2)
                return loss.mean()

        return CauchyLoss()
    elif loss_type == 'Quantile':
        class QuantileLoss(nn.Module):
            def __init__(self, quantile=0.5):
                super(QuantileLoss, self).__init__()
                self.quantile = quantile

            def forward(self, predictions, targets):
                errors = targets - predictions
                loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)
                return loss.mean()

        return QuantileLoss()
    elif loss_type == 'Tukey':
        class TukeyLoss(nn.Module):
            def __init__(self, c=4.685):
                super(TukeyLoss, self).__init__()
                self.c = c

            def forward(self, predictions, targets):
                residual = torch.abs(predictions - targets)
                squared_residual = (residual / self.c) ** 2
                loss = torch.where(
                    residual <= self.c,
                    (self.c ** 2 / 6) * (1 - (1 - squared_residual) ** 3),
                    (self.c ** 2 / 6)
                )
                return loss.mean()

        return TukeyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_classification_loss_function(loss_type, pos_weight=None):
    """
    Return the classification loss function based on the specified loss_type.

    ``pos_weight`` (a tensor) up-weights the positive class for BCEWithLogits to counter
    class imbalance; it is ignored by the other loss variants.
    """
    if loss_type == 'BCEWithLogits':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Default for binary/multilabel
    elif loss_type == 'Focal':
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1.0, gamma=2.0):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
                pt = torch.exp(-BCE_loss)
                F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
                return F_loss.mean()

        return FocalLoss()
    elif loss_type == 'LabelSmoothing':
        class LabelSmoothingLoss(nn.Module):
            def __init__(self, smoothing=0.1):
                super(LabelSmoothingLoss, self).__init__()
                self.smoothing = smoothing

            def forward(self, inputs, targets):
                # Apply label smoothing to binary targets
                targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
                return nn.functional.binary_cross_entropy_with_logits(inputs, targets)

        return LabelSmoothingLoss()
    elif loss_type == 'ConfidencePenalty':
        class ConfidencePenaltyLoss(nn.Module):
            def __init__(self, penalty_weight=0.1):
                super(ConfidencePenaltyLoss, self).__init__()
                self.penalty_weight = penalty_weight

            def forward(self, inputs, targets):
                # Standard binary cross-entropy loss
                ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets)

                # Apply confidence penalty (entropy regularization)
                probs = torch.sigmoid(inputs)
                penalty = self.penalty_weight * torch.mean(probs * torch.log(probs + 1e-8))  # Avoid log(0)

                # Combine losses
                return ce_loss + penalty

        return ConfidencePenaltyLoss()
    elif loss_type == 'QBCE':
        class QBCEWithLogitsLoss(nn.Module):
            def __init__(self, q=1.5, reduction='mean'):
                """
                Initializes the Q-Binary Cross-Entropy Loss.

                Args:
                    q (float): The q parameter in the q-logarithm. q=1 recovers standard BCE.
                    reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                """
                super(QBCEWithLogitsLoss, self).__init__()
                self.q = q
                self.reduction = reduction

            def forward(self, logits, targets):
                """
                Computes the Q-Binary Cross-Entropy Loss.

                Args:
                    logits (Tensor): Predicted logits of shape (N, *).
                    targets (Tensor): Ground truth binary labels of shape (N, *).

                Returns:
                    Tensor: The computed Q-BCE loss.
                """
                # Sigmoid activation to convert logits to probabilities.
                # Clamp away from 0/1 so the q-logarithm (and the q==1 log branch)
                # stay finite for saturated sigmoid outputs.
                eps = 1e-6
                probs = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)

                # Compute the q-logarithm
                if self.q == 1.0:
                    log_q = torch.log(probs)
                    log1m_q = torch.log(1 - probs)
                else:
                    log_q = (probs ** (1 - self.q) - 1) / (1 - self.q)
                    log1m_q = ((1 - probs) ** (1 - self.q) - 1) / (1 - self.q)

                # Q-BCE loss calculation
                loss = - (targets * log_q + (1 - targets) * log1m_q)

                # Apply reduction
                if self.reduction == 'mean':
                    return loss.mean()
                elif self.reduction == 'sum':
                    return loss.sum()
                else:  # 'none'
                    return loss

        return QBCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class MultiTaskLoss(nn.Module):
    def __init__(self, regression_loss, classification_loss, mim_weight):
        super(MultiTaskLoss, self).__init__()
        self.regression_loss = regression_loss
        self.classification_loss = classification_loss
        self.mim_weight = mim_weight

    def forward(self, mim_output, class_output, images, labels, mask):
        # Reconstruction loss only on masked pixels (mask is True for KEPT/visible pixels),
        # so the model cannot trivially copy visible patches.
        masked = ~mask
        mim_loss = self.regression_loss(mim_output[masked], images[masked])
        class_loss = self.classification_loss(class_output, labels.float())
        return class_loss + self.mim_weight * mim_loss


def compute_mmd(x, y):
    """Gaussian (RBF) kernel Maximum Mean Discrepancy between two batches.

    Each sample is flattened to a feature vector and an RBF kernel with a
    median-heuristic bandwidth is used, so the estimator is a proper (non-negative)
    MMD: ``E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]``.
    """
    x = x.flatten(1)
    y = y.flatten(1)

    def rbf(a, b):
        d2 = torch.cdist(a, b) ** 2
        sigma2 = d2.detach().median().clamp(min=1e-8)
        return torch.exp(-d2 / sigma2)

    return rbf(x, x).mean() + rbf(y, y).mean() - 2 * rbf(x, y).mean()


def get_model_run_id(model_type):
    if model_type == 'L1':
        return RUN_ID_MIM_REAL_L1
    elif model_type == 'MSE':
        return RUN_ID_MIM_REAL_MSE
    elif model_type == 'SMOOTH_L1':
        return RUN_ID_MIM_REAL_SMOOTH_L1
    elif model_type == 'HUBER':
        return RUN_ID_MIM_REAL_HUBER
    elif model_type == 'TUKEY':
        return RUN_ID_MIM_REAL_TUKEY
    elif model_type == 'SMOOTH_L1_FILTERED':
        return RUN_ID_MIM_REAL_SMOOTH_L1_FILTERED
    elif model_type == 'ULTRASOUND':
        return RUN_ID_MIM_ULTRASOUND_MSE
    elif model_type == 'ULTRASOUND_CLASS':
        return RUN_ID_CLASS_ULTRASOUND
    elif model_type == 'ULTRASOUND_MULTITASK':
        return RUN_ID_ULTRASOUND_MULTITASK
    elif model_type == 'REAL_MULTITASK':
        return RUN_ID_REAL_MULTITASK
    elif model_type == 'REAL_CLASS_MSE':
        return RUN_ID_CLASS_REAL_MSE
    elif model_type == 'BRATS_MIM_MSE':
        return RUN_ID_BRATS_MIM_MSE
    else:
        raise ValueError(f'Unknown model type: {model_type}')
