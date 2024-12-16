import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torch.nn import functional as F
from torchvision import transforms

from src.config import FIGURES_DIR, RAW_DATA_DIR


class DINOv2Segmentation:
    def __init__(self, model_name='dinov2_vits14'):
        """
        Initialize the DINOv2 segmentation class.

        Args:
            model_name (str): Name of the DINOv2 model to load.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_name)
        self.features = None
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self._register_hook()

    def _load_model(self, model_name):
        """
        Load the DINOv2 model from the PyTorch Hub.

        Args:
            model_name (str): Name of the DINOv2 model to load.

        Returns:
            torch.nn.Module: Loaded DINOv2 model.
        """
        model = torch.hub.load('facebookresearch/dinov2', model_name, source='github').to(self.device)
        model.eval()
        return model

    def _register_hook(self):
        """
        Register a forward hook to capture intermediate outputs from the model.
        """
        def hook_fn(module, input, output):
            self.features = output

        # Hook into the attention layer of the last block
        for name, module in self.model.named_modules():
            if name.endswith('blocks.11.attn'):
                module.register_forward_hook(hook_fn)
                break
        else:
            raise RuntimeError('Attention layer "blocks.11.attn" not found in the model.')

    def _extract_features(self, image: Image.Image):
        """
        Extract spatial features from the input image using the DINOv2 model.

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            np.ndarray: Extracted spatial feature map.
        """
        # Preprocess and move image to device
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Perform a forward pass to extract features
        with torch.no_grad():
            _ = self.model(image_tensor)

        if self.features is None:
            raise RuntimeError('Features were not captured. Ensure the hook is correctly registered.')

        features = self.features.squeeze(0)  # Remove batch dimension

        # Check for and remove the class token
        if features.size(0) == 257:  # 256 patches + 1 class token for ViT-S
            features = features[1:]  # Remove the first token (class token)

        # Ensure valid square number of patches
        num_patches = int(features.size(0) ** 0.5)
        if num_patches ** 2 != features.size(0):
            raise ValueError(f'Invalid number of patches: {features.size(0)}. Expected a square number.')

        # Reshape to (C, H, W) and interpolate to image size
        features = features.view(num_patches, num_patches, -1).permute(2, 0, 1)
        features = F.interpolate(features.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        return features.squeeze(0).cpu().numpy()

    def segment_image(self, image_path: str, num_clusters: int = 5):
        """
        Segment an image into clusters using k-means on the extracted features.

        Args:
            image_path (str): Path to the input image.
            num_clusters (int): Number of clusters for k-means.

        Returns:
            np.ndarray: Segmentation labels resized to the original image size.
        """
        image = Image.open(image_path).convert('RGB')
        features = self._extract_features(image)

        # Flatten features for clustering
        h, w = features.shape[1:]
        features_flat = features.reshape(features.shape[0], h * w).T

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(features_flat)
        labels = kmeans.labels_.reshape(h, w)

        # Resize labels to match original image dimensions
        original_size = image.size
        labels_resized = cv2.resize(labels.astype(np.float32), original_size, interpolation=cv2.INTER_NEAREST)
        return labels_resized.astype(int)

    def visualize_segmentation(self, image_path: str, labels: np.ndarray):
        """
        Visualize the segmentation by coloring each cluster.

        Args:
            image_path (str): Path to the original input image.
            labels (np.ndarray): Segmentation labels.

        Returns:
            np.ndarray: RGB image with segmentation visualization.
        """
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'Image not found at {image_path}.')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate random colors for each cluster
        unique_labels = np.unique(labels)
        colors = {label: np.random.randint(0, 255, 3).tolist() for label in unique_labels}

        # Map each label to its corresponding color
        segmented_image = np.zeros_like(image)
        for label, color in colors.items():
            segmented_image[labels == label] = color

        return segmented_image


def main():
    segmenter = DINOv2Segmentation()
    image_path = RAW_DATA_DIR / 'synthetic' / 'train' / 'train_image_1.png'
    segmented_image_path = FIGURES_DIR / 'segmented_image.png'

    try:
        labels = segmenter.segment_image(image_path, num_clusters=5)
        segmented_image = segmenter.visualize_segmentation(image_path, labels)
        Image.fromarray(segmented_image).save(segmented_image_path)
        print(f'Segmentation completed and saved as {segmented_image_path}.')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
