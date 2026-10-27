"""Appearance feature extraction modules for SmartTrack-AI.

Provides color histogram spatial pyramid extraction (baseline) and deep CNN
feature extraction (enhanced) for appearance-based track association.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

logger = logging.getLogger("smarttrack")

# Conditional PyTorch import for DeepAppearanceExtractor
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class AppearanceExtractor:
    """Baseline appearance feature extractor using HSV color histograms and spatial pyramids."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the appearance extractor.

        Args:
            config: Optional configuration dictionary.
        """
        self.config: Dict[str, Any] = config or {}
        appearance_cfg = self.config.get("appearance", {})
        self.feature_dim: int = int(appearance_cfg.get("feature_dim", 512))
        self.crop_size: tuple[int, int] = (128, 256)  # (width, height)

    def extract(
        self,
        image: np.ndarray,
        bbox: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Extract appearance feature vector from an image bounding box crop.

        Computes multi-channel HSV histograms across a 2x2 spatial pyramid grid
        (H: 8 bins, S: 8 bins, V: 4 bins = 20 bins per cell x 4 cells = 80 dims),
        then L2 normalizes and projects/pads to feature_dim.

        Args:
            image: Full frame image as a BGR numpy array [H, W, 3].
            bbox: Bounding box [x1, y1, x2, y2].

        Returns:
            Normalized feature vector of shape (feature_dim,).
        """
        if image is None or image.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        img_h, img_w = image.shape[:2]
        x1 = int(np.clip(round(float(bbox[0])), 0, img_w - 1))
        y1 = int(np.clip(round(float(bbox[1])), 0, img_h - 1))
        x2 = int(np.clip(round(float(bbox[2])), x1 + 1, img_w))
        y2 = int(np.clip(round(float(bbox[3])), y1 + 1, img_h))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Standardize crop resolution to (width=128, height=256)
        crop_resized = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

        # 2x2 Spatial Pyramid Partitioning
        h_mid = self.crop_size[1] // 2  # 128
        w_mid = self.crop_size[0] // 2  # 64

        cells = [
            hsv[0:h_mid, 0:w_mid],          # Top-left
            hsv[0:h_mid, w_mid:],           # Top-right
            hsv[h_mid:, 0:w_mid],           # Bottom-left
            hsv[h_mid:, w_mid:],           # Bottom-right
        ]

        cell_features: List[np.ndarray] = []
        for cell in cells:
            # H: 8 bins [0, 180], S: 8 bins [0, 256], V: 4 bins [0, 256] -> 20 bins
            hist_h = cv2.calcHist([cell], [0], None, [8], [0, 180]).flatten()
            hist_s = cv2.calcHist([cell], [1], None, [8], [0, 256]).flatten()
            hist_v = cv2.calcHist([cell], [2], None, [4], [0, 256]).flatten()

            cell_feat = np.concatenate([hist_h, hist_s, hist_v], axis=0)
            cell_features.append(cell_feat)

        # Concatenate 4 cells = 80 dims
        raw_feature = np.concatenate(cell_features, axis=0).astype(np.float32)

        # L2 normalization
        norm = np.linalg.norm(raw_feature)
        if norm > 1e-7:
            raw_feature /= norm

        # Pad or project to feature_dim
        if self.feature_dim == len(raw_feature):
            return raw_feature
        elif self.feature_dim > len(raw_feature):
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[:len(raw_feature)] = raw_feature
            return padded
        else:
            return raw_feature[:self.feature_dim]

    def compute_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine distance between two feature vectors.

        Args:
            feat1: First feature vector.
            feat2: Second feature vector.

        Returns:
            Cosine distance in range [0.0, 2.0].
        """
        f1 = np.asarray(feat1, dtype=np.float32).ravel()
        f2 = np.asarray(feat2, dtype=np.float32).ravel()

        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        if norm1 < 1e-7 or norm2 < 1e-7:
            return 1.0

        sim = float(np.dot(f1, f2) / (norm1 * norm2))
        dist = 1.0 - sim
        return float(np.clip(dist, 0.0, 2.0))

    def compute_distance_matrix(
        self,
        features1: Sequence[np.ndarray],
        features2: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Compute pairwise cosine distance matrix between two collections of features.

        Args:
            features1: Sequence of N feature vectors.
            features2: Sequence of M feature vectors.

        Returns:
            Numpy array of shape [N, M] with pairwise distances.
        """
        n = len(features1)
        m = len(features2)
        if n == 0 or m == 0:
            return np.zeros((n, m), dtype=np.float32)

        f1 = np.asarray(features1, dtype=np.float32)
        f2 = np.asarray(features2, dtype=np.float32)

        norm1 = np.linalg.norm(f1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(f2, axis=1, keepdims=True)

        f1_norm = np.where(norm1 > 1e-7, f1 / np.maximum(norm1, 1e-7), 0.0)
        f2_norm = np.where(norm2 > 1e-7, f2 / np.maximum(norm2, 1e-7), 0.0)

        similarity_matrix = np.dot(f1_norm, f2_norm.T)
        distance_matrix = 1.0 - similarity_matrix
        return np.clip(distance_matrix, 0.0, 2.0).astype(np.float32)


class DeepAppearanceExtractor(AppearanceExtractor):
    """Deep CNN feature extractor using MobileNetV2 backbone.

    Note:
        This model utilizes a general-purpose ImageNet-pretrained CNN backbone as a feature
        extractor, not a dedicated ReID model. While effective for appearance consistency
        across brief occlusions, dedicated ReID models trained on metric learning/triplet loss
        are recommended for long-term re-identification across camera viewpoints.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the deep appearance extractor.

        Falls back to baseline AppearanceExtractor if PyTorch/torchvision are unavailable.

        Args:
            config: Optional configuration dictionary.
        """
        super().__init__(config=config)
        self.fallback_mode: bool = False
        self.device = "cpu"
        self.model: Optional[Any] = None

        if not _TORCH_AVAILABLE:
            warnings.warn(
                "PyTorch / torchvision not installed. Falling back to color histogram AppearanceExtractor.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.fallback_mode = True
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # Instantiate MobileNetV2 backbone
            try:
                weights = models.MobileNet_V2_Weights.DEFAULT
                base_model = models.mobilenet_v2(weights=weights)
            except AttributeError:
                # Compatibility for older torchvision versions
                base_model = models.mobilenet_v2(pretrained=True)

            # Strip original classifier head and project to target feature_dim
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.projector = nn.Linear(base_model.last_channel, self.feature_dim, bias=False)

            self.model = nn.Sequential(
                self.features,
                self.pool,
                nn.Flatten(),
                self.projector,
            )
            self.model.to(self.device)
            self.model.eval()

            # Normalization constants (ImageNet standard)
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

            logger.info(
                f"DeepAppearanceExtractor initialized with MobileNetV2 on {self.device} (dim={self.feature_dim})."
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to initialize deep model ({exc}). Falling back to baseline AppearanceExtractor.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.fallback_mode = True

    def extract(
        self,
        image: np.ndarray,
        bbox: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Extract deep feature vector from image crop.

        Args:
            image: Full frame image in BGR format.
            bbox: Bounding box [x1, y1, x2, y2].

        Returns:
            L2-normalized feature vector of shape (feature_dim,).
        """
        if self.fallback_mode or self.model is None or not _TORCH_AVAILABLE:
            return super().extract(image, bbox)

        if image is None or image.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        img_h, img_w = image.shape[:2]
        x1 = int(np.clip(round(float(bbox[0])), 0, img_w - 1))
        y1 = int(np.clip(round(float(bbox[1])), 0, img_h - 1))
        x2 = int(np.clip(round(float(bbox[2])), x1 + 1, img_w))
        y2 = int(np.clip(round(float(bbox[3])), y1 + 1, img_h))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return np.zeros(self.feature_dim, dtype=np.float32)

        try:
            # Resize and convert BGR -> RGB
            resized = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Convert to float tensor [1, 3, H, W] in [0, 1]
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            tensor = (tensor - self.mean) / self.std

            with torch.no_grad():
                feat = self.model(tensor)
                feat = feat.squeeze(0).cpu().numpy()

            # L2 normalize
            norm = np.linalg.norm(feat)
            if norm > 1e-7:
                feat /= norm

            return feat.astype(np.float32)
        except Exception as exc:
            logger.debug(f"Deep extraction failed, using fallback: {exc}")
            return super().extract(image, bbox)
