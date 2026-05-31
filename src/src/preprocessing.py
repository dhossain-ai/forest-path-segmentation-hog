import cv2
import numpy as np


def load_image(image_path: str, max_width: int = 600):
    """
    Load image from disk. Resize if wider than max_width (preserves aspect ratio).
    Returns: img_bgr (color), img_gray (grayscale)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img_bgr.shape[:2]
    if w > max_width:
        scale   = max_width / w
        new_w   = max_width
        new_h   = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h))

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_gray


def extract_patches(image: np.ndarray, patch_size: int = 16, stride: int = 16):
    """
    Sliding-window patch extractor.

    Args:
        image:      Grayscale image (H x W)
        patch_size: Side length of each square patch in pixels
        stride:     Step between patches (= patch_size → no overlap)

    Returns:
        patches:   List of (patch_size x patch_size) arrays
        positions: List of (row, col) top-left corner for each patch
    """
    patches   = []
    positions = []
    h, w      = image.shape[:2]

    for row in range(0, h - patch_size + 1, stride):
        for col in range(0, w - patch_size + 1, stride):
            patch = image[row : row + patch_size, col : col + patch_size]
            patches.append(patch)
            positions.append((row, col))

    return patches, positions


def get_grid_shape(image: np.ndarray, patch_size: int = 16, stride: int = 16):
    """
    Returns (n_rows, n_cols): how many patches fit along each axis.
    """
    h, w   = image.shape[:2]
    n_rows = (h - patch_size) // stride + 1
    n_cols = (w - patch_size) // stride + 1
    return n_rows, n_cols