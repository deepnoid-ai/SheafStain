import os
import numpy as np

from PIL import Image
from skimage.filters import threshold_otsu


def get_tissue_ratio(img_array, use_otsu=True, background_threshold=200):
    """
    Calculate the tissue ratio in a histopathology image.

    Tissue is typically darker (stained), while background is lighter (white).

    Parameters:
        img_array (numpy.ndarray): RGB image array of shape (H, W, 3), values in [0, 255]
        use_otsu (bool): If True, use Otsu's method. If False, use fixed threshold.
        background_threshold (int): Fixed threshold for background if use_otsu=False.
                                   Pixels > threshold are considered background.

    Returns:
        float: Tissue ratio in range [0, 1], where 1 means 100% tissue.
    """
    # Convert RGB to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gray = img_array.astype(np.uint8)

    if use_otsu:
        # Otsu's method: automatically find optimal threshold
        try:
            threshold = threshold_otsu(gray)
        except ValueError:
            # If image is uniform, fall back to fixed threshold
            threshold = background_threshold
    else:
        threshold = background_threshold

    # Tissue pixels are darker than threshold
    tissue_mask = gray < threshold

    # Calculate ratio
    tissue_pixels = np.sum(tissue_mask)
    total_pixels = tissue_mask.size

    return tissue_pixels / total_pixels


def is_valid_tissue_patch(img_array, min_tissue_ratio=0.3, use_otsu=True):
    """
    Check if a patch contains sufficient tissue for training.

    Parameters:
        img_array (numpy.ndarray): RGB image array of shape (H, W, 3)
        min_tissue_ratio (float): Minimum required tissue ratio in [0, 1]
        use_otsu (bool): Whether to use Otsu's method

    Returns:
        bool: True if patch has sufficient tissue, False otherwise.
    """
    ratio = get_tissue_ratio(img_array, use_otsu=use_otsu)
    return ratio >= min_tissue_ratio


def get_tissue_statistics(img_array):
    """
    Get detailed tissue statistics for debugging/analysis.

    Parameters:
        img_array (numpy.ndarray): RGB image array

    Returns:
        dict: Statistics including:
            - tissue_ratio: Tissue ratio [0, 1]
            - threshold: Calculated threshold value
            - mean_intensity: Mean grayscale intensity
            - std_intensity: Standard deviation of intensity
    """
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gray = img_array.astype(np.uint8)

    # Calculate threshold
    try:
        threshold = threshold_otsu(gray)
    except ValueError:
        threshold = 200

    # Tissue mask
    tissue_mask = gray < threshold
    tissue_ratio = np.sum(tissue_mask) / tissue_mask.size

    # Statistics
    stats = {
        'tissue_ratio': tissue_ratio,
        'threshold': float(threshold),
        'mean_intensity': float(np.mean(gray)),
        'std_intensity': float(np.std(gray)),
        'min_intensity': int(np.min(gray)),
        'max_intensity': int(np.max(gray)),
    }

    return stats


# Standard thresholds based on pathology literature
TISSUE_RATIO_THRESHOLDS = {
    'strict': 0.8,      # 80% tissue - very strict
    'moderate': 0.5,    # 50% tissue - moderate
    'loose': 0.3,       # 30% tissue - permissive
}


if __name__ == "__main__":
    # Test the functions (image path)
    IMG_PATH="/Patch/Image/HnE/her2/train_or_test.png"

    # Test on a BCI image
    test_img_path = IMG_PATH

    if os.path.exists(test_img_path):
        img = Image.open(test_img_path).convert('RGB')
        img_array = np.array(img)

        print("Testing tissue filtering on BCI image:")
        print(f"  Image shape: {img_array.shape}")

        # Get statistics
        stats = get_tissue_statistics(img_array)
        print(f"\nTissue statistics:")
        for key, value in stats.items():
            if 'ratio' in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")

        # Test different thresholds
        print(f"\nValidation results:")
        for name, threshold in TISSUE_RATIO_THRESHOLDS.items():
            is_valid = is_valid_tissue_patch(img_array, min_tissue_ratio=threshold)
            print(f"  {name:10s} (>{threshold:.0%}): {'PASS' if is_valid else 'FAIL'}")
    else:
        print(f"Test image not found: {test_img_path}")
