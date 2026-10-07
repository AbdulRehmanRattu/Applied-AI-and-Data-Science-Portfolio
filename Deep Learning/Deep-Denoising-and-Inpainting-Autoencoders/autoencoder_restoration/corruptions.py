import numpy as np


def add_gaussian_noise(images: np.ndarray, noise_factor: float = 0.2, clip_range: tuple = (0.0, 1.0), seed: int = None) -> np.ndarray:
    """
    Applies additive zero-mean Gaussian noise to normalized image arrays.
    
    Formula:
        x_noisy = clip(x + noise_factor * N(0, 1), clip_min, clip_max)
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = images + noise_factor * noise
    return np.clip(noisy_images, clip_range[0], clip_range[1]).astype(np.float32)


def add_spatial_occlusions(images: np.ndarray, occlusion_size: tuple = (8, 8), mask_value: float = 0.0, seed: int = None) -> np.ndarray:
    """
    Applies stochastic rectangular spatial occlusion (masking/dropout) to simulate sensor occlusion.
    
    Formula:
        x_occluded[y : y + h, x : x + w, :] = mask_value
    """
    if seed is not None:
        np.random.seed(seed)
        
    occluded_images = images.copy()
    img_height, img_width = images.shape[1:3]
    occ_h, occ_w = occlusion_size

    for img in occluded_images:
        top_x = np.random.randint(0, max(1, img_width - occ_w + 1))
        top_y = np.random.randint(0, max(1, img_height - occ_h + 1))
        img[top_y : top_y + occ_h, top_x : top_x + occ_w, :] = mask_value

    return occluded_images.astype(np.float32)


def add_combined_corruption(images: np.ndarray, noise_factor: float = 0.2, occlusion_size: tuple = (8, 8), seed: int = None) -> np.ndarray:
    """
    Applies simultaneous Gaussian noise and spatial occlusion to simulate harsh multi-corruption environments.
    """
    if seed is not None:
        np.random.seed(seed)
        
    noisy = add_gaussian_noise(images, noise_factor=noise_factor, seed=seed)
    combined = add_spatial_occlusions(noisy, occlusion_size=occlusion_size, seed=seed)
    return combined.astype(np.float32)
