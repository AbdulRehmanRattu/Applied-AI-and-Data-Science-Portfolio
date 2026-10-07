import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes sample-averaged Mean Squared Error (MSE).
    """
    return float(np.mean((y_true - y_pred) ** 2))


def compute_psnr(y_true: np.ndarray, y_pred: np.ndarray, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Formula:
        PSNR = 10 * log10(max_val^2 / MSE)
    """
    mse = compute_mse(y_true, y_pred)
    if mse <= 1e-10:
        return 100.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_ssim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes average Structural Similarity Index Measure (SSIM) across batch.
    """
    scores = []
    for i in range(len(y_true)):
        # Support both older and modern versions of skimage
        try:
            val = ssim_fn(y_true[i], y_pred[i], channel_axis=-1, data_range=1.0)
        except TypeError:
            val = ssim_fn(y_true[i], y_pred[i], multichannel=True, data_range=1.0)
        scores.append(val)
    return float(np.mean(scores))


def evaluate_restoration_batch(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluates restoration performance across MSE, PSNR, and SSIM.
    """
    mse_val = compute_mse(y_true, y_pred)
    psnr_val = compute_psnr(y_true, y_pred)
    ssim_val = compute_ssim(y_true, y_pred)
    
    return {
        "mse": mse_val,
        "psnr_db": psnr_val,
        "ssim": ssim_val
    }
