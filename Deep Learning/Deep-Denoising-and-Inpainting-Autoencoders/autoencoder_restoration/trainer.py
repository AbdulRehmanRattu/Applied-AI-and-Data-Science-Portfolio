import numpy as np
from .corruptions import add_gaussian_noise, add_spatial_occlusions, add_combined_corruption
from .models.convolutional_autoencoder import build_keras_autoencoder
from .metrics import evaluate_restoration_batch


class AutoencoderTrainer:
    """
    Manages end-to-end corruptions, training cycles, loss history, and restoration evaluation.
    """
    def __init__(self, corruption_type: str = "gaussian", noise_factor: float = 0.2, 
                 occlusion_size: tuple = (8, 8), learning_rate: float = 0.0005):
        self.corruption_type = corruption_type.lower()
        self.noise_factor = noise_factor
        self.occlusion_size = occlusion_size
        self.learning_rate = learning_rate
        self.model = build_keras_autoencoder(learning_rate=learning_rate)
        self.history = None

    def corrupt(self, images: np.ndarray, seed: int = None) -> np.ndarray:
        if self.corruption_type == "gaussian":
            return add_gaussian_noise(images, noise_factor=self.noise_factor, seed=seed)
        elif self.corruption_type == "occlusion":
            return add_spatial_occlusions(images, occlusion_size=self.occlusion_size, seed=seed)
        elif self.corruption_type == "combined":
            return add_combined_corruption(images, noise_factor=self.noise_factor, 
                                          occlusion_size=self.occlusion_size, seed=seed)
        else:
            raise ValueError(f"Unknown corruption_type '{self.corruption_type}'. Choose from 'gaussian', 'occlusion', 'combined'.")

    def fit(self, x_train: np.ndarray, x_val: np.ndarray = None, epochs: int = 20, batch_size: int = 128, verbose: int = 1):
        x_train_corrupted = self.corrupt(x_train, seed=42)
        
        val_data = None
        if x_val is not None:
            x_val_corrupted = self.corrupt(x_val, seed=123)
            val_data = (x_val_corrupted, x_val)

        self.history = self.model.fit(
            x_train_corrupted, x_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=val_data,
            verbose=verbose
        )
        return self.history

    def evaluate(self, x_test: np.ndarray) -> dict:
        x_test_corrupted = self.corrupt(x_test, seed=999)
        reconstructed = self.model.predict(x_test_corrupted, verbose=0)
        metrics = evaluate_restoration_batch(x_test, reconstructed)
        return metrics

    def restore(self, x_corrupted: np.ndarray) -> np.ndarray:
        return self.model.predict(x_corrupted, verbose=0)
