"""
Deep Neural Network Architecture (MEGNet) for Source Parcel Localization.
PyTorch multi-layer perceptron with positive class weighting and Optuna tuned hyperparameters.
Author: Abdul Rehman Rattu
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MEGDataset(Dataset):
    """PyTorch Dataset for source-projected features and binary target matrices."""

    def __init__(self, X_source: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X_source)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MEGNetModule(nn.Module):
    """PyTorch MLP mapping 450 cortical parcel features to 450 activation logits."""

    def __init__(
        self,
        input_size: int = 450,
        hidden_size: int = 1024,
        output_size: int = 450,
        dropout: float = 0.107
    ):
        super(MEGNetModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


class MEGNetLocalizer:
    """
    Orchestration wrapper for MEGNet training, prediction, and dynamic sparsity thresholding.
    """

    def __init__(
        self,
        input_size: int = 450,
        hidden_size: int = 1024,
        output_size: int = 450,
        dropout: float = 0.107,
        lr: float = 0.000157,
        pos_weight: float = 1.16,
        batch_size: int = 64,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = MEGNetModule(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout
        ).to(self.device)

        self.lr = lr
        self.pos_weight = pos_weight
        self.batch_size = batch_size
        self.output_size = output_size

    def fit(
        self,
        X_train_features: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 20,
        verbose: bool = True
    ) -> "MEGNetLocalizer":
        """Trains MEGNet using positive-weighted Binary Cross Entropy."""
        dataset = MEGDataset(X_train_features, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        weight_tensor = torch.tensor([self.pos_weight] * self.output_size, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose and ((epoch + 1) % 5 == 0 or epoch == epochs - 1):
                avg_loss = total_loss / len(loader)
                print(f"  Epoch [{epoch+1:2d}/{epochs:2d}] Loss: {avg_loss:.4f}")

        return self

    def predict_proba(self, X_features: np.ndarray) -> np.ndarray:
        """Computes sigmoid parcel activation probabilities."""
        self.model.eval()
        dataset = MEGDataset(X_features, np.zeros((len(X_features), self.output_size)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probas = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                sig = torch.sigmoid(logits).cpu().numpy()
                probas.append(sig)

        return np.vstack(probas)

    def predict(
        self,
        X_features: np.ndarray,
        threshold: float = 0.5,
        max_sources: int = 3
    ) -> np.ndarray:
        """
        Predicts active cortical parcels with dynamic 1 to 3 sources selection.
        """
        raw_probs = self.predict_proba(X_features)
        n_samples = raw_probs.shape[0]
        y_pred = np.zeros_like(raw_probs, dtype=int)

        for i in range(n_samples):
            sorted_indices = np.argsort(raw_probs[i])[::-1]
            active_count = int(np.sum(raw_probs[i] > threshold))
            k = min(max_sources, max(1, active_count))
            top_indices = sorted_indices[:k]
            y_pred[i, top_indices] = 1

        return y_pred
