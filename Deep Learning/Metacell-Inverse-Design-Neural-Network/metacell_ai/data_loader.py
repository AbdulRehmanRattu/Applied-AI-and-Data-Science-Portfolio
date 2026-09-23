"""
Data Ingestion and Preprocessing Pipeline for Metacell Simulations
===================================================================
Loads 728 rigorous full-wave 3D CST Microwave Studio electromagnetic simulations
and formats forward and inverse design training manifolds.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


class MetacellDataLoader:
    def __init__(self, data_path=None, test_size=0.2, random_state=42):
        if data_path is None:
            # Default to relative data path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(os.path.dirname(current_dir), "data", "ANN.xlsx")
        
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        
        self.df = None
        self.scaler_geom = StandardScaler()
        self.scaler_em = StandardScaler()
        self._load_and_clean()

    def _load_and_clean(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Metacell dataset not found at {self.data_path}")
        
        self.df = pd.read_excel(self.data_path)
        # Drop identifier column if present
        if ' 3D Run ID' in self.df.columns:
            self.df = self.df.drop(columns=[' 3D Run ID'])
        
        # Standardize column naming
        self.geom_cols = ['c1', 'c2', 'c3']
        self.em_cols = [
            'Magnitude 0f transmission coefficients(1.952GHz)',
            'Phase 0f transmission coefficients(1.952GHz)'
        ]
        
        self.X_geom_raw = self.df[self.geom_cols].values.astype(np.float32)
        self.Y_em_raw = self.df[self.em_cols].values.astype(np.float32)

    def get_forward_data(self, as_tensors=True):
        """
        Forward Problem: Geometry (C1, C2, C3) -> S-Parameters (|S21|, Phase)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_geom_raw, self.Y_em_raw,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        X_train_scaled = self.scaler_geom.fit_transform(X_train)
        X_test_scaled = self.scaler_geom.transform(X_test)
        
        y_train_scaled = self.scaler_em.fit_transform(y_train)
        y_test_scaled = self.scaler_em.transform(y_test)
        
        if as_tensors:
            return (
                torch.tensor(X_train_scaled, dtype=torch.float32),
                torch.tensor(y_train_scaled, dtype=torch.float32),
                torch.tensor(X_test_scaled, dtype=torch.float32),
                torch.tensor(y_test_scaled, dtype=torch.float32)
            )
        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled

    def get_inverse_data(self, as_tensors=True):
        """
        Inverse Problem: S-Parameters (|S21|, Phase) -> Geometry (C1, C2, C3)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.Y_em_raw, self.X_geom_raw,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        X_train_scaled = self.scaler_em.transform(X_train)
        X_test_scaled = self.scaler_em.transform(X_test)
        
        y_train_scaled = self.scaler_geom.transform(y_train)
        y_test_scaled = self.scaler_geom.transform(y_test)
        
        if as_tensors:
            return (
                torch.tensor(X_train_scaled, dtype=torch.float32),
                torch.tensor(y_train_scaled, dtype=torch.float32),
                torch.tensor(X_test_scaled, dtype=torch.float32),
                torch.tensor(y_test_scaled, dtype=torch.float32)
            )
        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled

    def inverse_transform_em(self, y_scaled):
        if isinstance(y_scaled, torch.Tensor):
            y_scaled = y_scaled.detach().cpu().numpy()
        return self.scaler_em.inverse_transform(y_scaled)

    def inverse_transform_geom(self, y_scaled):
        if isinstance(y_scaled, torch.Tensor):
            y_scaled = y_scaled.detach().cpu().numpy()
        return self.scaler_geom.inverse_transform(y_scaled)

    def scale_em_input(self, em_raw):
        """Scale arbitrary target (|S21|, Phase) inputs for inverse design synthesis"""
        em_array = np.array(em_raw, dtype=np.float32)
        if em_array.ndim == 1:
            em_array = em_array.reshape(1, -1)
        scaled = self.scaler_em.transform(em_array)
        return torch.tensor(scaled, dtype=torch.float32)

    def scale_geom_input(self, geom_raw):
        """Scale arbitrary geometry (C1, C2, C3) inputs for forward simulation"""
        geom_array = np.array(geom_raw, dtype=np.float32)
        if geom_array.ndim == 1:
            geom_array = geom_array.reshape(1, -1)
        scaled = self.scaler_geom.transform(geom_array)
        return torch.tensor(scaled, dtype=torch.float32)
