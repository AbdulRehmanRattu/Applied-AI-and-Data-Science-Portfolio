"""
Deep Surrogate Neural Network Architectures for Metacell Design
================================================================
Implements Forward Surrogate and Inverse Synthesis Deep Neural Networks
in PyTorch with Batch Normalization, Dropout Regularization, and LeakyReLU/ReLU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os


class ForwardSurrogateNetwork(nn.Module):
    """
    Forward Surrogate Model: Maps Physical Dimensions (C1, C2, C3) -> S-Parameters (|S21|, Phase)
    Input Dimension: 3
    Output Dimension: 2
    """
    def __init__(self, input_dim=3, output_dim=2, hidden_dims=[128, 256, 256, 128], dropout_rate=0.15):
        super(ForwardSurrogateNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class InverseSynthesisNetwork(nn.Module):
    """
    Inverse Design Model: Maps Target S-Parameters (|S21|, Phase) -> Physical Dimensions (C1, C2, C3)
    Input Dimension: 2
    Output Dimension: 3
    """
    def __init__(self, input_dim=2, output_dim=3, hidden_dims=[128, 256, 256, 128], dropout_rate=0.15):
        super(InverseSynthesisNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_metacell_model(model, X_train, y_train, X_val, y_val, epochs=250, lr=0.001, weight_decay=1e-4, batch_size=32):
    """
    Train a surrogate model using mini-batch Adam optimization and Mean Squared Error.
    """
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        
        epoch_train_loss = running_loss / len(dataset)
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            epoch_val_loss = criterion(val_preds, y_val).item()
        
        scheduler.step(epoch_val_loss)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
    
    return history
