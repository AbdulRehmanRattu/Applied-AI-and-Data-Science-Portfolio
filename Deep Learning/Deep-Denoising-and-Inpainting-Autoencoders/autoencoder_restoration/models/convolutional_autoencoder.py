import os
import numpy as np


def build_keras_autoencoder(input_shape=(32, 32, 3), learning_rate: float = 0.0005):
    """
    Constructs and compiles a Deep Convolutional Autoencoder using Keras / TensorFlow.
    
    Architecture:
        Input: (32, 32, 3)
        Encoder:
            Conv2D(32, 3x3, relu) -> MaxPool(2x2) => (16, 16, 32)
            Conv2D(64, 3x3, relu) -> MaxPool(2x2) => (8, 8, 64)
            Conv2D(128, 3x3, relu) -> MaxPool(2x2) => (4, 4, 128) [Latent Bottleneck: 2048 dims]
        Decoder:
            UpSampling2D(2x2) -> Conv2D(64, 3x3, relu) => (8, 8, 64)
            UpSampling2D(2x2) -> Conv2D(32, 3x3, relu) => (16, 16, 32)
            UpSampling2D(2x2) -> Conv2D(3, 3x3, sigmoid) => (32, 32, 3)
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    input_img = Input(shape=input_shape, name="input_corrupted_image")

    # Encoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(input_img)
    x = MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
    x = MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="enc_conv3")(x)
    bottleneck = MaxPooling2D((2, 2), padding="same", name="latent_bottleneck")(x)

    # Decoder
    x = UpSampling2D((2, 2), name="dec_upsample1")(bottleneck)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="dec_conv1")(x)
    x = UpSampling2D((2, 2), name="dec_upsample2")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv2")(x)
    x = UpSampling2D((2, 2), name="dec_upsample3")(x)
    decoded = Conv2D(input_shape[-1], (3, 3), activation="sigmoid", padding="same", name="dec_output")(x)

    model = Model(inputs=input_img, outputs=decoded, name="deep_convolutional_autoencoder")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model


def get_model_layer_summary_table(model) -> list:
    """
    Extracts structural configuration of every layer into a structured tabular list.
    """
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
    
    rows = []
    block_num = 1
    is_decoder = False
    cur_shape = (None, 32, 32, 3)
    
    for layer in model.layers:
        config = layer.get_config()
        layer_cls = layer.__class__.__name__
        
        num_filters = "-"
        kernel_size = "-"
        strides = "-"
        activation = "-"
        batch_norm = "NO"
        
        try:
            cur_shape = layer.compute_output_shape(cur_shape)
            out_dim = "x".join(map(str, cur_shape[1:]))
        except Exception:
            out_dim = "32x32x3"

        if isinstance(layer, Conv2D):
            block_type = "Decoder" if is_decoder else "Encoder"
            label = f"{block_type} Block {block_num} ({layer_cls})"
            num_filters = str(config.get("filters", "-"))
            kernel_size = "x".join(map(str, config.get("kernel_size", "-")))
            strides = "x".join(map(str, config.get("strides", "-")))
            activation = config.get("activation", "-")
        elif isinstance(layer, MaxPooling2D):
            label = f"Encoder Pool {block_num} ({layer_cls})"
            pool_sz = config.get("pool_size", "-")
            kernel_size = "x".join(map(str, pool_sz)) if isinstance(pool_sz, (list, tuple)) else str(pool_sz)
            block_num += 1
        elif isinstance(layer, UpSampling2D):
            if not is_decoder:
                is_decoder = True
                block_num = 1
            label = f"Decoder UpSample {block_num} ({layer_cls})"
            up_sz = config.get("size", "-")
            kernel_size = "x".join(map(str, up_sz)) if isinstance(up_sz, (list, tuple)) else str(up_sz)
            block_num += 1
        else:
            label = f"Input ({layer_cls})"

        rows.append({
            "layer_name": layer.name,
            "block_type": label,
            "filters": num_filters,
            "kernel_size": kernel_size,
            "strides": strides,
            "activation": activation,
            "batch_norm": batch_norm,
            "output_dim": out_dim
        })
        
    return rows


try:
    import torch
    import torch.nn as nn

    class PyTorchConvAutoencoder(nn.Module):
        """
        PyTorch equivalent implementation of the Deep Convolutional Autoencoder.
        """
        def __init__(self, in_channels: int = 3):
            super().__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 16, 32)
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # (8, 8, 64)
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)   # (4, 4, 128) - Bottleneck
            )
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            latent = self.encoder(x)
            reconstruction = self.decoder(latent)
            return reconstruction

except ImportError:
    PyTorchConvAutoencoder = None
