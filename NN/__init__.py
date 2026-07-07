"""
nn — a minimal neural network framework built with NumPy.

Public API
----------
from NN import Network, Layer, Activation
from NN import Adam, AdaGrad, RMSProp, Momentum, SGD
"""

# Import order matters — dependencies must come before dependents.
# base_layer has no internal deps → first
# layer depends on base_layer (Activation lives here too) → second
# optimizer depends on nothing internal → third
# network depends on layer and optimizer → last
from .activation import Activation
from .base_layer import BaseLayer
from .layer      import (
    Layer,
    DenseLayer,
    Conv2DLayer,
    MaxPoolLayer,
    AvgPoolLayer,
    FlattenLayer,
    BatchNormLayer,
    DropoutLayer,
)
from .optimizer  import Adam, AdaGrad, RMSProp, Momentum, SGD
from .network    import Network

__all__ = [
    # Core
    "Network",
    "Layer",
    "Activation",
    "BaseLayer",
    # Concrete layer classes (needed for isinstance checks inside network.py)
    "DenseLayer",
    "Conv2DLayer",
    "MaxPoolLayer",
    "AvgPoolLayer",
    "FlattenLayer",
    "BatchNormLayer",
    "DropoutLayer",
    # Optimizers
    "Adam",
    "AdaGrad",
    "RMSProp",
    "Momentum",
    "SGD",
]