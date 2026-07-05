"""
nn — a minimal neural network framework built with NumPy.

Public API
----------
from nn import Network, Layer, Activation
"""

from .activation import Activation
from .layer   import Layer , _DenseSpec
from .network import Network

__all__ = ["Network", "Layer", "Activation", "_DenseSpec"]