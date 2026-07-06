"""
base_layer.py
-------------
Abstract base class that every layer type inherits from.

All layers share the same interface:
    forward(x)       — compute output, cache what backward needs
    backward(delta)  — compute gradients, return delta for previous layer
    update(lr)       — gradient descent step (no-op for layers with no params)
    get_params()     — return dict of learnable arrays (for save/load)
    set_params(d)    — restore learnable arrays from dict (for save/load)
    output_shape     — tuple (C, H, W) or (features,) set after build()

Network only ever calls these five methods — it never inspects the type
of the layer. This means adding a new layer type (BatchNorm, Dropout,
Attention...) requires zero changes to Network.
"""


class BaseLayer:

    # Set by build() or __init__ — Network reads this to chain shapes
    output_shape: tuple = None

    def forward(self, x):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()"
        )

    def backward(self, delta):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement backward()"
        )

    def update(self, lr: float) -> None:
        """
        Gradient descent step. Default is a no-op because layers like
        MaxPool and Flatten have no learnable parameters. Layers that do
        have parameters (Dense, Conv2D) override this.
        """
        pass

    def get_params(self) -> dict:
        """
        Return all learnable arrays as a flat dict of numpy arrays.
        Keys must be unique strings — Network prefixes them with the
        layer index before saving, e.g. "0_W", "0_b".
        Default: empty dict for parameterless layers.
        """
        return {}

    def set_params(self, params: dict) -> None:
        """
        Restore learnable arrays from a dict returned by get_params().
        Default: no-op for parameterless layers.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_shape={self.output_shape})"