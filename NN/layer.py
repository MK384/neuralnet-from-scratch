"""
layer.py
--------
All concrete layer types and the Activation class.

Public API (via Layer factory):
    Layer.Dense(n_out, activation)
    Layer.Conv2D(filters, kernel_size, activation, padding, stride)
    Layer.MaxPool(pool_size, stride)
    Layer.Flatten()

Activation:
    Activation("relu")
    Activation("sigmoid")
    Activation("tanh")
    Activation("softmax")
    Activation(fn=..., fn_prime=..., name=...)
"""

import numpy as np
from scipy.signal import correlate2d
from base_layer import BaseLayer
from activation import Activation
# ══════════════════════════════════════════════════════════════════════════ #
#  DenseLayer                                                                #
# ══════════════════════════════════════════════════════════════════════════ #

class DenseLayer(BaseLayer):
    """
    Fully-connected layer.

    Accepts 1-D input (features,) or 2-D batch (batch, features).
    If the previous layer outputs a 3-D shape (C, H, W), insert a
    Flatten layer before Dense.

    output_shape: (n_out,)
    """

    def __init__(self, n_in: int, n_out: int, activation: Activation):
        self.activation  = activation
        self.W           = activation.init_fn(n_out, n_in)
        self.b           = np.zeros(n_out)
        self.output_shape = (n_out,)
        self._a_in = None
        self._z    = None
        self.dW    = None
        self.db    = None

    def forward(self, x):
        self._a_in = x
        self._z    = x @ self.W.T + self.b
        return self.activation.fn(self._z)

    def backward(self, delta_in):
        delta = delta_in * self.activation.fn_prime(self._z)
        if delta.ndim == 1:
            self.dW = np.outer(delta, self._a_in)
        else:
            self.dW = (delta.T @ self._a_in) / delta.shape[0]
        self.db = delta.mean(axis=0) if delta.ndim > 1 else delta
        return delta @ self.W

    def backward_output_crossentropy(self, delta_in):
        """Combined Softmax+CrossEntropy gradient — skips fn_prime."""
        delta = delta_in
        if delta.ndim == 1:
            self.dW = np.outer(delta, self._a_in)
        else:
            self.dW = (delta.T @ self._a_in) / delta.shape[0]
        self.db = delta.mean(axis=0) if delta.ndim > 1 else delta
        return delta @ self.W

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def get_params(self):
        return {"W": self.W, "b": self.b}

    def set_params(self, params):
        self.W = params["W"]
        self.b = params["b"]

    def __repr__(self):
        n_in = self.W.shape[1]
        return (f"Dense({n_in} → {self.W.shape[0]}, "
                f"activation={self.activation.name}, "
                f"params={self.W.size + self.b.size})")


# ══════════════════════════════════════════════════════════════════════════ #
#  Conv2DLayer                                                               #
# ══════════════════════════════════════════════════════════════════════════ #

class Conv2DLayer(BaseLayer):
    """
    2-D Convolutional layer.

    Performs cross-correlation (the standard "convolution" in deep learning)
    between the input volume and a set of learnable filters.

    Parameters
    ----------
    input_shape  : (C_in, H_in, W_in)
    filters      : number of output channels (C_out)
    kernel_size  : int or (kH, kW) — filter spatial dimensions
    activation   : Activation object
    padding      : int, pixels of zero-padding on each side (default 0)
                   (ready for future use — currently applied but stride=1 only)
    stride       : int, step size of the sliding window (default 1)
                   (ready for future use — currently only stride=1 is fully tested)

    Shapes
    ------
    input  : (batch, C_in,  H_in,  W_in)
    W      : (C_out, C_in,  kH,    kW)    — one kernel per (output, input) channel pair
    b      : (C_out,)                      — one bias per output channel
    output : (batch, C_out, H_out, W_out)

    where H_out = (H_in + 2*padding - kH) // stride + 1
          W_out = (W_in + 2*padding - kW) // stride + 1

    Why cross-correlation and not convolution?
    ------------------------------------------
    True convolution flips the kernel before sliding. Cross-correlation
    does not flip. In practice every deep learning framework uses
    cross-correlation and calls it "convolution" because the kernels are
    learned anyway — whether the initial kernel is flipped or not doesn't
    matter since backprop will learn the right values either way.
    scipy.signal.correlate2d implements cross-correlation by default
    when mode='valid' or mode='same'.
    """

    def __init__(
        self,
        input_shape: tuple,
        filters: int,
        kernel_size,
        activation: Activation,
        padding: int = 0,
        stride: int = 1,
    ):
        C_in, H_in, W_in = input_shape
        kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.filters     = filters
        self.kH, self.kW = kH, kW
        self.activation  = activation
        self.padding     = padding
        self.stride      = stride
        self.C_in        = C_in

        # Weight init: He for ReLU, Xavier otherwise.
        # Fan-in for a conv filter = C_in * kH * kW
        fan_in = C_in * kH * kW
        std = np.sqrt(2.0 / fan_in) if activation.name.lower() == "relu" \
              else np.sqrt(1.0 / fan_in)
        self.W = np.random.randn(filters, C_in, kH, kW) * std
        self.b = np.zeros(filters)

        # Compute output spatial dimensions
        H_out = (H_in + 2 * padding - kH) // stride + 1
        W_out = (W_in + 2 * padding - kW) // stride + 1
        self.output_shape = (filters, H_out, W_out)

        # Cache for backward
        self._x_padded = None
        self.dW = None
        self.db = None

    def _pad(self, x):
        """Apply zero-padding to spatial dimensions if padding > 0."""
        if self.padding == 0:
            return x
        # x shape: (batch, C, H, W)
        return np.pad(
            x,
            ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant"
        )

    def forward(self, x):
        """
        x : (batch, C_in, H_in, W_in)
        """
        batch            = x.shape[0]
        x_padded         = self._pad(x)
        self._x_padded   = x_padded                           # cache for backward
        C_out, H_out, W_out = self.output_shape

        z = np.zeros((batch, C_out, H_out, W_out))

        # For each sample in the batch, each output channel (filter),
        # and each input channel: cross-correlate and accumulate.
        for n in range(batch):
            for f in range(C_out):
                for c in range(self.C_in):
                    # correlate2d 'valid' mode: no padding, output shrinks by kernel-1
                    z[n, f] += correlate2d(
                        x_padded[n, c], self.W[f, c], mode="valid"
                    )
                z[n, f] += self.b[f]

        self._z = z
        return self.activation.fn(z)                           # (batch, C_out, H_out, W_out)

    def backward(self, delta_in):
        """
        delta_in : (batch, C_out, H_out, W_out)
                   gradient of cost w.r.t. activation output of this layer

        Returns delta_out : (batch, C_in, H_in, W_in)
                            gradient to pass to the previous layer

        Three things to compute
        -----------------------
        1. delta = delta_in ⊙ activation′(z)      element-wise

        2. dW[f,c] = Σ_n correlate(x_padded[n,c], delta[n,f])
           The weight gradient for filter f, input channel c is the
           cross-correlation of the padded input with the output gradient.
           This is the conv-layer equivalent of dW = δ · aᵀ in Dense.

        3. db[f] = Σ_n Σ_hw delta[n,f,h,w]
           Bias gradient: sum of all spatial positions and batch items.

        4. delta_out[n,c] = Σ_f full_correlate(delta[n,f], W[f,c])
           'full' correlation pads the delta with kernel-1 zeros on each
           side — this is equivalent to backpropagating through the
           sliding window. In Dense layers this was delta @ W; here it's
           the same idea but in 2D.
        """
        batch            = delta_in.shape[0]
        C_out, H_out, W_out = self.output_shape

        # Step 1 — apply activation derivative
        delta = delta_in * self.activation.fn_prime(self._z)  # (batch, C_out, H_out, W_out)

        # Step 2 & 3 — weight and bias gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        for f in range(C_out):
            self.db[f] = delta[:, f, :, :].sum()
            for c in range(self.C_in):
                for n in range(batch):
                    self.dW[f, c] += correlate2d(
                        self._x_padded[n, c], delta[n, f], mode="valid"
                    )
        self.dW /= batch

        # Step 4 — delta for previous layer (full cross-correlation)
        # We need to "un-pad" the result if padding was applied
        C_in   = self.C_in
        H_in_p = self._x_padded.shape[2]
        W_in_p = self._x_padded.shape[3]
        delta_out_padded = np.zeros((batch, C_in, H_in_p, W_in_p))

        for n in range(batch):
            for c in range(C_in):
                for f in range(C_out):
                    # Flip kernel for true backprop through cross-correlation
                    W_flipped = self.W[f, c, ::-1, ::-1]
                    delta_out_padded[n, c] += correlate2d(
                        delta[n, f], W_flipped, mode="full"
                    )

        # Strip padding to recover original input spatial size
        if self.padding > 0:
            p = self.padding
            delta_out = delta_out_padded[:, :, p:-p, p:-p]
        else:
            delta_out = delta_out_padded

        return delta_out

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def get_params(self):
        return {"W": self.W, "b": self.b}

    def set_params(self, params):
        self.W = params["W"]
        self.b = params["b"]

    def __repr__(self):
        C_in = self.C_in
        return (f"Conv2D(in={C_in}, filters={self.filters}, "
                f"kernel=({self.kH},{self.kW}), "
                f"padding={self.padding}, stride={self.stride}, "
                f"activation={self.activation.name}, "
                f"output={self.output_shape}, "
                f"params={self.W.size + self.b.size})")


# ══════════════════════════════════════════════════════════════════════════ #
#  MaxPoolLayer                                                              #
# ══════════════════════════════════════════════════════════════════════════ #

class MaxPoolLayer(BaseLayer):
    """
    2-D Max Pooling layer.

    Downsamples each channel independently by taking the maximum value
    in each non-overlapping pool_size × pool_size window.

    Parameters
    ----------
    input_shape : (C, H_in, W_in)
    pool_size   : int or (pH, pW) — pooling window size (default 2)
    stride      : int or (sH, sW) — step between windows (default = pool_size)
                  When stride == pool_size, windows don't overlap (standard).

    output_shape: (C, H_out, W_out)
    where H_out = (H_in - pH) // sH + 1

    No learnable parameters — update() is a no-op.

    Backward pass (max-unpooling)
    -----------------------------
    Gradient flows only to the position that had the maximum value in
    each pooling window. All other positions receive zero gradient.
    We cache the argmax positions during forward for efficient backward.
    """

    def __init__(self, input_shape: tuple, pool_size=2, stride=None):
        C, H_in, W_in = input_shape
        pH, pW = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        sH, sW = (pH, pW) if stride is None else \
                 ((stride, stride) if isinstance(stride, int) else stride)

        self.pH, self.pW = pH, pW
        self.sH, self.sW = sH, sW
        self.C           = C

        H_out = (H_in - pH) // sH + 1
        W_out = (W_in - pW) // sW + 1
        self.output_shape = (C, H_out, W_out)

        # Cache filled during forward
        self._x_shape = None
        self._mask    = None    # boolean mask of max positions

    def forward(self, x):
        """x : (batch, C, H_in, W_in)"""
        batch, C, H_in, W_in = x.shape
        C_out, H_out, W_out  = self.output_shape
        self._x_shape = x.shape

        out  = np.zeros((batch, C, H_out, W_out))
        # Store mask as same shape as input — True where the max was taken
        self._mask = np.zeros_like(x, dtype=bool)

        for h in range(H_out):
            for w in range(W_out):
                h0, h1 = h * self.sH, h * self.sH + self.pH
                w0, w1 = w * self.sW, w * self.sW + self.pW

                window = x[:, :, h0:h1, w0:w1]               # (batch, C, pH, pW)
                out[:, :, h, w] = window.max(axis=(2, 3))

                # Build mask: mark position of max in each window
                max_vals = out[:, :, h, w][:, :, None, None]  # broadcast shape
                self._mask[:, :, h0:h1, w0:w1] |= (window == max_vals)

        return out

    def backward(self, delta_in):
        """
        delta_in : (batch, C, H_out, W_out)

        Upsamples gradient back to input shape by routing each gradient
        value to the position(s) that held the maximum. If two positions
        tied for the max, both receive the gradient (mask handles this).
        """
        delta_out = np.zeros(self._x_shape)
        C, H_out, W_out = self.output_shape

        for h in range(H_out):
            for w in range(W_out):
                h0, h1 = h * self.sH, h * self.sH + self.pH
                w0, w1 = w * self.sW, w * self.sW + self.pW

                # Broadcast delta value to all max positions in the window
                d = delta_in[:, :, h, w][:, :, None, None]    # (batch, C, 1, 1)
                delta_out[:, :, h0:h1, w0:w1] += \
                    d * self._mask[:, :, h0:h1, w0:w1]

        return delta_out

    def __repr__(self):
        return (f"MaxPool(pool=({self.pH},{self.pW}), "
                f"stride=({self.sH},{self.sW}), "
                f"output={self.output_shape})")


# ══════════════════════════════════════════════════════════════════════════ #
#  FlattenLayer                                                              #
# ══════════════════════════════════════════════════════════════════════════ #

class FlattenLayer(BaseLayer):
    """
    Reshapes (batch, C, H, W) → (batch, C*H*W).

    Bridges the gap between convolutional layers (3-D feature maps)
    and Dense layers (1-D vectors). No learnable parameters.

    Backward simply reshapes the gradient back to the input shape.
    """

    def __init__(self, input_shape: tuple):
        C, H, W = input_shape
        self.output_shape  = (C * H * W,)
        self._input_shape  = input_shape   # needed for backward reshape
        self._batch        = None

    def forward(self, x):
        """x : (batch, C, H, W)"""
        self._batch = x.shape[0]
        return x.reshape(self._batch, -1)  # (batch, C*H*W)

    def backward(self, delta_in):
        """delta_in : (batch, C*H*W) → (batch, C, H, W)"""
        C, H, W = self._input_shape
        return delta_in.reshape(self._batch, C, H, W)

    def __repr__(self):
        C, H, W = self._input_shape
        return f"Flatten({C}×{H}×{W} → {C*H*W})"


# ══════════════════════════════════════════════════════════════════════════ #
#  Spec objects — returned by Layer.* factory methods                        #
# ══════════════════════════════════════════════════════════════════════════ #

class _DenseSpec:
    def __init__(self, n_out, activation):
        self.n_out      = n_out
        self.activation = activation

    def build(self, input_shape):
        # input_shape is (features,) for 1-D, or (C,H,W) for 3-D
        if isinstance(input_shape, tuple) and len(input_shape) == 3:
            raise ValueError(
                "Cannot feed a 3-D shape directly into Dense. "
                "Add a Flatten() layer first."
            )
        n_in = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        return DenseLayer(n_in, self.n_out, self.activation)

    def __repr__(self):
        return f"DenseSpec(n_out={self.n_out}, activation={self.activation.name})"


class _Conv2DSpec:
    def __init__(self, filters, kernel_size, activation, padding, stride):
        self.filters     = filters
        self.kernel_size = kernel_size
        self.activation  = activation
        self.padding     = padding
        self.stride      = stride

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                f"Conv2D expects a 3-D input_shape (C, H, W), "
                f"got {input_shape}. "
                f"Ensure input_shape is (channels, height, width)."
            )
        return Conv2DLayer(
            input_shape  = input_shape,
            filters      = self.filters,
            kernel_size  = self.kernel_size,
            activation   = self.activation,
            padding      = self.padding,
            stride       = self.stride,
        )

    def __repr__(self):
        return (f"Conv2DSpec(filters={self.filters}, "
                f"kernel={self.kernel_size}, "
                f"activation={self.activation.name})")


class _MaxPoolSpec:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride    = stride

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                f"MaxPool expects a 3-D input_shape (C, H, W), got {input_shape}."
            )
        return MaxPoolLayer(input_shape, self.pool_size, self.stride)

    def __repr__(self):
        return f"MaxPoolSpec(pool_size={self.pool_size}, stride={self.stride})"


class _FlattenSpec:
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                f"Flatten expects a 3-D input_shape (C, H, W), got {input_shape}."
            )
        return FlattenLayer(input_shape)

    def __repr__(self):
        return "FlattenSpec()"


# ══════════════════════════════════════════════════════════════════════════ #
#  Layer — public factory (the only thing users import)                      #
# ══════════════════════════════════════════════════════════════════════════ #

class Layer:
    """
    Factory class — never instantiated directly.

    All methods are class methods that return spec objects.
    Network.build() calls spec.build(input_shape) to create real layers.

    Examples
    --------
    Layer.Dense(128, activation=Activation("relu"))
    Layer.Conv2D(32, kernel_size=3, activation=Activation("relu"))
    Layer.Conv2D(64, kernel_size=3, activation=Activation("relu"), padding=1)
    Layer.MaxPool(pool_size=2)
    Layer.Flatten()
    """

    @classmethod
    def Dense(cls, n_out: int, activation: Activation = None):
        act = activation if activation is not None else Activation("sigmoid")
        return _DenseSpec(n_out, act)

    @classmethod
    def Conv2D(
        cls,
        filters: int,
        kernel_size=3,
        activation: Activation = None,
        padding: int = 0,
        stride: int = 1,
    ):
        act = activation if activation is not None else Activation("relu")
        return _Conv2DSpec(filters, kernel_size, act, padding, stride)

    @classmethod
    def MaxPool(cls, pool_size=2, stride=None):
        return _MaxPoolSpec(pool_size, stride)

    @classmethod
    def Flatten(cls):
        return _FlattenSpec()