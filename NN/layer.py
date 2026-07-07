"""
layer.py
--------
All concrete layer types and the Activation class.

Public API (via Layer factory):
    Layer.Dense(n_out, activation)
    Layer.Conv2D(filters, kernel_size, activation, padding, stride)
    Layer.MaxPool(pool_size, stride)
    Layer.AvgPool(pool_size, stride)
    Layer.Flatten()
    Layer.BatchNorm(momentum, epsilon)
    Layer.Dropout(rate)
"""

import numpy as np
from scipy.signal import correlate2d
from base_layer import BaseLayer
from activation import Activation

# ══════════════════════════════════════════════════════════════════════════ #
#  DenseLayer                                                                #
# ══════════════════════════════════════════════════════════════════════════ #

class DenseLayer(BaseLayer):
    """Fully-connected layer. output_shape: (n_out,)"""

    def __init__(self, n_in, n_out, activation):
        self.activation   = activation
        self.W            = activation.init_fn(n_out, n_in)
        self.b            = np.zeros(n_out)
        self.output_shape = (n_out,)
        self._a_in = self._z = self.dW = self.db = None

    def forward(self, x, training=True):
        self._a_in = x
        self._z    = x @ self.W.T + self.b
        return self.activation.fn(self._z)

    def backward(self, delta_in):
        delta = delta_in * self.activation.fn_prime(self._z)
        self.dW = (np.outer(delta, self._a_in) if delta.ndim == 1
                   else (delta.T @ self._a_in) / delta.shape[0])
        self.db = delta if delta.ndim == 1 else delta.mean(axis=0)
        return delta @ self.W

    def backward_output_crossentropy(self, delta_in):
        """Combined Softmax+CrossEntropy gradient — skips fn_prime."""
        delta   = delta_in
        self.dW = (np.outer(delta, self._a_in) if delta.ndim == 1
                   else (delta.T @ self._a_in) / delta.shape[0])
        self.db = delta if delta.ndim == 1 else delta.mean(axis=0)
        return delta @ self.W

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def get_params(self):        return {"W": self.W, "b": self.b}
    def set_params(self, p):     self.W = p["W"]; self.b = p["b"]

    def __repr__(self):
        return (f"Dense({self.W.shape[1]} → {self.W.shape[0]}, "
                f"act={self.activation.name}, params={self.W.size+self.b.size})")


# ══════════════════════════════════════════════════════════════════════════ #
#  Conv2DLayer                                                               #
# ══════════════════════════════════════════════════════════════════════════ #

class Conv2DLayer(BaseLayer):
    """
    2-D Convolutional layer using cross-correlation.

    W shape : (filters, C_in, kH, kW)
    b shape : (filters,)
    output  : (batch, filters, H_out, W_out)
    """

    def __init__(self, input_shape, filters, kernel_size,
                 activation, padding=0, stride=1):
        C_in, H_in, W_in = input_shape
        kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) \
                  else kernel_size
        self.filters     = filters
        self.kH, self.kW = kH, kW
        self.activation  = activation
        self.padding     = padding
        self.stride      = stride
        self.C_in        = C_in

        fan_in = C_in * kH * kW
        std    = (np.sqrt(2.0 / fan_in) if activation.name.lower() == "relu"
                  else np.sqrt(1.0 / fan_in))
        self.W = np.random.randn(filters, C_in, kH, kW) * std
        self.b = np.zeros(filters)

        H_out = (H_in + 2*padding - kH) // stride + 1
        W_out = (W_in + 2*padding - kW) // stride + 1
        self.output_shape = (filters, H_out, W_out)
        self._x_padded = self._z = self.dW = self.db = None

    def _pad(self, x):
        if self.padding == 0: return x
        return np.pad(x,
            ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))

    def forward(self, x, training=True):
        batch            = x.shape[0]
        x_padded         = self._pad(x)
        self._x_padded   = x_padded
        C_out, H_out, W_out = self.output_shape
        z = np.zeros((batch, C_out, H_out, W_out))
        for n in range(batch):
            for f in range(C_out):
                for c in range(self.C_in):
                    z[n,f] += correlate2d(x_padded[n,c], self.W[f,c], mode="valid")
                z[n,f] += self.b[f]
        self._z = z
        return self.activation.fn(z)

    def backward(self, delta_in):
        batch            = delta_in.shape[0]
        C_out, H_out, W_out = self.output_shape
        delta            = delta_in * self.activation.fn_prime(self._z)
        self.dW          = np.zeros_like(self.W)
        self.db          = np.zeros_like(self.b)
        for f in range(C_out):
            self.db[f] = delta[:,f].sum()
            for c in range(self.C_in):
                for n in range(batch):
                    self.dW[f,c] += correlate2d(
                        self._x_padded[n,c], delta[n,f], mode="valid")
        self.dW /= batch
        delta_out_padded = np.zeros_like(self._x_padded)
        for n in range(batch):
            for c in range(self.C_in):
                for f in range(C_out):
                    delta_out_padded[n,c] += correlate2d(
                        delta[n,f], self.W[f,c,::-1,::-1], mode="full")
        if self.padding > 0:
            p = self.padding
            return delta_out_padded[:,:,p:-p,p:-p]
        return delta_out_padded

    def update(self, lr):    self.W -= lr*self.dW; self.b -= lr*self.db
    def get_params(self):    return {"W": self.W, "b": self.b}
    def set_params(self, p): self.W = p["W"]; self.b = p["b"]

    def __repr__(self):
        return (f"Conv2D(filters={self.filters}, kernel=({self.kH},{self.kW}), "
                f"pad={self.padding}, stride={self.stride}, "
                f"act={self.activation.name}, out={self.output_shape})")


# ══════════════════════════════════════════════════════════════════════════ #
#  MaxPoolLayer                                                              #
# ══════════════════════════════════════════════════════════════════════════ #

class MaxPoolLayer(BaseLayer):
    """
    2-D Max Pooling. Gradient flows only to the maximum position in each window.
    No learnable parameters.
    """

    def __init__(self, input_shape, pool_size=2, stride=None):
        C, H_in, W_in = input_shape
        pH, pW = (pool_size,pool_size) if isinstance(pool_size,int) else pool_size
        sH, sW = (pH,pW) if stride is None else \
                 ((stride,stride) if isinstance(stride,int) else stride)
        self.pH,self.pW,self.sH,self.sW,self.C = pH,pW,sH,sW,C
        H_out = (H_in - pH) // sH + 1
        W_out = (W_in - pW) // sW + 1
        self.output_shape = (C, H_out, W_out)
        self._x_shape = self._mask = None

    def forward(self, x, training=True):
        batch,C,H_in,W_in = x.shape
        _,H_out,W_out     = self.output_shape
        self._x_shape     = x.shape
        out               = np.zeros((batch,C,H_out,W_out))
        self._mask        = np.zeros_like(x, dtype=bool)
        for h in range(H_out):
            for w in range(W_out):
                h0,h1 = h*self.sH, h*self.sH+self.pH
                w0,w1 = w*self.sW, w*self.sW+self.pW
                win   = x[:,:,h0:h1,w0:w1]
                out[:,:,h,w] = win.max(axis=(2,3))
                self._mask[:,:,h0:h1,w0:w1] |= \
                    (win == out[:,:,h,w][:,:,None,None])
        return out

    def backward(self, delta_in):
        delta_out         = np.zeros(self._x_shape)
        _,H_out,W_out     = self.output_shape
        for h in range(H_out):
            for w in range(W_out):
                h0,h1 = h*self.sH, h*self.sH+self.pH
                w0,w1 = w*self.sW, w*self.sW+self.pW
                delta_out[:,:,h0:h1,w0:w1] += \
                    delta_in[:,:,h,w][:,:,None,None] * self._mask[:,:,h0:h1,w0:w1]
        return delta_out

    def __repr__(self):
        return (f"MaxPool(pool=({self.pH},{self.pW}), "
                f"stride=({self.sH},{self.sW}), out={self.output_shape})")


# ══════════════════════════════════════════════════════════════════════════ #
#  AvgPoolLayer                (NEW)                                         #
# ══════════════════════════════════════════════════════════════════════════ #

class AvgPoolLayer(BaseLayer):
    """
    2-D Average Pooling.

    Forward: take the mean of each pooling window.
    Backward: distribute the gradient equally to every position in the window.

    vs MaxPool
    ----------
    MaxPool routes the gradient to one winner position.
    AvgPool spreads it evenly — every position contributed equally to the
    average, so every position receives an equal share of the gradient.
    AvgPool is smoother and sometimes used in the final spatial reduction
    before Dense layers (Global Average Pooling).

    No learnable parameters.
    """

    def __init__(self, input_shape, pool_size=2, stride=None):
        C, H_in, W_in = input_shape
        pH, pW = (pool_size,pool_size) if isinstance(pool_size,int) else pool_size
        sH, sW = (pH,pW) if stride is None else \
                 ((stride,stride) if isinstance(stride,int) else stride)
        self.pH,self.pW,self.sH,self.sW = pH,pW,sH,sW
        H_out = (H_in - pH) // sH + 1
        W_out = (W_in - pW) // sW + 1
        self.output_shape = (C, H_out, W_out)
        self._x_shape     = None

    def forward(self, x, training=True):
        batch,C,H_in,W_in = x.shape
        _,H_out,W_out     = self.output_shape
        self._x_shape     = x.shape
        out               = np.zeros((batch,C,H_out,W_out))
        for h in range(H_out):
            for w in range(W_out):
                h0,h1 = h*self.sH, h*self.sH+self.pH
                w0,w1 = w*self.sW, w*self.sW+self.pW
                out[:,:,h,w] = x[:,:,h0:h1,w0:w1].mean(axis=(2,3))
        return out

    def backward(self, delta_in):
        """
        Each position in the pooling window contributed 1/(pH*pW) of the
        output. By the chain rule, each receives 1/(pH*pW) of the gradient.
        """
        delta_out     = np.zeros(self._x_shape)
        _,H_out,W_out = self.output_shape
        pool_area     = self.pH * self.pW
        for h in range(H_out):
            for w in range(W_out):
                h0,h1 = h*self.sH, h*self.sH+self.pH
                w0,w1 = w*self.sW, w*self.sW+self.pW
                # Broadcast: every cell in the window gets an equal share
                delta_out[:,:,h0:h1,w0:w1] += \
                    delta_in[:,:,h,w][:,:,None,None] / pool_area
        return delta_out

    def __repr__(self):
        return (f"AvgPool(pool=({self.pH},{self.pW}), "
                f"stride=({self.sH},{self.sW}), out={self.output_shape})")


# ══════════════════════════════════════════════════════════════════════════ #
#  FlattenLayer                                                              #
# ══════════════════════════════════════════════════════════════════════════ #

class FlattenLayer(BaseLayer):
    """Reshapes (batch, C, H, W) → (batch, C*H*W). No parameters."""

    def __init__(self, input_shape):
        C,H,W = input_shape
        self.output_shape = (C*H*W,)
        self._input_shape = input_shape
        self._batch       = None

    def forward(self, x, training=True):
        self._batch = x.shape[0]
        return x.reshape(self._batch, -1)

    def backward(self, delta_in):
        C,H,W = self._input_shape
        return delta_in.reshape(self._batch, C, H, W)

    def __repr__(self):
        C,H,W = self._input_shape
        return f"Flatten({C}×{H}×{W} → {C*H*W})"


# ══════════════════════════════════════════════════════════════════════════ #
#  BatchNormLayer              (NEW)                                         #
# ══════════════════════════════════════════════════════════════════════════ #

class BatchNormLayer(BaseLayer):
    """
    Batch Normalisation — normalises activations across the batch,
    then applies a learned scale (gamma) and shift (beta).

    Why batch normalisation?
    ------------------------
    Without it, the distribution of each layer's input shifts as the
    weights of earlier layers change during training — called internal
    covariate shift. This forces each layer to constantly adapt to a
    moving target, slowing training significantly.

    BatchNorm fixes this by normalising the input to each layer to have
    mean=0 and variance=1 (per feature), then letting the network learn
    the optimal scale and shift via gamma and beta. This allows:
      - Much higher learning rates (training is more stable)
      - Less sensitivity to weight initialisation
      - Acts as a mild regulariser (reduces overfitting slightly)

    Two modes
    ---------
    Training   — normalise using the current BATCH mean and variance.
                 Also updates running_mean and running_var for inference.
    Inference  — normalise using the RUNNING mean and variance accumulated
                 during training. The batch may be size 1 at inference,
                 making batch statistics meaningless.

    Supports both:
      - 1-D input (batch, features)         — after Dense layers
      - 4-D input (batch, C, H, W)          — after Conv2D layers
        normalises per channel across all spatial positions (H, W)

    Learnable parameters
    --------------------
    gamma : scale,  shape (features,) or (C,)  — initialised to 1
    beta  : shift,  shape (features,) or (C,)  — initialised to 0

    Parameters
    ----------
    input_shape : (features,) or (C, H, W)
    momentum    : weight for the running average update (default 0.9)
                  running = momentum * running + (1-momentum) * batch_stat
    epsilon     : small constant added to variance for numerical stability
                  prevents division by zero when variance is near 0
    """

    def __init__(self, input_shape, momentum=0.9, epsilon=1e-8):
        self.momentum = momentum
        self.epsilon  = epsilon

        # Determine number of features to normalise over
        if len(input_shape) == 1:
            # Dense output: (features,)
            n_features = input_shape[0]
            self._mode = "1d"
        elif len(input_shape) == 3:
            # Conv output: (C, H, W) — normalise per channel
            n_features = input_shape[0]
            self._mode = "4d"
        else:
            raise ValueError(
                f"BatchNorm expects input_shape of length 1 or 3, "
                f"got {len(input_shape)}: {input_shape}"
            )

        self.output_shape = input_shape

        # Learnable parameters
        self.gamma = np.ones(n_features)
        self.beta  = np.zeros(n_features)

        # Running statistics (updated during training, used during inference)
        self.running_mean = np.zeros(n_features)
        self.running_var  = np.ones(n_features)

        # Gradients and cache
        self.dgamma = None
        self.dbeta  = None
        self._cache = None   # (x_norm, var, batch_size) — needed for backward

    def _reshape_params(self, x):
        """
        Reshape gamma/beta for broadcasting against x.

        For (batch, features)    → shape (1, features)   trivial
        For (batch, C, H, W)     → shape (1, C, 1, 1)    broadcasts over H,W
        """
        if self._mode == "1d":
            return self.gamma[None,:], self.beta[None,:]
        else:
            C = self.gamma.shape[0]
            return self.gamma.reshape(1,C,1,1), self.beta.reshape(1,C,1,1)

    def forward(self, x, training=True):
        """
        Forward pass.

        Training:
            mean  = mean over batch (and H,W for conv)
            var   = variance over batch
            x_norm = (x - mean) / sqrt(var + eps)
            out   = gamma * x_norm + beta
            Also updates running_mean and running_var.

        Inference:
            Uses stored running_mean and running_var.
            No running stat update.
        """
        if self._mode == "1d":
            # x: (batch, features)
            axes = (0,)
        else:
            # x: (batch, C, H, W) — average over batch + spatial dims
            axes = (0, 2, 3)

        if training:
            mean      = x.mean(axis=axes, keepdims=True)
            var       = x.var( axis=axes, keepdims=True)
            x_norm    = (x - mean) / np.sqrt(var + self.epsilon)

            # Update running statistics
            # Squeeze to (features,) or (C,) for storage
            batch_mean = mean.squeeze()
            batch_var  = var.squeeze()
            self.running_mean = (self.momentum * self.running_mean
                                 + (1-self.momentum) * batch_mean)
            self.running_var  = (self.momentum * self.running_var
                                 + (1-self.momentum) * batch_var)

            # Cache for backward
            batch_size    = x.shape[0]
            self._cache   = (x_norm, var, batch_size)
        else:
            # Inference — use running statistics
            if self._mode == "1d":
                mean   = self.running_mean[None,:]
                var    = self.running_var[ None,:]
            else:
                C = self.gamma.shape[0]
                mean   = self.running_mean.reshape(1,C,1,1)
                var    = self.running_var.reshape( 1,C,1,1)
            x_norm = (x - mean) / np.sqrt(var + self.epsilon)

        gamma, beta = self._reshape_params(x)
        return gamma * x_norm + beta

    def backward(self, delta_in):
        """
        Backward pass through batch normalisation.

        This is more involved than most layers because normalisation
        creates a dependency between all samples in the batch — changing
        one sample's output changes the mean and variance used to
        normalise everyone else.

        The full derivation yields three terms for the input gradient:
            dx = (1/N) * gamma/std * (
                N * delta_norm
                - delta_norm.sum(axis=0)
                - x_norm * (delta_norm * x_norm).sum(axis=0)
            )

        where delta_norm = delta_in * gamma  (chain rule through gamma*x_norm)

        For gamma and beta gradients:
            dgamma = sum(delta_in * x_norm, over batch and spatial dims)
            dbeta  = sum(delta_in,          over batch and spatial dims)
        """
        x_norm, var, N = self._cache
        gamma, _       = self._reshape_params(None)

        if self._mode == "1d":
            sum_axes       = (0,)
            N_effective    = N
        else:
            # For conv: sum over batch + spatial positions
            sum_axes       = (0, 2, 3)
            N_effective    = N * x_norm.shape[2] * x_norm.shape[3]

        std        = np.sqrt(var + self.epsilon)
        delta_norm = delta_in * gamma

        # Gradient w.r.t. gamma and beta
        if self._mode == "1d":
            self.dgamma = (delta_in * x_norm).sum(axis=0)
            self.dbeta  = delta_in.sum(axis=0)
        else:
            self.dgamma = (delta_in * x_norm).sum(axis=(0,2,3))
            self.dbeta  = delta_in.sum(axis=(0,2,3))

        # Gradient w.r.t. input x
        dx = (1.0 / (N_effective * std)) * (
            N_effective * delta_norm
            - delta_norm.sum(axis=sum_axes, keepdims=True)
            - x_norm * (delta_norm * x_norm).sum(axis=sum_axes, keepdims=True)
        )
        return dx

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta  -= lr * self.dbeta

    def get_params(self):
        return {
            "gamma":        self.gamma,
            "beta":         self.beta,
            "running_mean": self.running_mean,
            "running_var":  self.running_var,
        }

    def set_params(self, p):
        self.gamma        = p["gamma"]
        self.beta         = p["beta"]
        self.running_mean = p["running_mean"]
        self.running_var  = p["running_var"]

    def __repr__(self):
        return (f"BatchNorm(features={self.gamma.shape[0]}, "
                f"momentum={self.momentum}, eps={self.epsilon})")


# ══════════════════════════════════════════════════════════════════════════ #
#  DropoutLayer               (NEW)                                          #
# ══════════════════════════════════════════════════════════════════════════ #

class DropoutLayer(BaseLayer):
    """
    Dropout regularisation.

    During training, randomly zeroes each neuron's output with probability
    `rate`. This prevents neurons from co-adapting — each neuron must learn
    features that are useful independently, which reduces overfitting.

    During inference, all neurons are active. To keep the expected output
    magnitude the same as training, we use inverted dropout:
    during training, surviving activations are scaled up by 1/(1-rate).
    This means inference requires no scaling adjustment at all.

    Why inverted dropout (scale at training, not inference)?
    --------------------------------------------------------
    The alternative is to scale down at inference by (1-rate). But that
    requires remembering to apply the scaling every single time you do
    inference — easy to forget and causes silent bugs. Inverted dropout
    bakes the correction into training so inference is always clean.

    Parameters
    ----------
    rate : float in [0, 1) — fraction of neurons to zero out.
           rate=0.0 means no dropout (layer is a pass-through).
           rate=0.5 means 50% of neurons are randomly dropped.
           Typical values: 0.2–0.5 for Dense layers.

    No learnable parameters. Backward simply masks with the same pattern
    used in forward (the mask is cached).
    """

    def __init__(self, input_shape, rate=0.5):
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}.")
        self.rate         = rate
        self.output_shape = input_shape
        self._mask        = None

    def forward(self, x, training=True):
        if not training or self.rate == 0.0:
            # Inference — pass through unchanged
            return x

        # Training — generate binary mask and scale
        # mask[i] = 1 with probability (1 - rate)
        #         = 0 with probability rate
        self._mask = (np.random.rand(*x.shape) > self.rate).astype(float)

        # Inverted dropout: scale surviving activations up so that the
        # expected value of each neuron's output is unchanged:
        # E[masked] = (1-rate) * x * 1/(1-rate) = x
        return x * self._mask / (1.0 - self.rate)

    def backward(self, delta_in):
        if self._mask is None:
            # Was called in inference mode — gradient passes through
            return delta_in
        # Apply the same mask: neurons that were zeroed contribute no gradient
        return delta_in * self._mask / (1.0 - self.rate)

    def __repr__(self):
        return f"Dropout(rate={self.rate}, out={self.output_shape})"


# ══════════════════════════════════════════════════════════════════════════ #
#  Spec objects                                                              #
# ══════════════════════════════════════════════════════════════════════════ #

class _DenseSpec:
    def __init__(self, n_out, activation):
        self.n_out = n_out; self.activation = activation
    def build(self, input_shape):
        if isinstance(input_shape, tuple) and len(input_shape) == 3:
            raise ValueError("Add Flatten() before Dense for 3-D inputs.")
        n_in = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        return DenseLayer(n_in, self.n_out, self.activation)

class _Conv2DSpec:
    def __init__(self, filters, kernel_size, activation, padding, stride):
        self.filters=filters; self.kernel_size=kernel_size
        self.activation=activation; self.padding=padding; self.stride=stride
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Conv2D needs 3-D input_shape (C,H,W), got {input_shape}.")
        return Conv2DLayer(input_shape, self.filters, self.kernel_size,
                           self.activation, self.padding, self.stride)

class _MaxPoolSpec:
    def __init__(self, pool_size, stride):
        self.pool_size=pool_size; self.stride=stride
    def build(self, input_shape):
        if len(input_shape)!=3:
            raise ValueError(f"MaxPool needs 3-D input_shape, got {input_shape}.")
        return MaxPoolLayer(input_shape, self.pool_size, self.stride)

class _AvgPoolSpec:
    def __init__(self, pool_size, stride):
        self.pool_size=pool_size; self.stride=stride
    def build(self, input_shape):
        if len(input_shape)!=3:
            raise ValueError(f"AvgPool needs 3-D input_shape, got {input_shape}.")
        return AvgPoolLayer(input_shape, self.pool_size, self.stride)

class _FlattenSpec:
    def build(self, input_shape):
        if len(input_shape)!=3:
            raise ValueError(f"Flatten needs 3-D input_shape, got {input_shape}.")
        return FlattenLayer(input_shape)

class _BatchNormSpec:
    def __init__(self, momentum, epsilon):
        self.momentum=momentum; self.epsilon=epsilon
    def build(self, input_shape):
        return BatchNormLayer(input_shape, self.momentum, self.epsilon)

class _DropoutSpec:
    def __init__(self, rate):
        self.rate=rate
    def build(self, input_shape):
        return DropoutLayer(input_shape, self.rate)


# ══════════════════════════════════════════════════════════════════════════ #
#  Layer — public factory                                                    #
# ══════════════════════════════════════════════════════════════════════════ #

class Layer:
    """
    Factory — never instantiated directly.

    Examples
    --------
    Layer.Dense(128, activation=Activation("relu"))
    Layer.Conv2D(32, kernel_size=3, activation=Activation("relu"))
    Layer.Conv2D(32, kernel_size=3, activation=Activation("relu"), padding=1)
    Layer.MaxPool(pool_size=2)
    Layer.AvgPool(pool_size=2)
    Layer.Flatten()
    Layer.BatchNorm()
    Layer.Dropout(rate=0.5)
    """

    @classmethod
    def Dense(cls, n_out, activation=None):
        return _DenseSpec(n_out, activation or Activation("sigmoid"))

    @classmethod
    def Conv2D(cls, filters, kernel_size=3, activation=None, padding=0, stride=1):
        return _Conv2DSpec(filters, kernel_size, activation or Activation("relu"),
                           padding, stride)

    @classmethod
    def MaxPool(cls, pool_size=2, stride=None):
        return _MaxPoolSpec(pool_size, stride)

    @classmethod
    def AvgPool(cls, pool_size=2, stride=None):
        return _AvgPoolSpec(pool_size, stride)

    @classmethod
    def Flatten(cls):
        return _FlattenSpec()

    @classmethod
    def BatchNorm(cls, momentum=0.9, epsilon=1e-8):
        return _BatchNormSpec(momentum, epsilon)

    @classmethod
    def Dropout(cls, rate=0.5):
        return _DropoutSpec(rate)