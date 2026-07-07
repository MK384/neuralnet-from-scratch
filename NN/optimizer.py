"""
optimizer.py
------------
Gradient-based optimizers. Each optimizer is stateful — it accumulates
information across batches (velocity, squared gradients, etc.) and uses
that history to compute smarter parameter updates than plain gradient descent.

Public API
----------
    SGD(lr=0.01)
    Momentum(lr=0.01, beta=0.9)
    RMSProp(lr=0.001, beta=0.9, epsilon=1e-8)
    AdaGrad(lr=0.01, epsilon=1e-8)
    Adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

Usage
-----
    net = Network(..., optimizer=Adam(lr=0.001))
    net.train(X, y, epochs=30, batch_size=32)

Design
------
Each optimizer stores its state in dicts keyed by layer index:
    self._state[layer_id][param_name] = numpy array

This keeps layer code completely clean — layers only hold W, b, dW, db.
The optimizer is the sole owner of velocity, sq_grad, etc.

Adding a new optimizer: subclass BaseOptimizer, implement step(layer_id, params, grads).
Adding a new layer type: zero changes to any optimizer.
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════ #
#  Base                                                                      #
# ══════════════════════════════════════════════════════════════════════════ #

class BaseOptimizer:
    """
    Abstract base for all optimizers.

    Subclasses implement step(layer_id, params, grads) which takes the
    current parameter dict and gradient dict for one layer, updates params
    in-place, and returns them.

    Network calls:
        optimizer.update(layer_index, layer)
    which extracts params/grads from the layer, calls step(), and writes
    updated values back.
    """

    def update(self, layer_id: int, layer) -> None:
        """
        Called by Network for each layer after backward().
        Extracts params and grads, delegates to step(), writes back.
        """
        params = layer.get_params()
        if not params:
            return   # Dropout, MaxPool etc — no parameters

        # Build gradient dict with the same keys as params
        grads = self._get_grads(layer, params)

        # Initialize state for this layer on first call
        if layer_id not in self._state:
            self._init_state(layer_id, params)

        # Compute and apply update
        updated = self.step(layer_id, params, grads)

        # Write updated values back into the layer
        layer.set_params(updated)

    def _get_grads(self, layer, params: dict) -> dict:
        """
        Build gradient dict matching the params dict.
        Convention: gradient for param 'W' is layer.dW, for 'b' is layer.db,
        for 'gamma' is layer.dgamma, etc.
        """
        grad_map = {
            "W":     "dW",
            "b":     "db",
            "gamma": "dgamma",
            "beta":  "dbeta",
        }
        grads = {}
        for key in params:
            attr = grad_map.get(key)
            if attr and hasattr(layer, attr):
                grads[key] = getattr(layer, attr)
            else:
                grads[key] = np.zeros_like(params[key])
        return grads

    def _init_state(self, layer_id: int, params: dict) -> None:
        """Initialize optimizer state for a layer. Subclasses override."""
        self._state[layer_id] = {}

    def step(self, layer_id: int, params: dict, grads: dict) -> dict:
        """Apply one parameter update. Subclasses must implement."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear all accumulated state — call before restarting training."""
        self._state = {}
        self._t     = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ══════════════════════════════════════════════════════════════════════════ #
#  SGD (plain gradient descent)                                              #
# ══════════════════════════════════════════════════════════════════════════ #

class SGD(BaseOptimizer):
    """
    Plain stochastic gradient descent.

        w ← w − lr · ∂C/∂w

    No accumulated state — each update depends only on the current gradient.
    Included here so Network always has a consistent optimizer interface,
    even when no fancy optimizer is needed.

    Parameters
    ----------
    lr : learning rate η
    """

    def __init__(self, lr: float = 0.01):
        self.lr     = lr
        self._state = {}
        self._t     = 0

    def step(self, layer_id, params, grads):
        return {k: params[k] - self.lr * grads[k] for k in params}

    def __repr__(self):
        return f"SGD(lr={self.lr})"


# ══════════════════════════════════════════════════════════════════════════ #
#  Momentum                                                                  #
# ══════════════════════════════════════════════════════════════════════════ #

class Momentum(BaseOptimizer):
    """
    SGD with Momentum.

    Keeps a velocity vector that accumulates gradients over time,
    like a ball rolling downhill that builds up speed in consistent
    directions and is dampened in oscillating directions.

        v ← β·v + (1−β)·g          (or simply β·v + g in some formulations)
        w ← w − lr·v

    We use the standard formulation: v ← β·v + g (no (1-β) scaling).
    This means the effective learning rate at convergence is lr/(1-β),
    which is why Momentum typically uses a smaller lr than plain SGD.

    Parameters
    ----------
    lr   : learning rate
    beta : momentum coefficient (default 0.9)
           High β = strong momentum, slow to change direction.
           Low  β = weak momentum, behaves more like plain SGD.

    State per parameter: 1 array (velocity v)
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        self.lr     = lr
        self.beta   = beta
        self._state = {}
        self._t     = 0

    def _init_state(self, layer_id, params):
        self._state[layer_id] = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, layer_id, params, grads):
        state   = self._state[layer_id]
        updated = {}
        for k in params:
            # Accumulate velocity
            state[k] = self.beta * state[k] + grads[k]
            # Update parameter
            updated[k] = params[k] - self.lr * state[k]
        return updated

    def __repr__(self):
        return f"Momentum(lr={self.lr}, beta={self.beta})"


# ══════════════════════════════════════════════════════════════════════════ #
#  AdaGrad                                                                   #
# ══════════════════════════════════════════════════════════════════════════ #

class AdaGrad(BaseOptimizer):
    """
    Adaptive Gradient Algorithm.

    Adapts the learning rate per parameter based on the cumulative sum of
    squared gradients. Parameters that have received large gradients in
    the past get smaller effective learning rates; rarely updated parameters
    keep a larger effective rate.

        G ← G + g²           (accumulate squared gradients)
        w ← w − (lr / √(G + ε)) · g

    The ε prevents division by zero when G is near 0.

    Strength  — great for sparse features (NLP, embeddings) where some
                parameters are rarely updated and need larger steps.
    Weakness  — G only ever grows. Eventually every learning rate shrinks
                to near zero and learning stalls completely. Not suitable
                for very long training runs. RMSProp fixes this.

    Parameters
    ----------
    lr      : global learning rate (typically 0.01)
    epsilon : numerical stability constant (default 1e-8)

    State per parameter: 1 array (G, cumulative squared gradient)
    """

    def __init__(self, lr: float = 0.01, epsilon: float = 1e-8):
        self.lr      = lr
        self.epsilon = epsilon
        self._state  = {}
        self._t      = 0

    def _init_state(self, layer_id, params):
        self._state[layer_id] = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, layer_id, params, grads):
        state   = self._state[layer_id]
        updated = {}
        for k in params:
            # Accumulate squared gradients (G only grows)
            state[k] += grads[k] ** 2
            # Adapted update
            updated[k] = params[k] - (self.lr / (np.sqrt(state[k]) + self.epsilon)) * grads[k]
        return updated

    def __repr__(self):
        return f"AdaGrad(lr={self.lr}, epsilon={self.epsilon})"


# ══════════════════════════════════════════════════════════════════════════ #
#  RMSProp                                                                   #
# ══════════════════════════════════════════════════════════════════════════ #

class RMSProp(BaseOptimizer):
    """
    Root Mean Square Propagation.

    Fixes AdaGrad's "learning rate collapses to zero" problem by using an
    exponential moving average of squared gradients instead of accumulating
    them forever. Old gradients are gradually forgotten.

        v ← β·v + (1−β)·g²          (exponential moving avg of g²)
        w ← w − (lr / √(v + ε)) · g

    The β controls how quickly old information is forgotten:
        β = 0.9  → roughly last 10 gradient steps matter
        β = 0.99 → roughly last 100 steps matter

    Difference from AdaGrad: v is an EMA (bounded) vs G (unbounded sum).
    This means the learning rate never collapses permanently.

    Parameters
    ----------
    lr      : learning rate (typically 0.001)
    beta    : decay rate for squared gradient EMA (default 0.9)
    epsilon : numerical stability constant (default 1e-8)

    State per parameter: 1 array (v, EMA of squared gradients)
    """

    def __init__(self, lr: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8):
        self.lr      = lr
        self.beta    = beta
        self.epsilon = epsilon
        self._state  = {}
        self._t      = 0

    def _init_state(self, layer_id, params):
        self._state[layer_id] = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, layer_id, params, grads):
        state   = self._state[layer_id]
        updated = {}
        for k in params:
            # EMA of squared gradients
            state[k] = self.beta * state[k] + (1 - self.beta) * grads[k] ** 2
            # Adapted update
            updated[k] = params[k] - (self.lr / (np.sqrt(state[k]) + self.epsilon)) * grads[k]
        return updated

    def __repr__(self):
        return f"RMSProp(lr={self.lr}, beta={self.beta}, epsilon={self.epsilon})"


# ══════════════════════════════════════════════════════════════════════════ #
#  Adam                                                                      #
# ══════════════════════════════════════════════════════════════════════════ #

class Adam(BaseOptimizer):
    """
    Adaptive Moment Estimation.

    Combines Momentum (first moment — direction smoothing) with RMSProp
    (second moment — per-parameter learning rate scaling). Then applies
    bias correction to compensate for both moments being initialised at 0,
    which would otherwise cause small steps at the start of training.

        m  ← β₁·m  + (1−β₁)·g            first moment  (velocity)
        v  ← β₂·v  + (1−β₂)·g²           second moment (squared grad EMA)

        m̂  = m  / (1 − β₁ᵗ)              bias-corrected first moment
        v̂  = v  / (1 − β₂ᵗ)              bias-corrected second moment

        w  ← w − lr · m̂ / (√v̂ + ε)

    Why bias correction?
    --------------------
    At step t=1, m = (1-β₁)·g which is much smaller than g since β₁=0.9.
    Dividing by (1-β₁¹) = 0.1 rescales it back to approximately g.
    As t grows, β₁ᵗ → 0 and the correction factor → 1 (no effect).
    Without correction, the first few hundred steps take tiny steps
    regardless of how large the gradient is — bias correction fixes this.

    Parameters
    ----------
    lr      : learning rate (default 0.001 — works well across most tasks)
    beta1   : first moment decay  (default 0.9   — standard)
    beta2   : second moment decay (default 0.999 — standard)
    epsilon : numerical stability (default 1e-8  — standard)

    State per parameter: 2 arrays (m first moment, v second moment)
    + 1 global timestep counter self._t

    Adam is the recommended default optimizer for most deep learning tasks.
    """

    def __init__(
        self,
        lr:      float = 0.001,
        beta1:   float = 0.9,
        beta2:   float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.lr      = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self._state  = {}
        self._t      = 0     # global timestep — shared across all layers

    def _init_state(self, layer_id, params):
        self._state[layer_id] = {
            k: {"m": np.zeros_like(v), "v": np.zeros_like(v)}
            for k, v in params.items()
        }

    def update(self, layer_id: int, layer) -> None:
        """
        Override update() to increment the global timestep once per
        optimizer.update() call, not once per parameter.
        All layers in one training step share the same t.
        """
        params = layer.get_params()
        if not params:
            return
        grads = self._get_grads(layer, params)
        if layer_id not in self._state:
            self._init_state(layer_id, params)
        updated = self.step(layer_id, params, grads)
        layer.set_params(updated)

    def step(self, layer_id, params, grads):
        state   = self._state[layer_id]
        updated = {}
        for k in params:
            s = state[k]
            # Update biased first and second moment estimates
            s["m"] = self.beta1 * s["m"] + (1 - self.beta1) * grads[k]
            s["v"] = self.beta2 * s["v"] + (1 - self.beta2) * grads[k] ** 2
            # Bias-corrected estimates
            m_hat = s["m"] / (1 - self.beta1 ** self._t)
            v_hat = s["v"] / (1 - self.beta2 ** self._t)
            # Parameter update
            updated[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated

    def __repr__(self):
        return (f"Adam(lr={self.lr}, beta1={self.beta1}, "
                f"beta2={self.beta2}, epsilon={self.epsilon})")