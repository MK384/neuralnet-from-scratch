"""
network.py
----------
Network class — stacks any mix of layer types and runs the training loop.
"""

import numpy as np
from layer import DenseLayer

class Network:
    """
    A neural network that accepts any sequence of layer types:
    Conv2D, MaxPool, Flatten, Dense — in any valid order.

    Usage — CNN for image classification
    -------------------------------------
        net = Network(
            input_shape=(1, 28, 28),
            layers=[
                Layer.Conv2D(8,  kernel_size=3, activation=Activation("relu")),
                Layer.MaxPool(pool_size=2),
                Layer.Flatten(),
                Layer.Dense(64, activation=Activation("relu")),
                Layer.Dense(10, activation=Activation("softmax")),
            ],
            loss="cross_entropy"
        )

    Usage — MLP (backward compatible)
    -----------------------------------
        net = Network(
            input_shape=(784,),
            layers=[
                Layer.Dense(128, activation=Activation("relu")),
                Layer.Dense(10,  activation=Activation("softmax")),
            ],
            loss="cross_entropy"
        )

    input_shape
    -----------
    1-D inputs (MLP):  (features,)           e.g. (784,)
    2-D image inputs:  (channels, H, W)      e.g. (1, 28, 28)
    """

    SUPPORTED_LOSSES = ("mse", "cross_entropy")

    def __init__(self, input_shape: tuple, layers: list, loss: str = "mse"):
        if loss not in self.SUPPORTED_LOSSES:
            raise ValueError(
                f"Unknown loss '{loss}'. Choose from: {self.SUPPORTED_LOSSES}"
            )
        self.loss = loss

        # ── Build layers by chaining shapes ───────────────────────────────
        self.layers   = []
        current_shape = input_shape   # updated after each layer is built

        for spec in layers:
            layer         = spec.build(current_shape)
            current_shape = layer.output_shape
            self.layers.append(layer)

        # ── Validate cross-entropy requires softmax output ─────────────────
        if self.loss == "cross_entropy":
            last = self.layers[-1]
            if not (isinstance(last, DenseLayer) and
                    last.activation.name.lower() == "softmax"):
                raise ValueError(
                    "loss='cross_entropy' requires the last layer to be "
                    "Dense with Activation('softmax')."
                )

    # ---------------------------------------------------------------------- #
    # Forward pass                                                            #
    # ---------------------------------------------------------------------- #

    def forward(self, X: np.ndarray) -> np.ndarray:
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    # ---------------------------------------------------------------------- #
    # Cost functions                                                          #
    # ---------------------------------------------------------------------- #

    def _cost(self, y_pred, y_true):
        if self.loss == "mse":
            return np.mean((y_pred - y_true) ** 2)
        y_c = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        return -np.mean(np.sum(y_true * np.log(y_c), axis=-1))

    def _cost_gradient(self, y_pred, y_true):
        if self.loss == "mse":
            return 2 * (y_pred - y_true) / y_pred.shape[0]
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        return (y_pred - y_true) / N

    # ---------------------------------------------------------------------- #
    # Backward pass                                                           #
    # ---------------------------------------------------------------------- #

    def backward(self, y_pred, y_true):
        delta = self._cost_gradient(y_pred, y_true)
        for i, layer in enumerate(reversed(self.layers)):
            is_output = (i == 0)
            if is_output and self.loss == "cross_entropy":
                delta = layer.backward_output_crossentropy(delta)
            else:
                delta = layer.backward(delta)

    # ---------------------------------------------------------------------- #
    # Parameter update                                                        #
    # ---------------------------------------------------------------------- #

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    # ---------------------------------------------------------------------- #
    # Training loop                                                           #
    # ---------------------------------------------------------------------- #

    def train(self, X, y, epochs, lr, batch_size=32, verbose=True):
        """
        Mini-batch gradient descent.

        X : (N, features) for MLP  |  (N, C, H, W) for CNN
        y : (N, n_classes) one-hot
        """
        N = X.shape[0]
        history = []

        for epoch in range(1, epochs + 1):
            idx        = np.random.permutation(N)
            X_s, y_s   = X[idx], y[idx]
            epoch_cost = 0.0
            n_batches  = 0

            for start in range(0, N, batch_size):
                end     = min(start + batch_size, N)
                xb, yb  = X_s[start:end], y_s[start:end]
                yp      = self.forward(xb)
                epoch_cost += self._cost(yp, yb)
                self.backward(yp, yb)
                self.update(lr)
                n_batches += 1

            avg = epoch_cost / n_batches
            history.append(avg)
            if verbose:
                print(f"Epoch {epoch:>4}/{epochs}  cost: {avg:.6f}")

        return history

    # ---------------------------------------------------------------------- #
    # Inference                                                               #
    # ---------------------------------------------------------------------- #

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, X, y):
        return np.mean(self.predict(X) == np.argmax(y, axis=1))

    # ---------------------------------------------------------------------- #
    # Save / Load                                                             #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """
        Save all learnable parameters to a .npz file.
        Parameterless layers (MaxPool, Flatten) contribute nothing to the file.
        """
        params = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.get_params().items():
                params[f"{i}_{k}"] = v

        np.savez(path, **params)
        display = path if path.endswith(".npz") else path + ".npz"
        n_params = sum(v.size for v in params.values())
        print(f"Saved '{display}'  ({n_params:,} parameters)")

    def load(self, path: str) -> None:
        """Restore learnable parameters from a .npz file."""
        display = path if path.endswith(".npz") else path + ".npz"
        data    = np.load(display)

        for i, layer in enumerate(self.layers):
            expected_keys = list(layer.get_params().keys())
            if not expected_keys:
                continue
            layer.set_params({k: data[f"{i}_{k}"] for k in expected_keys})

        n_params = sum(v.size for v in data.values())
        print(f"Loaded '{display}'  ({n_params:,} parameters)")

    # ---------------------------------------------------------------------- #
    # Utility                                                                 #
    # ---------------------------------------------------------------------- #

    def summary(self):
        total = 0
        print("=" * 62)
        print(f"  Loss: {self.loss}")
        print("=" * 62)
        print(f"  {'Layer':<20} {'Output shape':<18} {'Params':>7}")
        print("-" * 62)
        for layer in self.layers:
            params = sum(v.size for v in layer.get_params().values())
            total += params
            print(f"  {layer.__class__.__name__:<20} "
                  f"{str(layer.output_shape):<18} {params:>7}")
        print("=" * 62)
        print(f"  {'Total parameters':<38} {total:>7}")
        print("=" * 62)