"""
network.py — Network class with pluggable optimizer support.
"""

import numpy as np
from .layer import DenseLayer
from .optimizer import SGD, BaseOptimizer


class Network:
    """
    Neural network supporting any combination of layer types and optimizers.

    Usage
    -----
        from optimizer import Adam, Momentum, RMSProp, AdaGrad, SGD

        net = Network(
            input_shape=(784,),
            layers=[
                Layer.Dense(128, activation=Activation("relu")),
                Layer.BatchNorm(),
                Layer.Dropout(rate=0.3),
                Layer.Dense(10,  activation=Activation("softmax")),
            ],
            loss="cross_entropy",
            optimizer=Adam(lr=0.001)     ← plug in any optimizer here
        )

        net.train(X_train, y_train, epochs=30, batch_size=32)

    Note: lr is now owned by the optimizer, not by train().
    train() no longer accepts an lr argument.
    """

    SUPPORTED_LOSSES = ("mse", "cross_entropy")

    def __init__(
        self,
        input_shape: tuple,
        layers:      list,
        loss:        str           = "mse",
        optimizer:   BaseOptimizer = None,
    ):
        if loss not in self.SUPPORTED_LOSSES:
            raise ValueError(
                f"Unknown loss '{loss}'. Choose from {self.SUPPORTED_LOSSES}."
            )
        self.loss      = loss
        self.optimizer = optimizer if optimizer is not None else SGD(lr=0.01)
        self.training  = True

        # Build layers by chaining shapes
        self.layers    = []
        current_shape  = input_shape
        for spec in layers:
            layer         = spec.build(current_shape)
            current_shape = layer.output_shape
            self.layers.append(layer)

        # Validate cross-entropy requires softmax output
        if self.loss == "cross_entropy":
            last = self.layers[-1]
            if not (isinstance(last, DenseLayer) and
                    last.activation.name.lower() == "softmax"):
                raise ValueError(
                    "loss='cross_entropy' requires the last layer to be "
                    "Dense with Activation('softmax')."
                )

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #

    def forward(self, X: np.ndarray) -> np.ndarray:
        a = X
        for layer in self.layers:
            a = layer.forward(a, training=self.training)
        return a

    # ---------------------------------------------------------------------- #
    # Cost                                                                    #
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
    # Backward                                                                #
    # ---------------------------------------------------------------------- #

    def backward(self, y_pred, y_true) -> None:
        delta = self._cost_gradient(y_pred, y_true)
        for i, layer in enumerate(reversed(self.layers)):
            is_output = (i == 0)
            if is_output and self.loss == "cross_entropy":
                delta = layer.backward_output_crossentropy(delta)
            else:
                delta = layer.backward(delta)

    # ---------------------------------------------------------------------- #
    # Update — delegates to optimizer                                         #
    # ---------------------------------------------------------------------- #

    def update(self) -> None:
        """
        Apply one optimizer step to every layer.

        For Adam, we increment the global timestep once per update()
        call (once per batch), not once per layer. This ensures all
        layers share the same t for bias correction.
        """
        # Increment Adam timestep before any layer is updated
        if hasattr(self.optimizer, '_t'):
            self.optimizer._t += 1

        for i, layer in enumerate(self.layers):
            self.optimizer.update(i, layer)

    # ---------------------------------------------------------------------- #
    # Training loop                                                           #
    # ---------------------------------------------------------------------- #

    def train(
        self,
        X:          np.ndarray,
        y:          np.ndarray,
        epochs:     int,
        batch_size: int  = 32,
        verbose:    bool = True,
    ) -> list:
        """
        Mini-batch training loop.

        Note: lr is owned by the optimizer passed to __init__.
              train() no longer accepts an lr argument.

        Parameters
        ----------
        X          : (N, features) or (N, C, H, W)
        y          : (N, n_classes) one-hot
        epochs     : number of full passes through the dataset
        batch_size : samples per gradient update
        verbose    : print cost every epoch if True

        Returns
        -------
        history : list of per-epoch average costs
        """
        self.training = True
        N = X.shape[0]
        history = []

        for epoch in range(1, epochs + 1):
            idx       = np.random.permutation(N)
            X_s, y_s  = X[idx], y[idx]
            epoch_cost = 0.0
            n_batches  = 0

            for start in range(0, N, batch_size):
                end    = min(start + batch_size, N)
                xb, yb = X_s[start:end], y_s[start:end]
                yp     = self.forward(xb)
                epoch_cost += self._cost(yp, yb)
                self.backward(yp, yb)
                self.update()                          # ← no lr argument
                n_batches += 1

            avg = epoch_cost / n_batches
            history.append(avg)
            if verbose:
                print(f"Epoch {epoch:>4}/{epochs}  "
                      f"cost: {avg:.6f}  "
                      f"[{self.optimizer}]")

        return history

    # ---------------------------------------------------------------------- #
    # Inference                                                               #
    # ---------------------------------------------------------------------- #

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.training = False
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        self.training = False
        return np.mean(self.predict(X) == np.argmax(y, axis=1))

    # ---------------------------------------------------------------------- #
    # Save / Load                                                             #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        params = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.get_params().items():
                params[f"{i}_{k}"] = v
        np.savez(path, **params)
        display  = path if path.endswith(".npz") else path + ".npz"
        n_params = sum(v.size for v in params.values())
        print(f"Saved '{display}'  ({n_params:,} parameters)")

    def load(self, path: str) -> None:
        display = path if path.endswith(".npz") else path + ".npz"
        data    = np.load(display)
        for i, layer in enumerate(self.layers):
            keys = list(layer.get_params().keys())
            if keys:
                layer.set_params({k: data[f"{i}_{k}"] for k in keys})
        n_params = sum(v.size for v in data.values())
        print(f"Loaded '{display}'  ({n_params:,} parameters)")

    # ---------------------------------------------------------------------- #
    # Utility                                                                 #
    # ---------------------------------------------------------------------- #

    def summary(self) -> None:
        total = 0
        print("=" * 66)
        print(f"  Loss:      {self.loss}")
        print(f"  Optimizer: {self.optimizer}")
        print("=" * 66)
        print(f"  {'Layer':<22} {'Output shape':<18} {'Params':>7}")
        print("-" * 66)
        for layer in self.layers:
            params = sum(v.size for v in layer.get_params().values())
            total += params
            name   = layer.__class__.__name__.replace("Layer", "")
            print(f"  {name:<22} {str(layer.output_shape):<18} {params:>7}")
        print("=" * 66)
        print(f"  {'Total parameters':<40} {total:>7}")
        print("=" * 66)

    def __repr__(self) -> str:
        layers_str = " → ".join(
            l.__class__.__name__.replace("Layer", "") for l in self.layers
        )
        total = sum(
            sum(v.size for v in l.get_params().values()) for l in self.layers
        )
        return (f"Network([{layers_str}], loss={self.loss}, "
                f"optimizer={self.optimizer}, params={total})")