import numpy as np
from NN import Layer , _DenseSpec


class Network:
    """
    A fully-connected feedforward neural network.

    Usage — MSE + Sigmoid (default)
    --------------------------------
        net = Network(
            input_size=784,
            layers=[
                Layer.Dense(16, activation=Activation("sigmoid")),
                Layer.Dense(16, activation=Activation("sigmoid")),
                Layer.Dense(10, activation=Activation("sigmoid")),
            ]
        )

    Usage — Cross-Entropy + Softmax
    --------------------------------
        net = Network(
            input_size=784,
            layers=[
                Layer.Dense(16, activation=Activation("relu")),
                Layer.Dense(16, activation=Activation("relu")),
                Layer.Dense(10, activation=Activation("softmax")),
            ],
            loss="cross_entropy"
        )

    Supported loss values
    ---------------------
        "mse"            — Mean Squared Error  (default)
        "cross_entropy"  — Categorical Cross-Entropy (requires Softmax output)
    """

    SUPPORTED_LOSSES = ("mse", "cross_entropy")

    def __init__(self, input_size: int, layers: list, loss: str = "mse"):
        """
        Parameters
        ----------
        input_size : number of input features (e.g. 784 for MNIST)
        layers     : list of Layer.Dense(...) specs, ordered input → output
        loss       : "mse" or "cross_entropy"
        """
        # ── Validate loss ─────────────────────────────────────────────────
        if loss not in self.SUPPORTED_LOSSES:
            raise ValueError(
                f"Unknown loss '{loss}'. "
                f"Choose from: {self.SUPPORTED_LOSSES}"
            )
        self.loss = loss

        # ── Build layers from specs ───────────────────────────────────────
        # Each spec knows n_out and activation but not n_in.
        # We chain sizes: input_size → spec[0].n_out → spec[1].n_out → ...
        self.layers = []
        n_in = input_size
        for spec in layers:
            if isinstance(spec, _DenseSpec):
                layer = spec.build(n_in)
            elif isinstance(spec, Layer):
                # Allow pre-built Layer objects too, for flexibility
                layer = spec
            else:
                raise TypeError(
                    f"Expected Layer.Dense(...) or Layer, got {type(spec)}. "
                    f"Use Layer.Dense(n_out, activation=...) to define layers."
                )
            self.layers.append(layer)
            n_in = layer.W.shape[0]   # this layer's n_out becomes next layer's n_in

        # ── Validate cross-entropy requires Softmax output ─────────────────
        if self.loss == "cross_entropy":
            last_act = self.layers[-1].activation.name.lower()
            if last_act != "softmax":
                raise ValueError(
                    f"loss='cross_entropy' requires the last layer to use "
                    f"Activation('softmax'), but got Activation('{last_act}'). "
                    f"Either change the last layer's activation to 'softmax' "
                    f"or switch to loss='mse'."
                )

    # ---------------------------------------------------------------------- #
    # Forward pass                                                            #
    # ---------------------------------------------------------------------- #

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Run a full forward pass through all layers.

        Parameters
        ----------
        X : shape (n_in,) for a single sample or (batch, n_in) for a batch

        Returns
        -------
        Output activations of the last layer.
        shape (n_out,) or (batch, n_out)
        """
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    # ---------------------------------------------------------------------- #
    # Cost functions                                                          #
    # ---------------------------------------------------------------------- #

    def _cost(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.loss == "mse":
            return self._mse(y_pred, y_true)
        return self._cross_entropy(y_pred, y_true)

    def _cost_gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        if self.loss == "mse":
            return self._mse_prime(y_pred, y_true)
        return self._cross_entropy_softmax_gradient(y_pred, y_true)

    # ── MSE ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def _mse_prime(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_pred.shape[0]

    # ── Cross-Entropy ─────────────────────────────────────────────────────

    @staticmethod
    def _cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Categorical cross-entropy:
            CE = -(1/N) Σ Σ y_true * log(y_pred)

        Clipping y_pred prevents log(0) = -inf.
        We clip to [1e-12, 1-1e-12] — small enough to never affect
        the result meaningfully, large enough to keep log finite.
        """
        y_pred_clipped = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=-1))

    @staticmethod
    def _cross_entropy_softmax_gradient(
        y_pred: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        """
        Combined gradient of CrossEntropy loss with respect to the
        pre-activation z of the Softmax output layer.

        Derivation
        ----------
        Normally backprop needs two steps at the output layer:
            ∂CE/∂a  (cost w.r.t. softmax output)
            ∂a/∂z   (softmax jacobian)

        When you multiply them together, almost everything cancels and
        you get this beautifully simple result:

            ∂CE/∂z = y_pred - y_true   (divided by N for the batch mean)

        This is one of the great mathematical gifts of pairing Softmax
        with Cross-Entropy. The gradient is cleaner than MSE+Sigmoid.
        """
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        return (y_pred - y_true) / N

    # ---------------------------------------------------------------------- #
    # Backward pass                                                           #
    # ---------------------------------------------------------------------- #

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Run backpropagation through all layers in reverse order.

        For MSE: uses the standard backward() on every layer.
        For cross-entropy: injects the combined gradient at the output
        layer via backward_output_crossentropy(), then uses standard
        backward() for all hidden layers.
        """
        delta = self._cost_gradient(y_pred, y_true)

        for i, layer in enumerate(reversed(self.layers)):
            is_output_layer = (i == 0)

            if is_output_layer and self.loss == "cross_entropy":
                # Combined Softmax+CE gradient — skip fn_prime
                delta = layer.backward_output_crossentropy(delta)
            else:
                delta = layer.backward(delta)

    # ---------------------------------------------------------------------- #
    # Parameter update                                                        #
    # ---------------------------------------------------------------------- #

    def update(self, lr: float) -> None:
        for layer in self.layers:
            layer.update(lr)

    # ---------------------------------------------------------------------- #
    # Training loop                                                           #
    # ---------------------------------------------------------------------- #

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list:
        """
        Mini-batch gradient descent training loop.

        Parameters
        ----------
        X          : training inputs,  shape (N, n_in)
        y          : training labels,  shape (N, n_out) — one-hot encoded
        epochs     : number of full passes through the dataset
        lr         : learning rate η
        batch_size : samples per gradient update
        verbose    : print cost every epoch if True

        Returns
        -------
        history : list of per-epoch average costs
        """
        N = X.shape[0]
        history = []

        for epoch in range(1, epochs + 1):
            indices    = np.random.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_cost = 0.0
            n_batches  = 0

            for start in range(0, N, batch_size):
                end        = min(start + batch_size, N)
                X_batch    = X_shuffled[start:end]
                y_batch    = y_shuffled[start:end]

                y_pred     = self.forward(X_batch)
                batch_cost = self._cost(y_pred, y_batch)
                self.backward(y_pred, y_batch)
                self.update(lr)

                epoch_cost += batch_cost
                n_batches  += 1

            avg_cost = epoch_cost / n_batches
            history.append(avg_cost)

            if verbose:
                print(f"Epoch {epoch:>4}/{epochs}  cost: {avg_cost:.6f}")

        return history

    # ---------------------------------------------------------------------- #
    # Inference                                                               #
    # ---------------------------------------------------------------------- #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class index for each sample. Shape (N,)."""
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy as a float in [0, 1]."""
        return np.mean(self.predict(X) == np.argmax(y, axis=1))

    # ---------------------------------------------------------------------- #
    # Save / Load                                                             #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """
        Save all learned parameters (weights and biases) to a .npz file.

        What gets saved
        ---------------
        For each layer i we store two arrays:
            "W_{i}" — weight matrix, shape (n_out, n_in)
            "b_{i}" — bias vector,   shape (n_out,)

        What does NOT get saved
        -----------------------
        The architecture (layer sizes, activations, loss type) is NOT saved.
        It lives in your code where you define the Network. This is a
        deliberate choice — mixing architecture with weights in one file
        creates a dependency on Python's pickle format, which breaks across
        versions and is a security risk. Keeping them separate means:
            - weights file is just numbers, readable by any tool
            - architecture is version-controlled alongside your code

        Parameters
        ----------
        path : file path, e.g. "mnist_model.npz"
               .npz extension is added automatically if omitted.

        Usage
        -----
            net.save("mnist_model")          # saves to mnist_model.npz
            net.save("models/checkpoint")    # saves to models/checkpoint.npz
        """
        params = {}
        for i, layer in enumerate(self.layers):
            params[f"W_{i}"] = layer.W
            params[f"b_{i}"] = layer.b

        np.savez(path, **params)
        # np.savez appends .npz if not present — normalise for the message
        display_path = path if path.endswith(".npz") else path + ".npz"
        print(f"Model saved to '{display_path}'  "
              f"({len(self.layers)} layers, "
              f"{sum(l.W.size + l.b.size for l in self.layers):,} parameters)")

    def load(self, path: str) -> None:
        """
        Load parameters from a .npz file into the current network.

        The network architecture must match the one that was saved —
        same number of layers, same shapes. If they don't match, numpy
        will raise a clear shape error when assigning W and b.

        Parameters
        ----------
        path : file path, e.g. "mnist_model.npz"
               .npz extension is added automatically if omitted.

        Usage
        -----
            # Rebuild the same architecture, then load weights:
            net = Network(
                input_size=784,
                layers=[
                    Layer.Dense(128, activation=Activation("relu")),
                    Layer.Dense(64,  activation=Activation("relu")),
                    Layer.Dense(10,  activation=Activation("softmax")),
                ],
                loss="cross_entropy"
            )
            net.load("mnist_model")   # restores trained weights instantly
            net.predict(x_test)       # ready to use, no training needed
        """
        display_path = path if path.endswith(".npz") else path + ".npz"
        data = np.load(display_path)

        # Validate that the file has the right number of layers
        n_saved = sum(1 for k in data if k.startswith("W_"))
        if n_saved != len(self.layers):
            raise ValueError(
                f"Architecture mismatch: file has {n_saved} layers "
                f"but this network has {len(self.layers)} layers. "
                f"Rebuild the network with the same architecture before loading."
            )

        for i, layer in enumerate(self.layers):
            W_saved = data[f"W_{i}"]
            b_saved = data[f"b_{i}"]

            # Validate shape match before assigning
            if W_saved.shape != layer.W.shape:
                raise ValueError(
                    f"Shape mismatch at layer {i}: "
                    f"file has W shape {W_saved.shape} "
                    f"but network expects {layer.W.shape}."
                )

            layer.W = W_saved
            layer.b = b_saved

        print(f"Model loaded from '{display_path}'  "
              f"({len(self.layers)} layers, "
              f"{sum(l.W.size + l.b.size for l in self.layers):,} parameters)")

    # ---------------------------------------------------------------------- #
    # Utility                                                                 #
    # ---------------------------------------------------------------------- #

    def summary(self) -> None:
        total_params = 0
        print("=" * 58)
        print(f"  Loss: {self.loss}")
        print("=" * 58)
        print(f"{'Layer':<8} {'Shape (n_in→n_out)':<22} {'Activation':<12} {'Params':>6}")
        print("-" * 58)
        for i, layer in enumerate(self.layers):
            n_in   = layer.W.shape[1]
            n_out  = layer.W.shape[0]
            params = layer.W.size + layer.b.size
            total_params += params
            label  = "output" if i == len(self.layers) - 1 else f"hidden {i+1}"
            print(f"  {label:<6} {str(n_in)+' → '+str(n_out):<22} "
                  f"{layer.activation.name:<12} {params:>6}")
        print("=" * 58)
        print(f"{'Total parameters':<48} {total_params:>6}")
        print("=" * 58)

    def __repr__(self) -> str:
        layer_str = " → ".join(
            f"{l.W.shape[1]}({l.activation.name})" for l in self.layers
        ) + f" → {self.layers[-1].W.shape[0]}"
        total = sum(l.W.size + l.b.size for l in self.layers)
        return f"Network({layer_str}, loss={self.loss}, total_params={total})"