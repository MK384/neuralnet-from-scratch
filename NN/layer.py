import numpy as np
from NN import Activation

class Layer:

    """
    A fully-connected (dense) layer.

    Do not instantiate directly. Use the class methods:
        Layer.Dense(n_out, activation=Activation("relu"))

    This keeps the door open for future layer types (Conv, Dropout, etc.)
    without breaking the existing API.
    """

    def __init__(self, n_in: int, n_out: int, activation: Activation):
        """
        Internal constructor. Called by Layer.Dense() after n_in is
        resolved by Network from the previous layer's n_out.
        """
        self.activation = activation
        self.W  = self.activation.init_fn(n_out, n_in)  # (n_out, n_in)
        self.b  = np.zeros(n_out)                        # (n_out,)
        self._a_in = None
        self._z    = None
        self.dW    = None
        self.db    = None

    # ------------------------------------------------------------------ #
    # Public factory — the only way to create a layer from user code      #
    # ------------------------------------------------------------------ #

    @classmethod
    def Dense(cls, n_out: int, activation: Activation = None) -> "_DenseSpec":
        """
        Declare a dense layer with n_out neurons and an activation.

        n_in is NOT required here — Network infers it automatically
        from the previous layer (or from input_size for the first layer).

        Returns a _DenseSpec, a lightweight spec object that Network
        uses to build the real Layer once it knows n_in.

        Example
        -------
            Layer.Dense(16, activation=Activation("relu"))
            Layer.Dense(10, activation=Activation("softmax"))
        """
        act = activation if activation is not None else Activation("sigmoid")
        return _DenseSpec(n_out=n_out, activation=act)

    # ------------------------------------------------------------------ #
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #

    def forward(self, a_in: np.ndarray) -> np.ndarray:
        self._a_in = a_in
        self._z    = a_in @ self.W.T + self.b
        return self.activation.fn(self._z)

    # ------------------------------------------------------------------ #
    # Backward pass                                                        #
    # ------------------------------------------------------------------ #

    def backward(self, delta_in: np.ndarray) -> np.ndarray:
        """
        General backward pass used for all hidden layers.

        For the output layer with cross-entropy + softmax, Network
        bypasses this and injects the combined gradient directly,
        so fn_prime is never called on a Softmax layer.
        """
        delta = delta_in * self.activation.fn_prime(self._z)

        if delta.ndim == 1:         # 1 sample
            self.dW = np.outer(delta, self._a_in)
        else:                       # a batch of samples
            self.dW = (delta.T @ self._a_in) / delta.shape[0]

        self.db = delta.mean(axis=0) if delta.ndim > 1 else delta
        return delta @ self.W

    def backward_output_crossentropy(self, delta_in: np.ndarray) -> np.ndarray:
        """
        Specialised backward for the output layer when using cross-entropy.

        The combined Softmax+CrossEntropy gradient is:
            ∂(CE)/∂z = y_pred - y_true   (already divided by N)

        This is passed in directly as delta_in — no fn_prime needed.
        We skip straight to computing dW and db and passing delta back.
        """
        delta = delta_in   # already the correct gradient, no fn_prime step

        if delta.ndim == 1:
            self.dW = np.outer(delta, self._a_in)
        else:
            self.dW = (delta.T @ self._a_in) / delta.shape[0]

        self.db = delta.mean(axis=0) if delta.ndim > 1 else delta
        return delta @ self.W

    # ------------------------------------------------------------------ #
    # Parameter update                                                   #
    # ------------------------------------------------------------------ #

    def update(self, lr: float) -> None:
        self.W -= lr * self.dW
        self.b -= lr * self.db

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (f"Layer(n_in={self.W.shape[1]}, n_out={self.W.shape[0]}, "
                f"activation={self.activation.name}, "
                f"params={self.W.size + self.b.size})")


class _DenseSpec:
    """
    Lightweight specification object returned by Layer.Dense().

    Holds n_out and activation but NOT n_in — that is filled in by
    Network once it knows the output size of the previous layer.
    Users never interact with this directly.
    """
    def __init__(self, n_out: int, activation: Activation):
        self.n_out      = n_out
        self.activation = activation

    def build(self, n_in: int) -> Layer:
        """Materialise into a real Layer once n_in is known."""
        return Layer(n_in=n_in, n_out=self.n_out, activation=self.activation)

    def __repr__(self) -> str:
        return f"DenseSpec(n_out={self.n_out}, activation={self.activation.name})"
