import numpy as np


class Activation:
    """
    Bundles an activation function, its derivative, and its weight
    initialiser together.

    Two ways to create an instance
    --------------------------------
    1. By name — recommended, picks up a built-in:
           act = Activation("relu")
           act = Activation("sigmoid")
           act = Activation("tanh")
           act = Activation("softmax")   ← new

    2. Custom — supply your own functions:
           act = Activation(fn=my_fn, fn_prime=my_fn_prime,
                            init_fn=my_init, name="custom")
           If init_fn is omitted, defaults to Xavier.

    Names are case-insensitive: "ReLU", "relu", "RELU" all work.

    Note on Softmax
    ---------------
    Softmax is special: its derivative is a Jacobian matrix, not a
    simple element-wise operation. However, when paired with
    cross-entropy loss, the combined gradient simplifies beautifully
    to just (y_pred - y_true). Network handles this cancellation
    directly and never calls softmax.fn_prime during backprop.
    We still store a placeholder fn_prime so the object is uniform.
    """

    # ------------------------------------------------------------------ #
    # Built-in activation functions (private static methods)              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_prime(z: np.ndarray) -> np.ndarray:
        s = Activation._sigmoid(z)
        return s * (1.0 - s)

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_prime(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    @staticmethod
    def _tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def _tanh_prime(z: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(z) ** 2

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability — prevents e^z overflow.
        # Subtracting a constant from every element doesn't change the
        # output because it cancels in the numerator and denominator.
        # Works for both single sample (1D) and batch (2D).
        z_stable = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z_stable)
        return e / np.sum(e, axis=-1, keepdims=True)

    @staticmethod
    def _softmax_prime(z: np.ndarray) -> np.ndarray:
        # Placeholder — never called when loss="cross_entropy".
        # Network short-circuits the gradient to (y_pred - y_true) / N
        # which is the combined Softmax + CrossEntropy gradient.
        raise NotImplementedError(
            "Softmax derivative is not used directly. "
            "Use loss='cross_entropy' in Network, which applies the "
            "combined Softmax+CrossEntropy gradient automatically."
        )

    # ------------------------------------------------------------------ #
    # Built-in weight initialisers (private static methods)               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _xavier_init(n_out: int, n_in: int) -> np.ndarray:
        std = np.sqrt(2.0 / (n_in + n_out))
        return np.random.randn(n_out, n_in) * std

    @staticmethod
    def _he_init(n_out: int, n_in: int) -> np.ndarray:
        std = np.sqrt(2.0 / n_in)
        return np.random.randn(n_out, n_in) * std

    # ------------------------------------------------------------------ #
    # Registry — maps lowercase name → (fn, fn_prime, init_fn)           #
    # Adding a new built-in = one new entry here. Nothing else changes.  #
    # ------------------------------------------------------------------ #

    _REGISTRY = {
        "sigmoid": (_sigmoid.__func__, _sigmoid_prime.__func__, _xavier_init.__func__),
        "relu":    (_relu.__func__,    _relu_prime.__func__,    _he_init.__func__),
        "tanh":    (_tanh.__func__,    _tanh_prime.__func__,    _xavier_init.__func__),
        "softmax": (_softmax.__func__, _softmax_prime.__func__, _xavier_init.__func__),
    }

    # ------------------------------------------------------------------ #
    # Constructor                                                          #
    # ------------------------------------------------------------------ #

    def __init__(self, name: str = None, fn=None, fn_prime=None, init_fn=None):
        if fn is not None and fn_prime is not None:
            self.fn       = fn
            self.fn_prime = fn_prime
            self.init_fn  = init_fn if init_fn is not None else self._xavier_init
            self.name     = name or "custom"
        elif name is not None:
            key = name.lower()
            if key not in self._REGISTRY:
                available = ", ".join(self._REGISTRY.keys())
                raise ValueError(
                    f"Unknown activation '{name}'. "
                    f"Available built-ins: {available}. "
                    f"For a custom activation, pass fn and fn_prime."
                )
            self.fn, self.fn_prime, self.init_fn = self._REGISTRY[key]
            self.name = key.capitalize()
        else:
            raise ValueError(
                "Provide a built-in name (e.g. Activation('relu')) "
                "or custom functions (Activation(fn=..., fn_prime=..., name=...))."
            )

    def __repr__(self) -> str:
        return f"Activation({self.name})"
