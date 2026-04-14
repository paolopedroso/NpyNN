import numpy as np
from abc import abstractmethod


class Optimizer:
    """
    Abstract base class for all optimizers.
    """

    @abstractmethod
    def update_params(self, layer):
        """
        Updates the weights and biases of a single trainable layer.

        Args:
            layer: A trainable layer with weights, biases, dweights, and dbiases attributes.
        """
        pass

    def pre_update_params(self):
        """
        Applies inverse decay to the learning rate before each update step.
        Only runs if a non-zero decay value was set.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def post_update_params(self):
        """
        Called after all parameter updates. Increments the iteration counter
        used for learning rate decay and bias correction.
        """
        self.iterations += 1


class Adam(Optimizer):
    """
    Adam optimizer. Combines momentum and RMSprop-style adaptive learning rates,
    with bias correction applied to both the first and second moment estimates.

    Well suited as a general-purpose optimizer for most neural network tasks.
    """

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        """
        Initializes Adam optimizer hyperparameters.

        Args:
            learning_rate (float, optional): Initial learning rate. Default is 0.001.
            decay (float, optional): Learning rate decay applied each step. Default is 0.
            epsilon (float, optional): Small constant for numerical stability in
                                       the denominator. Default is 1e-7.
            beta_1 (float, optional): Exponential decay rate for the first moment
                                      (momentum) estimate. Default is 0.9.
            beta_2 (float, optional): Exponential decay rate for the second moment
                                      (cache) estimate. Default is 0.999.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer):
        """
        Applies the Adam update rule to a layer's weights and biases.

        Initializes per-layer momentum and cache arrays on the first call.
        Computes bias-corrected first and second moment estimates, then
        updates parameters using the scaled gradient.

        Args:
            layer: A trainable layer with weights, biases, dweights, and dbiases attributes.
        """
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentum = np.zeros_like(layer.biases)

        layer.weight_momentum = self.beta_1 * layer.weight_momentum + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentum = self.beta_1 * layer.bias_momentum + \
            (1 - self.beta_1) * layer.dbiases

        weight_momentum_corrected = layer.weight_momentum / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentum_corrected = layer.bias_momentum / \
            (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights -= (self.current_learning_rate * weight_momentum_corrected) / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= (self.current_learning_rate * bias_momentum_corrected) / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum and learning rate decay.

    Without momentum, parameters are updated by subtracting the scaled gradient directly.
    With momentum, a velocity term accumulates and smooths updates across steps.
    """

    def __init__(self, learning_rate=1e-1, decay=0., momentum=0):
        """
        Initializes SGD optimizer hyperparameters.

        Args:
            learning_rate (float, optional): Initial learning rate. Default is 0.1.
            decay (float, optional): Learning rate decay applied each step. Default is 0.
            momentum (float, optional): Momentum coefficient. Set to 0 to disable. Default is 0.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def update_params(self, layer):
        """
        Applies the SGD update rule to a layer's weights and biases.

        If momentum is enabled, initializes per-layer velocity arrays on the
        first call and accumulates gradients over steps. Otherwise performs a
        standard gradient descent step.

        Args:
            layer: A trainable layer with weights, biases, dweights, and dbiases attributes.
        """
        if self.momentum > 0:
            if not hasattr(layer, "weight_momentum"):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.biases)

            layer.weight_momentum = self.momentum * layer.weight_momentum - \
                self.current_learning_rate * layer.dweights
            layer.bias_momentum = self.momentum * layer.bias_momentum - \
                self.current_learning_rate * layer.dbiases

            layer.weights += layer.weight_momentum
            layer.biases += layer.bias_momentum

        else:
            layer.weights -= self.current_learning_rate * layer.dweights
            layer.biases -= self.current_learning_rate * layer.dbiases


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer. Adapts the learning rate per parameter by accumulating
    the sum of squared gradients. Parameters with large historical gradients
    receive smaller updates over time.

    Well suited for sparse data but may cause the learning rate to shrink too
    aggressively on long training runs.
    """

    def __init__(self, learning_rate=1e-2, decay=0., epsilon=1e-7):
        """
        Initializes AdaGrad optimizer hyperparameters.

        Args:
            learning_rate (float, optional): Initial learning rate. Default is 0.01.
            decay (float, optional): Learning rate decay applied each step. Default is 0.
            epsilon (float, optional): Small constant for numerical stability in
                                       the denominator. Default is 1e-7.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        """
        Applies the AdaGrad update rule to a layer's weights and biases.

        Initializes per-layer cache arrays on the first call. Accumulates
        squared gradients into the cache and scales the update accordingly.

        Args:
            layer: A trainable layer with weights, biases, dweights, and dbiases attributes.
        """
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights -= self.current_learning_rate / \
            (np.sqrt(layer.weight_cache) + self.epsilon) * layer.dweights
        layer.biases -= self.current_learning_rate / \
            (np.sqrt(layer.bias_cache) + self.epsilon) * layer.dbiases


class RMSprop(Optimizer):
    """
    RMSprop optimizer. Maintains a decaying average of squared gradients
    to normalize the update step, preventing the learning rate from
    shrinking indefinitely as AdaGrad does.
    """

    def __init__(self, learning_rate=1e-3, decay=0., beta=0.9, epsilon=1e-7):
        """
        Initializes RMSprop optimizer hyperparameters.

        Args:
            learning_rate (float, optional): Initial learning rate. Default is 0.001.
            decay (float, optional): Learning rate decay applied each step. Default is 0.
            beta (float, optional): Decay rate for the moving average of squared
                                    gradients. Default is 0.9.
            epsilon (float, optional): Small constant for numerical stability in
                                       the denominator. Default is 1e-7.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.beta = beta
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        """
        Applies the RMSprop update rule to a layer's weights and biases.

        Initializes per-layer cache arrays on the first call. Updates the
        exponentially decaying average of squared gradients and scales the
        parameter update by the root of the cache.

        Args:
            layer: A trainable layer with weights, biases, dweights, and dbiases attributes.
        """
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = layer.weight_cache * self.beta + \
            (1 - self.beta) * (layer.dweights ** 2)
        layer.bias_cache = layer.bias_cache * self.beta + \
            (1 - self.beta) * (layer.dbiases ** 2)

        layer.weights -= self.current_learning_rate / \
            (np.sqrt(layer.weight_cache) + self.epsilon) * layer.dweights
        layer.biases -= self.current_learning_rate / \
            (np.sqrt(layer.bias_cache) + self.epsilon) * layer.dbiases