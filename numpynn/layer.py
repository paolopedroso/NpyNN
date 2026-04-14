import numpy as np
from abc import abstractmethod


class Layer:
    """
    Abstract base class for all layers.

    All layers must implement forward and backward passes.
    """

    @abstractmethod
    def forward(self, inputs, training):
        """
        Computes the layer output given inputs.

        Args:
            inputs (np.ndarray): Input values from the previous layer.
            training (bool): Whether the model is in training mode.
        """
        pass

    @abstractmethod
    def backward(self, dvalues):
        """
        Computes gradients with respect to the layer's inputs.

        Args:
            dvalues (np.ndarray): Gradient values from the next layer.
        """
        pass


class Dense(Layer):
    """
    Fully connected (dense) layer. Every input is connected to every neuron.

    Supports optional L1 and L2 regularization on both weights and biases.
    """

    def __init__(self, n_inputs: int, n_neurons: int,
                 weight_regularizer_l1=0, bias_regularizer_l1=0,
                 weight_regularizer_l2=0, bias_regularizer_l2=0):
        """
        Initializes weights and biases, and stores regularization strengths.

        Weights are randomly initialized scaled by 0.01; biases are initialized to zero.

        Args:
            n_inputs (int): Number of inputs coming into this layer.
            n_neurons (int): Number of neurons in this layer.
            weight_regularizer_l1 (float, optional): L1 regularization strength for weights. Default is 0.
            bias_regularizer_l1 (float, optional): L1 regularization strength for biases. Default is 0.
            weight_regularizer_l2 (float, optional): L2 regularization strength for weights. Default is 0.
            bias_regularizer_l2 (float, optional): L2 regularization strength for biases. Default is 0.
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs: np.ndarray, training) -> None:
        """
        Computes the layer output as a dot product of inputs and weights, plus biases.

        Args:
            inputs (np.ndarray): Input data from the previous layer.
            training (bool): Whether the model is in training mode.
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Computes gradients for weights, biases, and inputs. Applies regularization
        gradient contributions to dweights and dbiases if regularization strengths are set.

        Args:
            dvalues (np.ndarray): Gradient values from the next layer.
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self) -> tuple:
        """
        Returns the layer's current weights and biases.

        Returns:
            tuple: (weights, biases) as numpy arrays.
        """
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        """
        Overwrites the layer's weights and biases with the provided values.

        Args:
            weights (np.ndarray): New weights to assign.
            biases (np.ndarray): New biases to assign.
        """
        self.weights = weights
        self.biases = biases


class Dropout(Layer):
    """
    Dropout layer for regularization. Randomly zeros out a fraction of neurons
    during training to reduce overfitting. Outputs are scaled so the expected
    value remains consistent between training and inference.
    """

    def __init__(self, rate):
        """
        Stores the keep rate (inverse of the dropout rate).

        Args:
            rate (float): Fraction of neurons to drop during training (e.g. 0.2 drops 20%).
        """
        self.rate = 1 - rate

    def forward(self, inputs, training):
        """
        Applies the dropout mask during training. During inference, passes
        inputs through unchanged.

        Args:
            inputs (np.ndarray): Input values from the previous layer.
            training (bool): Whether the model is in training mode.
                             If False, no dropout is applied.
        """
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        self.output = self.inputs * self.binary_mask

    def backward(self, dvalues):
        """
        Passes gradients through only the neurons that were kept during the forward pass.

        Args:
            dvalues (np.ndarray): Gradient values from the next layer.
        """
        self.dinputs = dvalues * self.binary_mask


class Input_Layer:
    """
    Passthrough layer used as the entry point of the model. Holds the raw
    input data so that the first real layer can reference it via the standard
    prev.output interface.
    """

    def forward(self, inputs, training):
        """
        Stores inputs as output so the next layer can access them.

        Args:
            inputs (np.ndarray): Raw input data passed into the model.
            training (bool): Whether the model is in training mode.
        """
        self.output = inputs

    def backward(self, dvalues):
        """
        Not implemented, gradients are never propagated past the input layer.

        Raises:
            NotImplementedError: 
        """
        raise NotImplementedError("")
    