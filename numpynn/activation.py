import numpy as np
from abc import abstractmethod

class Activation:
    """
    Abstract base class for all activation functions.
    """

    @abstractmethod
    def forward(self, inputs: np.ndarray, training: bool):
        """
        Computes the activation output given inputs.

        Args:
            inputs (np.ndarray): Input values from the previous layer.
            training (bool): Whether the model is in training mode.
        """
        pass

    @abstractmethod
    def backward(self, dvalues: np.ndarray):
        """
        Computes the gradient of the activation with respect to its inputs.

        Args:
            dvalues (np.ndarray): Gradient values from the next layer.
        """
        pass

    @abstractmethod
    def predictions(self, outputs: np.ndarray):
        """
        Converts raw activation outputs into predictions.

        Args:
            outputs (np.ndarray): Output of the forward pass.

        Returns:
            np.ndarray: Predicted values or class indices.
        """
        pass


class Linear(Activation):
    """
    Linear activation function. Output equals input, no transformation applied.
    """

    def forward(self, inputs: np.ndarray, training: bool):
        """
        Passes inputs through unchanged.

        Args:
            inputs (np.ndarray): Input values from the previous layer.
            training (bool): Whether the model is in training mode.
        """
        self.inputs, self.output = inputs, inputs

    def backward(self, dvalues: np.ndarray):
        """
        Passes gradients through unchanged, since the derivative of a linear
        function is 1.

        Args:
            dvalues (np.ndarray): Gradient values from the next layer.
        """
        self.dinputs = dvalues.copy()

    def predictions(self, outputs: np.ndarray):
        """
        Returns outputs directly, as no thresholding or argmax is needed.

        Args:
            outputs (np.ndarray): Output of the forward pass.

        Returns:
            np.ndarray: The raw outputs unchanged.
        """
        return outputs


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.
    Outputs the input directly if positive, otherwise outputs zero.
    """

    def forward(self, inputs, training):
        """
        Applies the ReLU function, clipping all negative values to zero.

        Args:
            inputs (np.ndarray): Input values from the previous layer.
            training (bool): Whether the model is in training mode.
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues: np.ndarray):
        """
        Computes gradients, zeroing out positions where the original input
        was zero or negative.

        Args:
            dvalues (np.ndarray): Gradient values from the next layer.
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs: np.ndarray):
        """
        Returns outputs directly, as ReLU is used in regression or hidden layers.

        Args:
            outputs (np.ndarray): Output of the forward pass.

        Returns:
            np.ndarray: The raw outputs unchanged.
        """
        return outputs


class Sigmoid(Activation):
    """
    Sigmoid activation function. Squashes inputs to a range of (0, 1),
    commonly used for binary classification output layers.
    """

    def forward(self, inputs: np.ndarray, training):
        """
        Applies the sigmoid function to each input value.

        Args:
            inputs (np.ndarray): Input values from the previous layer.
            training (bool): Whether the model is in training mode.
        """
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues: np.ndarray):
        """
        Computes gradients using the sigmoid derivative: sigmoid(x) * (1 - sigmoid(x)).

        Args:
            dvalues (np.ndarray): Gradient values from the next layer.
        """
        self.dinputs = dvalues * self.output * (1 - self.output)

    def predictions(self, outputs: np.ndarray):
        """
        Thresholds outputs at 0.5 to produce binary class predictions (0 or 1).

        Args:
            outputs (np.ndarray): Output of the forward pass.

        Returns:
            np.ndarray: Binary predictions as integers.
        """
        return (outputs > 0.5) * 1


class Softmax(Activation):
    """
    Softmax activation function. Converts raw scores into a probability
    distribution across classes. Typically used as the output activation
    for multi-class classification.
    """

    def forward(self, inputs: np.ndarray, training):
        """
        Applies the softmax function row-wise, with max subtraction for
        numerical stability.

        Args:
            inputs (np.ndarray): Input values from the previous layer.
            training (bool): Whether the model is in training mode.
        """
        self.inputs = inputs
        # Subtract row-wise max before exponentiating for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues: np.ndarray):
        """
        Computes gradients via the Jacobian matrix of the softmax function
        for each sample in the batch.

        Args:
            dvalues (np.ndarray): Gradient values from the next layer.
        """
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs: np.ndarray):
        """
        Returns the index of the highest-probability class for each sample.

        Args:
            outputs (np.ndarray): Output of the forward pass.

        Returns:
            np.ndarray: Predicted class indices.
        """
        return np.argmax(outputs, axis=1)