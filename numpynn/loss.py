import numpy as np
from abc import abstractmethod


class Loss:
    """
    Abstract base class for all loss functions.
    """

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Computes per-sample losses given predictions and ground truth.

        Args:
            y_pred (np.ndarray): Predicted values from the model.
            y_true (np.ndarray): Ground truth labels or values.

        Returns:
            np.ndarray: Per-sample loss values.
        """
        pass

    @abstractmethod
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        """
        Computes the gradient of the loss with respect to its inputs.

        Args:
            dvalues (np.ndarray): Output of the forward pass (predictions).
            y_true (np.ndarray): Ground truth labels or values.
        """
        pass

    def regularization_loss(self):
        """
        Computes the total regularization loss across all trainable layers.

        Accumulates L1 and L2 penalties on weights and biases for any layer
        where the corresponding regularization strength is greater than zero.

        Returns:
            float: Total regularization loss summed over all trainable layers.
        """
        regularization_loss = 0

        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                    np.sum(layer.weights ** 2)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                    np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                    np.sum(layer.biases ** 2)

        return regularization_loss

    def calculate(self, output: np.ndarray, y: np.ndarray
                  , *, include_regularization=False):
        """
        Computes mean data loss for the current batch and updates accumulators.

        Args:
            output (np.ndarray): Model predictions.
            y (np.ndarray): Ground truth labels or values.
            include_regularization (bool, optional): If True, also returns
                                                     regularization loss. Default is False.

        Returns:
            float: Mean data loss for the batch.
            tuple(float, float): (data_loss, regularization_loss) if include_regularization is True.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        """
        Computes mean data loss across all batches seen since the last `new_pass` call.

        Args:
            include_regularization (bool, optional): If `True`, also returns
                                                     regularization loss. Default is `False`.

        Returns:
            float: Accumulated mean data loss.
            tuple(float, float): `(data_loss, regularization_loss)` if `include_regularization` is `True`.
        """
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def remember_trainable_layers(self, trainable_layers: list):
        """
        Stores a reference to the model's trainable layers for use in regularization.

        Args:
            `trainable_layers` (list): List of layers that have weights and biases.
        """
        self.trainable_layers = trainable_layers

    def new_pass(self):
        """
        Resets the accumulated loss sum and sample count. Should be called
        at the start of each epoch.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0


class CategoricalCrossEntropy(Loss):
    """
    Categorical cross-entropy loss for multi-class classification.

    Supports both sparse labels and one-hot encoded label arrays.
    """

    def forward(self, y_pred, y_true):
        """
        Computes per-sample categorical cross-entropy loss.

        Clips predictions to avoid log(0). Handles both sparse and
        one-hot encoded ground truth labels.

        Args:
            y_pred (np.ndarray): Predicted probabilities, shape (samples, classes).
            y_true (np.ndarray): Ground truth, either class indices (1D)
                                 or one-hot encoded vectors (2D).

        Returns:
            np.ndarray: Per-sample loss values.
        """
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 2:
            sample_losses = -np.sum(y_true * np.log(y_pred_clip), axis=1)

        elif len(y_true.shape) == 1:
            correct_confidences = y_pred_clip[range(len(y_pred)), y_true]
            sample_losses = -np.log(correct_confidences)

        return sample_losses

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        """
        Computes the gradient of the loss with respect to predictions.

        Converts sparse labels to one-hot encoding if needed, then
        computes and normalizes the gradient across samples.

        Args:
            dvalues (np.ndarray): Predicted probabilities from the forward pass.
            y_true (np.ndarray): Ground truth, either class indices (1D)
                                 or one-hot encoded vectors (2D).
        """
        if len(y_true.shape) == 1:
            labels = len(dvalues[0])
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs /= len(dvalues)


class SoftmaxCategoricalCrossEntropy:
    """
    Combined Softmax activation and Categorical Cross-Entropy loss.
    """

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        """
        Computes the combined gradient of Softmax and CCE in one step.

        Converts one-hot labels to sparse indices if needed, subtracts 1
        from the predicted probability at the true class index, then
        normalizes across samples.

        Args:
            dvalues (np.ndarray): Softmax output from the forward pass.
            y_true (np.ndarray): Ground truth, either class indices (1D)
                                 or one-hot encoded vectors (2D).
        """
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


class BinaryCrossEntropy(Loss):
    """
    Binary cross-entropy loss for binary classification.

    Expects Sigmoid output and labels of 0 or 1. Supports multi-output
    binary classification where each output is treated independently.
    """

    def forward(self, y_pred, y_true):
        """
        Computes per-sample binary cross-entropy loss.

        Clips predictions to avoid log(0), then averages the loss
        across outputs for each sample.

        Args:
            y_pred (np.ndarray): Predicted probabilities in range (0, 1).
            y_true (np.ndarray): Ground truth binary labels (0 or 1).

        Returns:
            np.ndarray: Per-sample mean loss values.
        """
        y_true = y_true.reshape(y_pred.shape)
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = y_true * np.log(y_pred_clip) + \
            (1 - y_true) * np.log(1 - y_pred_clip)
        sample_losses = -np.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        """
        Computes the gradient of the binary cross-entropy loss.

        Clips dvalues to avoid division by zero, then normalizes
        across both outputs and samples.

        Args:
            dvalues (np.ndarray): Predicted probabilities from the forward pass.
            y_true (np.ndarray): Ground truth binary labels (0 or 1).
        """
        outputs = len(dvalues[0])
        y_true = y_true.reshape(dvalues.shape)
        dvalue_clip = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = (dvalue_clip - y_true) / \
            (dvalue_clip * (1 - dvalue_clip)) / outputs
        self.dinputs /= len(dvalues)


class MeanSquareError(Loss):
    """
    Mean squared error loss for regression tasks.

    Measures the average squared difference between predictions and targets
    across all outputs.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Computes per-sample mean squared error.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): Ground truth values.

        Returns:
            np.ndarray: Per-sample MSE values.
        """
        sample_losses = np.mean((y_pred - y_true) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        """
        Computes the gradient of MSE with respect to predictions,
        normalized across outputs and samples.

        Args:
            dvalues (np.ndarray): Predicted values from the forward pass.
            y_true (np.ndarray): Ground truth values.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / (outputs * samples)
