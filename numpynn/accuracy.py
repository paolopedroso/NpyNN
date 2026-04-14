import numpy as np


class Accuracy:
    """
    Abstract base class for all accuracy metrics.

    Handles per-batch and accumulated accuracy calculation. All accuracy
    classes must implement init and compare.
    """

    def calculate(self, predictions, y):
        """
        Computes mean accuracy for the current batch and updates accumulators.

        Args:
            predictions (np.ndarray): Predicted values or class indices.
            y (np.ndarray): Ground truth labels or values.

        Returns:
            float: Mean accuracy for the current batch.
        """
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):
        """
        Computes mean accuracy across all batches seen since the last new_pass call.

        Returns:
            float: Accumulated mean accuracy.
        """
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def new_pass(self):
        """
        Resets the accumulated sum and sample count. Should be called
        at the start of each epoch.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Accuracy_Categorical(Accuracy):
    """
    Accuracy metric for categorical classification tasks.

    Supports both standard multi-class predictions (argmax outputs) and
    binary predictions from a Sigmoid output layer.
    """

    def __init__(self, *, binary=False):
        """
        Args:
            binary (bool, optional): If True, treats the task as binary classification
                                     and skips argmax conversion of labels. Default is False.
        """
        self.binary = binary

    def init(self, y):
        """
        No initialization required for categorical accuracy.

        Args:
            y (np.ndarray): Ground truth labels. Unused, included for interface consistency.
        """
        pass

    def compare(self, predictions, y):
        """
        Compares predicted class indices against ground truth labels.

        Converts one-hot encoded labels to sparse indices for multi-class tasks
        unless binary mode is enabled.

        Args:
            predictions (np.ndarray): Predicted class indices or binary outputs.
            y (np.ndarray): Ground truth labels, either sparse (1D) or one-hot (2D).

        Returns:
            np.ndarray: Boolean array of correct predictions.
        """
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


class Accuracy_Regression(Accuracy):
    """
    Accuracy metric for regression tasks.

    A prediction is considered correct if it falls within a precision threshold
    derived from the standard deviation of the training targets.
    """

    def __init__(self):
        """
        Initializes the precision threshold to None until init is called.
        """
        self.precision = None

    def init(self, y, reinit=False):
        """
        Sets the precision threshold based on the standard deviation of y.

        The threshold is set to std(y) / 250. Skips reinitialization unless
        reinit is explicitly set to True.

        Args:
            y (np.ndarray): Ground truth values used to compute the threshold.
            reinit (bool, optional): If True, forces the threshold to be
                                     recomputed even if already set. Default is False.
        """
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        """
        Checks whether each prediction falls within the precision threshold of its target.

        Args:
            predictions (np.ndarray): Predicted values.
            y (np.ndarray): Ground truth values.

        Returns:
            np.ndarray: Boolean array where True means the prediction was close enough.
        """
        return np.absolute(predictions - y) < self.precision