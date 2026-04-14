import os
import copy
import pickle
import numpy as np
from numpynn.layer import Input_Layer

class Model:
    """
    Neural network model class that manages layers, training, evaluation, and inference.

    Usage:
        model = Model()
        model.add(Dense(2, 64))
        model.add(ReLU())
        model.set(loss=..., optimizer=..., accuracy=...)
        model.finalize()
        model.train(X, y, epochs=100)
    """

    def __init__(self):
        """
        Initializes an empty model with no layers.
        """
        self.layers = []
        self.softmax_classifier_output = None  # combined softmax+CCE
    
    def add(self, layer):
        """
        Adds a layer to the model.

        Args:
            layer: Any layer, activation, or dropout object to append to the network.
        """
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        """
        Sets the loss function, optimizer, and accuracy object for the model.

        Args:
            loss (Loss): Loss function to use during training.
            optimizer (Optimizer): Optimizer to use for parameter updates.
            accuracy (Accuracy): Accuracy object to track training performance.
        """
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy
        
    def finalize(self):
        """
        Finalizes the model by creating a doubly linked list of layers for
        forward and backward passes. Must be called before training.

        Also detects if the last layer is Softmax paired with CategoricalCrossEntropy
        and enables the faster combined backward pass automatically.
        """
        layer_count = len(self.layers)
        self.trainable_layers = []
        self.input_layer = Input_Layer()

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            elif i == layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
        
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and loss is CCE,
        # use combined activation/loss for faster backward step
        from numpynn.activation import Softmax
        from numpynn.loss import CategoricalCrossEntropy, SoftmaxCategoricalCrossEntropy
        if isinstance(self.layers[-1], Softmax) and \
           isinstance(self.loss, CategoricalCrossEntropy):
            self.softmax_classifier_output = SoftmaxCategoricalCrossEntropy()

    def forward(self, X: np.ndarray, training: bool):
        """
        Performs a forward pass through all layers in the model.

        Args:
            X (np.ndarray): Input data.
            training (bool): Whether the model is in training mode.
                             Affects layers like Dropout.

        Returns:
            np.ndarray: Output of the last layer.
        """
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        return layer.output
    
    def backward(self, output, y):
        """
        Performs a backward pass through all layers in reverse order.

        If Softmax + CategoricalCrossEntropy are used, takes the faster
        combined backward path. Otherwise falls back to standard backprop.

        Args:
            output (np.ndarray): Output of the forward pass.
            y (np.ndarray): Ground truth labels.
        """
        # If softmax classifier, use combined backward for speed
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(self, X: np.ndarray, y: np.ndarray, *, epochs=1,
              batch_size=None, print_every=1, validation_data=None):
        """
        Trains the model on the provided data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            epochs (int, optional): Number of training epochs. Default is 1.
            batch_size (int, optional): Number of samples per batch.
                                        If None, uses the full dataset. Default is None.
            print_every (int, optional): Print summary every N epochs. Default is 1.
            validation_data (tuple, optional): Tuple of (X_val, y_val) for
                                               validation after each epoch. Default is None.
        """
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }
        self.accuracy
        self.accuracy.init(y)
        train_steps = 1

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

        for epoch in range(1, epochs+1):

            print(f'epoch: {epoch}')

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions.flatten(), batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')
                    
            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(
                    include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            self.history['train_loss'].append(float(epoch_loss))
            self.history['train_acc'].append(float(epoch_accuracy))
            self.history['lr'].append(float(self.optimizer.current_learning_rate))

            if validation_data is not None:
                self.evaluate(*validation_data,
                              batch_size=batch_size)
                self.history['val_loss'].append(float(self.validation_loss))
                self.history['val_acc'].append(float(self.validation_accuracy))

    def evaluate(self, X_val, y_val, *, batch_size=None):
        """
        Evaluates the model on validation data without updating parameters.

        Args:
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            batch_size (int, optional): Number of samples per batch.
                                        If None, uses the full dataset. Default is None.
        """
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions.flatten(), batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        self.validation_loss = validation_loss
        self.validation_accuracy = validation_accuracy

        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    def predict(self, X, *, batch_size=None):
        """
        Runs inference on input data and returns raw model outputs.

        Args:
            X (np.ndarray): Input data to predict on.
            batch_size (int, optional): Number of samples per batch.
                                        If None, uses the full dataset. Default is None.

        Returns:
            np.ndarray: Stacked predictions across all batches.
        """
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []

        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            output.append(self.forward(batch_X, training=False))

        return np.vstack(output)

    def get_parameters(self):
        """
        Retrieves weights and biases from all trainable layers.

        Returns:
            list: List of (weights, biases) tuples for each trainable layer.
        """
        return [layer.get_parameters() for layer in self.trainable_layers]

    def set_parameters(self, parameters):
        """
        Updates all trainable layers with provided weights and biases.

        Args:
            parameters (list): List of (weights, biases) tuples,
                               one per trainable layer.
        """
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        """
        Saves only the model's weights and biases to a file.

        Args:
            path (str): File path to save parameters to.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        """
        Loads weights and biases from a file and applies them to the model.

        Args:
            path (str): File path to load parameters from.
        """
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        """
        Saves the entire model object including architecture and parameters to a file.

        Args:
            path (str): File path to save the model to.
        """
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        for layer in model.layers:
            for prop in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(prop, None)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        """
        Loads and returns a saved model from a file.

        Args:
            path (str): File path to load the model from.

        Returns:
            Model: The loaded model instance.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)