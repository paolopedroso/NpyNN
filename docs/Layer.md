# Layers

We talk about the `layer.py` here.

## Layer `Dense`

### Initialization

```python
def __init__(self, n_inputs, n_neurons):
    self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
```

### Forward Pass: `forward(self, inputs)`
- Input: `inputs`
- Ouput: `None`
```python
def forward(self, inputs):
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases
```

### Backwards Pass: `backward(self, dvalues)`
- Input: `dvalues`
- Output: `None`
```python
def backward(self, dvalues):
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    self.dinputs = np.dot(dvalues, self.weights.T)
```

>Note: `dvalues` are the values from the next layer during backpropagation.

Dependence chain: $X \leftarrow Z \leftarrow L$

### Deriving $W$

Code snippet
```python
self.dweights = np.dot(self.inputs.T, dvalues)
```

$$
\underbrace{\frac{\partial L}{\partial W}}_{\text{dweights}} = \underbrace{\frac{\partial L}{\partial Z}}_{\text{dvalues}} \cdot \underbrace{\frac{\partial Z}{\partial W}}_{\text{inputs}}
$$

$$
\frac{\partial Z}{\partial W}(\underbrace{X \cdot W + b}_{\text{Z}}) = \underbrace{X}_{\text{inputs}}
$$

<!-- add note for * -->

`self.dweights` must match the shape of `self.weights`, which is `(n_inputs, n_neurons)`. We have `self.inputs` of shape `(batches, n_inputs)` and `dvalues` of shape `(batch, n_neurons)`*. To get the target shape `(n_inputs, n_neurons)` from a dot product, we transpose `self.inputs` from `(batches, n_inputs)` to `(n_inputs, batches)`, so that `np.dot(self.inputs.T, dvalues)` gives us `(n_inputs, batches) · (batches, n_neurons)` -> `(n_inputs, n_neurons)`.

\* `dvalues` come at a shape of `(batch, n_neurons)` because of $Z = W \cdot X + b$.

### Deriving $b$ or biases:

Code snippet:
```
self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
```

$$
\underbrace{\frac{\partial L}{\partial b}}_{\text{dbiases}} = \underbrace{\frac{\partial L}{\partial Z}}_{\text{dvalues}} \cdot \underbrace{\frac{\partial Z}{\partial b}}_{\text{biases}}
$$

$$
\frac{\partial Z}{\partial b}(\underbrace{X \cdot W + b}_{\text{Z}}) = 1
$$

Since $\frac{\partial Z}{\partial b} = 1$, `dbiases` is just `dvalues` itself. However, biases are shared across all samples in the batch, meaning each sample contributes its own gradient to the same bias. We therefore sum over `axis=0` to accumulate those gradients into a single update of shape `(1, n_neurons)`, matching the shape of $b$.

### Deriving $X$ or Inputs

Code snippet:
```python
self.dinputs = np.dot(dvalues, self.weights.T)
```

$$
\underbrace{\frac{\partial L}{\partial X}}_{\text{dinputs}} = \underbrace{\frac{\partial L}{\partial Z}}_{\text{dvalues}} \cdot \underbrace{\frac{\partial Z}{\partial X}}_{\text{weights}}
$$

$$
\frac{\partial Z}{\partial X}(\underbrace{X \cdot W + b}_{\text{Z}}) = \underbrace{W}_{\text{weights}}
$$

`dinputs` must match the shape of `self.inputs`, which is `(batches, n_inputs)`. We have `dvalues` of shape `(batches, n_neurons)` and `weights` of shape `(n_inputs, n_neurons)`. Transposing `weights` to `(n_neurons, n_inputs)` lets us compute `np.dot(dvalues, self.weights.T)` as `(batches, n_neurons) · (n_neurons, n_inputs)` -> `(batches, n_inputs)`, which is exactly the shape we need.

## Layer `Dropout`

### Initialization

```python
def __init__(self, rate):
    self.rate = 1 - rate
```

### Forward Pass `forward(self, inputs)`

```python
def forward(self, inputs):
    self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
    self.output = self.inputs * self.binary_mask
```

We use `np.random.binomial(1, self.rate, size=inputs.shape)` to generate a binary mask of the same shape as `inputs`, filled with `0`s and `1`s, where each value has a `self.rate` probability of being `1` (survive) and `1 - self.rate` of being `0` (drop).

However, randomly zeroing out neurons reduces the overall signal, for example, with a 50% rate, roughly half the neurons are dropped, so the sum of activations is halved during training. At inference, when dropout is disabled, all neurons are active and the sum is back to its original magnitude, causing a mismatch.

To fix this, we divide the mask by `self.rate`, which scales up the surviving neurons to compensate for the ones that were dropped. This way the expected sum stays equal to the original input sum during training, so no correction is needed at inference time.

### Backward Pass `backward(self, inputs)`

```python
def backward(self, dvalues):
    self.dinputs = dvalues * self.binary_mask
```

Same reasoning for the forward pass, we use the already calculated `self.binary_mask` multiplied by `dvalues`.