# Activations

We talk about the `activation.py` here.

## ReLU Activation

$$
\text{ReLU} = max(0, x)
$$

### Forward Pass `forward(self, inputs, training)`:
- Input: `inputs`, `training`
- Output: `None`
```python
def forward(self, inputs, training):
    self.inputs = inputs
    self.output = np.maximum(0, inputs)
```

ReLU outputs only positive values of the given inputs. We use numpy's `np.maximum`. The difference with `np.max` is that `np.maximum` returns an element wise maximum of the array. `np.max` returns a single maximum across all arrays.

### Backward Pass `backward(self, dvalues)`:
- Input: `dvalues`
- Output: `None`
```python
def backward(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0
```

### Deriving ReLU:

Given:

$$
\text{ReLU} = max(0, x)
$$

The full chain rule goes:
$$
\underbrace{\frac{\partial L}{\partial x}}_{\text L w.r.t. x} = \underbrace{\frac{\partial L}{\partial r}}_{\text{L w.r.t. relu}} \cdot \underbrace{\frac{\partial r}{\partial x}}_{\text{relu w.r.t. z}}
$$

We need to know how much of $x$ input affects $L$, thus we need to solve for: $\frac{\partial L}{\partial x}$. Since $\frac{\partial L}{\partial r}$ is given from upstream from next layer, we multply $\frac{\partial r}{\partial x}$, or $r$ relu w.r.t. $x$. Since postive values at $0$ returns a $x$ in the `forward` method. The derivative at positive values, is just 1. Given here:

$$
\frac{\partial r}{\partial x} = \frac{\partial}{\partial x}(\max(0,x)) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

Finalizing our $\frac{\partial L}{\partial x}$ gives us:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial r} \cdot 1[x > 0]
$$

To translate in our code we can just represent it as:

```python
dinputs = dvalues.copy()
dinputs[inputs <= 0] = 0
```

## Sigmoid Activation

$$
\text{Sigmoid} = \sigma(z) = \frac{1}{1+e^{-z}}
$$
- Where $z$ is the input value.

### Forward Pass `forward(self, inputs)`:
- Input: `inputs`
- Output: `None`
```python
def forward(self, inputs):
    self.output = 1 / (1 + -np.exp(inputs))
```

This code snippet directly translate to the sigmoid function.

### Backward Pass `backward(self, dvalues)`:
- Input: `dvalues`
- Output: `None`
```python
def backward(self, dvalues):
    self.dinputs = dvalues * self.output * (1 - self.output)
```

### Deriving Sigmoid:

Given: 
$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

The full chain of the $L$ w.r.t. $z$ inputs is represented as:

$$
\underbrace{\frac{\partial L}{\partial z}}_{\text L w.r.t. x} = \underbrace{\frac{\partial L}{\partial \sigma}}_{\text{L w.r.t. sigmoid}} \cdot \underbrace{\frac{\partial \sigma}{\partial z}}_{\text{relu w.r.t. z}}
$$

It's given this way because we want to know how much $z$ (inputs) affects $L$ (loss), thus giving us the partial derivative, $\frac{\partial L}{\partial z}$, The full chain is $\frac{\partial L}{\partial \sigma}$ multiplied by $\frac{\partial \sigma}{\partial z}$ because this will tell us how much $z$ affects $L$. $\frac{\partial L}{\partial \sigma}$ is added here because this is the gradient from upstream. Meaning that during backpropagation, the layer next to `Sigmoid` at upstream, the output of `Sigmoid` already calcualtes the derivative value, $\frac{\partial L}{\partial \sigma}$, thus this is `dvalues` in this sense at backpropagation.

Given: 
$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

We can rewrite this as:

$$
\sigma(z) = (1+e^{-z})^{-1}
$$

Applying the chain rule:

$$
\frac{\partial \sigma}{\partial z} = -(1+e^{-z})^{-2} \cdot \frac{\partial}{\partial z}(1+e^{-z})
$$

The derivative of $(1+e^{-z})$ w.r.t. $z$ is $-e^{-z}$, so:

$$
\frac{\partial \sigma}{\partial z} = -(1+e^{-z})^{-2} \cdot (-e^{-z})
$$

$$
\frac{\partial \sigma}{\partial z} = \frac{e^{-z}}{(1+e^{-z})^{2}}
$$

We can split the fraction to get it in terms of $\sigma(z)$:

$$
\frac{\partial \sigma}{\partial z} = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}}
$$

The left term is simply $\sigma(z)$, and the right term can be rewritten as $\frac{(1 + e^{-z}) - 1}{1+e^{-z}} = 1 - \sigma(z)$, therefore:

$$
\frac{\partial \sigma}{\partial z} = \sigma(z) \cdot (1 - \sigma(z))
$$

Now plugging $\frac{\partial \sigma}{\partial z}$ back to here: 

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial z}
$$

Gives us:

$$
\underbrace{\frac{\partial L}{\partial z}}_{\text{dinputs}} = \underbrace{\frac{\partial L}{\partial \sigma}}_{\text{dvalues}} \cdot \sigma(z) \cdot (1 - \sigma(z))
$$

Representing this as code:
```python
# forward pass ...
self.ouput = 1 / (1 + -np.exp(inputs))

# backward pass ...
dinputs = dvalues * self.output (1 - self.output)
```

## Softmax Activation

The softmax formula gives us:

$$
\text{Softmax(z)}_i = \sigma(z)_i = \frac{e^i}{\sum_{j=1}^{K}e^j}
$$

### Foward Pass `forward(self, inputs)`:
- Input: `inputs`
- Output: `None`
```python
def forward(self, inputs):
    self.inputs = inputs
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)    
```

Softmax is defined as:

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

The problem is that $e^{z_i}$ grows extremely fast, for example $e^{1000}$ overflows to `inf` in numpy, making the entire output `nan`. To fix this we use the **numerical stability trick**:

$$
\sigma(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{n} e^{z_j - \max(z)}}
$$

This is mathematically identical to the original since we can factor out $e^{-\max(z)}$ from both numerator and denominator and it cancels:

$$
\frac{e^{-\max(z)} \cdot e^{z_i}}{e^{-\max(z)} \cdot \sum_{j=1}^{n} e^{z_j}} = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

By subtracting $\max(z)$ first, the largest value in the exponent becomes $e^0 = 1$ and all others are smaller, keeping values in a safe range. The `axis=1, keepdims=True` ensures the max is computed **per sample** across all classes, and the shape is preserved for broadcasting against the full `inputs` matrix.

In the denominator we do `np.sum(..., axis=1)` because `inputs` has shape of `(batches, features)`. Since `axis=1` is the **features** dimension, summing along it collapses all class scores into one value per sample, this becomes the denominator that each sample's `exp_values` are divided by, normalizing each sample's class scores into a probability distribution that sums to 1.

### Backward Pass `backward(self, dvalues)`:
- Input: `dvalues`
- Output: `None`
```python
def backward(self, dvalues: np.ndarray):
    self.dinputs = np.empty_like(dvalues)

    for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
        single_output = single_output.reshape(-1, 1)
        jacobian_matrix = np.diagflat(single_output) - \
                            np.dot(single_output, single_output.T)
        self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
```

Unlike ReLU or sigmoid which apply element-wise, softmax's backward pass is more complex because each output $\sigma(z_i)$ depends on **all** inputs $z_j$ through the denominator $\sum_{j} e^{z_j}$. This means we need a **Jacobian matrix**, a matrix of all partial derivatives of each output w.r.t. each input:

$$J = \begin{pmatrix} \frac{\partial \sigma_1}{\partial z_1} & \frac{\partial \sigma_1}{\partial z_2} & \cdots & \frac{\partial \sigma_1}{\partial z_n} \\ \frac{\partial \sigma_2}{\partial z_1} & \frac{\partial \sigma_2}{\partial z_2} & \cdots & \frac{\partial \sigma_2}{\partial z_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial \sigma_n}{\partial z_1} & \frac{\partial \sigma_n}{\partial z_2} & \cdots & \frac{\partial \sigma_n}{\partial z_n} \end{pmatrix}$$

The derivative has two cases depending on whether $i = j$ or $i \neq j$:

$$\frac{\partial \sigma_i}{\partial z_j} = \begin{cases} \sigma_i(1 - \sigma_i) & \text{if } i = j \\ -\sigma_i \sigma_j & \text{if } i \neq j \end{cases}$$

Which can be written compactly as:

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)$$

Where $\delta_{ij}$ is 1 when $i = j$ and 0 otherwise. In code this becomes:

- `np.diagflat(single_output)`, places $\sigma_i$ on the diagonal, representing the $i = j$ case
- `np.dot(single_output, single_output.T)`, produces the outer product $\sigma_i \sigma_j$ for all pairs, representing the $i \neq j$ case
- Subtracting the two gives the full Jacobian

The full chain rule for the backward pass is:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial z}$$

Where $\frac{\partial \sigma}{\partial z}$ is the Jacobian $J$, so:

$$\frac{\partial L}{\partial z_i} = \sum_{j} \frac{\partial L}{\partial \sigma_j} \cdot \frac{\partial \sigma_j}{\partial z_i} = \sum_{j} \frac{\partial L}{\partial \sigma_j} \cdot \sigma_j(\delta_{ij} - \sigma_i)$$

The $\sum_j$ appears because changing $z_i$ affects **every** output $\sigma_j$, so all those paths must be summed, unlike sigmoid or ReLU where $z_i$ only affects $\sigma_i$. Then `np.dot(jacobian_matrix, single_dvalues)` computes exactly this sum per sample, multiplying the Jacobian by the upstream gradient `dvalues` to get $\frac{\partial L}{\partial z}$.