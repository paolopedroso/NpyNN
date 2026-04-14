# Loss

We talk about the `loss.py` here.

## Loss `CategoricalCrossEntropy`

$$
Loss = -\sum_{i=1} y_i \cdot \log{\hat{y}_i}
$$

Where $y_i$ are the truth values and $\hat{y}_i$ are the predicted values.

### Forward pass: `forward(self, y_pred, y_true)`
- Input: `y_pred`, `y_true`
- Output: `loss`
```python
def forward(self, y_pred, y_true):
    y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)

    if len(y_true.shape) == 2:
        sample_losses = -np.sum(y_true * np.log(y_pred), axis=1)
    elif len(y_true.shape) == 1:
        correct_confidences = y_pred_clip[range(len(y_pred)), y_true]
        sample_losses = -np.log(correct_confidences)
    
    return sample_losses
```
### Case `len(y_true.shape) == 2`:

If `y_true.shape` returns a shape of 2, then we know the argument is passed in as a one-hot vector. A one hot vector can be seen as:

$$
\text{One-hot Vector} = \begin{pmatrix} 
0 & 1 & \cdots & a_{1,j} \\ 
1 & 0 & \cdots & a_{2,j} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
a_{i,1} & a_{i,2} & \cdots & a_{i,j} 
\end{pmatrix}
$$

If `y_true` is given this shape, we can therefore multiply the one-hot vector with the `y_pred` values. 

$$
\begin{pmatrix} 
0 & 1 & \cdots & a_{1,j} \\ 
1 & 0 & \cdots & a_{2,j} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
a_{i,1} & a_{i,2} & \cdots & a_{i,j} 
\end{pmatrix} \cdot \begin{pmatrix} 
0.4 & 0.6 & \cdots & b_{1,j} \\ 
0.9 & 0.1 & \cdots & b_{2,j} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
b_{i,1} & b_{i,2} & \cdots & b_{i,j} 
\end{pmatrix}
$$

Or can be seen as:

$$
\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \end{pmatrix} \cdot \begin{pmatrix} 0.4 & 0.6 \\\\ 0.9 & 0.1 \end{pmatrix} = \begin{pmatrix} 0 & 0.6 \\\\ 0.9 & 0 \end{pmatrix}
$$

Then summing it through `axis=1` (summing through columns) gives us:

$$
\begin{pmatrix} 0.6 \\\\ 0.9 \end{pmatrix}
$$

### Case `len(y_true.shape) == 1`:

If `y_true.shape` returns a shape of 1, then we know that the argument is passed in as sparse labels, it can be seen as:

$$\text{Sparse Labels} = [0, 1, 2, 3, \cdots, n_i]$$

Where each value is the **index** of the correct class for that sample, rather than a full one-hot vector.

With this we can use numpy's fancy indexing to our advantage to directly select the correct class confidence from `y_pred`. For example given:

$$y_\text{true} = [0, 1, 1]$$

$$y_\text{pred} = \begin{pmatrix} 0.7 & 0.2 & 0.1 \\\\ 0.1 & 0.8 & 0.1 \\\\ 0.3 & 0.6 & 0.1 \end{pmatrix}$$

Numpy fancy indexing `y_pred[range(len(y_pred)), y_true]` selects:

$$\begin{pmatrix} y_\text{pred}[0, 0] \\\\ y_\text{pred}[1, 1] \\\\ y_\text{pred}[2, 1] \end{pmatrix} = \begin{pmatrix} 0.7 \\\\ 0.8 \\\\ 0.6 \end{pmatrix}$$

In more pythonic terms fancy indexing is like:

$$\begin{pmatrix} y_\text{pred}[0][0] \\\\ y_\text{pred}[1][1] \\\\ y_\text{pred}[2][1] \end{pmatrix} = \begin{pmatrix} 0.7 \\\\ 0.8 \\\\ 0.6 \end{pmatrix}$$

This is equivalent to the one-hot case, we get the same vector of correct class confidences, just without needing to construct the full one-hot matrix. We then apply $-\log$ to each:

$$\text{sample\\_losses} = -\log\begin{pmatrix} 0.7 \\\\ 0.8 \\\\ 0.6 \end{pmatrix} = \begin{pmatrix} 0.357 \\\\ 0.223 \\\\ 0.511 \end{pmatrix}$$

### Backward Pass `backward(self, dvalues, y_true)`
- Input: `dvalues`, `y_true`
- Output: `loss`
```python
def backward(self, dvalues, y_true):
    if len(y_true.shape) == 1:
        labels = len(dvalues[0])
        y_true = np.eye(labels)[y_true]

    self.dinputs = -y_true / dvalues
    self.dinputs /= len(dvalues)
```

### Deriving $L$:

Starting from the categorical cross entropy loss:

$$L = -\sum_{i} y_i \cdot \log(\hat{y}_i)$$

We want to find how the loss changes with respect to each predicted value $\hat{y}_i$:

$$\frac{\partial L}{\partial \hat{y}_i} = \frac{\partial}{\partial \hat{y}_i} \left( -\sum_{i} y_i \cdot \log(\hat{y}_i) \right)$$

Since we are differentiating w.r.t. a specific $\hat{y}_i$, all other terms in the sum vanish:

$$\frac{\partial L}{\partial \hat{y}_i} = -y_i \cdot \frac{\partial}{\partial \hat{y}_i} \log(\hat{y}_i)$$

Applying the derivative of $\log$:

$$\frac{\partial}{\partial \hat{y}_i} \log(\hat{y}_i) = \frac{1}{\hat{y}_i}$$

Therefore:

$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}$$

Which is exactly `self.dinputs = -y_true / dvalues` in code, element-wise division of the one-hot truth values by the predicted values. The `self.dinputs /= len(dvalues)` normalises by the number of samples so the gradient magnitude doesn't scale with batch size.

### Case `len(y_true.shape) == 1`:

If the length of `y_true` values are passed in as sparse values, we transform it to a shape matching `dvalues` (or derivative of the output from the next layer). For example, if `len(dvalues) = 3` using `np.eye(3)`:

$$\text{np.eye}(3) = \begin{pmatrix} 1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \end{pmatrix}$$

It returns a diagonal $n \times n$ of 1's, shape is determined by $n$. We can control the diagonal by passing in $i$ for example here: `np.eye(3)[i]`. If we pass in $i$ as 2, which would return:

$$\text{np.eye}(3)[2] = \begin{pmatrix} 0 & 0 & 1 \end{pmatrix}$$

The one-hot encoded values are determined by $i$. So if we pass in a list to $i$ to `np.eye(3)[i]`. For example if $i = [2, 1, 1]$. It would return:

$$\text{np.eye}(3)[[2, 1, 1]] = \begin{pmatrix} 0 & 0 & 1 \\\\ 0 & 1 & 0 \\\\ 0 & 1 & 0 \end{pmatrix}$$

Then we represent $i$ the `y_true` values to calculate backpropagation in the `CategoricalCrossEntropy` class.

At the end we normalize it by `dinputs = dinputs / len(dvalues)`.

## Loss `BinaryCrossEntropy`:

$$
\text{Binary Cross Entropy} = -\frac{1}{n}\sum_{i=1}^{n}\left[ y \cdot \log{\hat{y}} + (1 - y) \cdot \log{(1 - \hat{y})} \right]
$$

Where $\hat{y}$ are the predicted values and $y$ are the ground truth values.

### Forward pass `forward(self, y_pred, y_true)`
- Input: `y_pred`, `y_true`
- Output: `loss`
```python
def forward(self, y_pred, y_true):
    y_true = y_true.reshape(y_pred.shape)
    y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
    sample_losses = y_true * np.log(y_pred_clip) + \
        (1 - y_true) * np.log(1 - y_pred_clip)
    sample_losses = -np.mean(sample_losses, axis=-1)
    return sample_losses
```

### Backward Pass `backward(self, dvalues, y_true)`:

```python
def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
    outputs = len(dvalues[0])
    y_true = y_true.reshape(dvalues.shape)
    dvalue_clip = np.clip(dvalues, 1e-7, 1 - 1e-7)
    self.dinputs = (dvalue_clip - y_true) / \
        (dvalue_clip * (1 - dvalue_clip)) / outputs
    self.dinputs /= len(dvalues)
```

### Deriving $L$ for Binary Cross Entropy:

Starting from the binary cross entropy loss, unlike CCE which sums over $n$ classes, BCE only has two outcomes, 1 or 0:

$$L = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right]$$

We want to find how the loss changes with respect to each predicted value $\hat{y}_i$:

$$\frac{\partial L}{\partial \hat{y}_i} = \frac{\partial}{\partial \hat{y}_i} \left( -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right] \right)$$

Pulling the constant $-\frac{1}{n}$ out and dropping the sum since we differentiate w.r.t. a specific $\hat{y}_i$:

$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{1}{n} \left[ \frac{\partial}{\partial \hat{y}_i} y_i \cdot \log(\hat{y}_i) + \frac{\partial}{\partial \hat{y}_i} (1 - y_i) \cdot \log(1 - \hat{y}_i) \right]$$

Applying the derivative of $\log$ to each term separately. For the first term:

$$\frac{\partial}{\partial \hat{y}_i} y_i \cdot \log(\hat{y}_i) = \frac{y_i}{\hat{y}_i}$$

For the second term, applying chain rule since we have $\log(1 - \hat{y}_i)$:

$$\frac{\partial}{\partial \hat{y}_i} (1 - y_i) \cdot \log(1 - \hat{y}_i) = -\frac{1 - y_i}{1 - \hat{y}_i}$$

Combining both terms:

$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{1}{n} \left[ \frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i} \right]$$

The given $\frac{\partial L}{\partial \hat{y}_i}$ is our `self.dinputs` here.

```python
outputs = len(dvalues[0])
y_true = y_true.reshape(dvalues.shape)
dvalue_clip = np.clip(dvalues, 1e-7, 1 - 1e-7)
self.dinputs = (dvalue_clip - y_true) / (dvalue_clip * (1 - dvalue_clip)) / outputs
```

Then we must normalize the values by dividing $\frac{\partial L}{\partial \hat{y}_i}$ or `dinputs` by the length of dvalues here:
```python
self.dinputs = self.dinputs / len(dvalues)
```

## Loss `MeanSquaredError`:

$$
\text{Mean Squared Error} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

Where $y_i$ are the truth values and $\hat{y}_i$ are the predicted values.

### Forward Pass `forward(self, y_pred, y_true)`:
```python
def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
    sample_losses = np.mean((y_pred - y_true) ** 2, axis=-1)
    return sample_losses
```

Directly translates to the python code above.

### Backward Pass `backward(self, dvalues, y_true)`:
```python
def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
    samples = len(dvalues)
    outputs = len(dvalues[0])
    self.dinputs = -2 * (y_true - dvalues) / (outputs * samples)
```

### Deriving $L$:

We let $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ where we differentiate $L$ w.r.t $\hat{y}$.

$$\frac{\partial L}{\partial \hat{y}_i} = \frac{\partial}{\partial \hat{y}_i}\left(\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2\right)$$

Deriving it is simply:

$$\frac{\partial L}{\partial \hat{y}_i} = -2 \cdot \frac{1}{n}(y_i - \hat{y}_i)$$

The $\sum$ collapses because when differentiating w.r.t. a specific $\hat{y}_i$, all other terms in the sum vanish since they do not depend on $\hat{y}_i$.
