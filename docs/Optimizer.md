# Optimizer

We talk about `optim.py` here.

- [Adam Optimizer](#adam-optimizer)
- [AdamW Optimizer](#adamw-optimizer)
- [SGD Optimizer](#sgd-optimizer)
- [AdaGrad Optimizer](#adagrad-optimizer)
- [RMSprop Optimizer](#rmsprop-optimizer)

## Base Class Methods
- `update_params`
- `pre_update_params`
- `post_update_params`

### `update_params(self)`:
Updates the weights and biases of a single trainable layer.

| Args | Description |
| --- | --- |
| `layer` | A trainable layer with weights, biases, dweights, and dbiases attributes. |

| Return | Description |
| --- | --- |
| None | None |


```python 
@abstractmethod
def update_params(self, layer):
    pass
```

### `pre_update_params(self)`:

Applies inverse `lr_decay` to the learning rate before each update step. Only runs if a non-zero `lr_decay` value was set.

```python
def pre_update_params(self):
    if self.lr_decay:
        self.current_learning_rate = self.lr * \
            (1. / (1. + self.lr_decay * self.iterations))
```

Learning rate decay formula:

$$
\eta_{~t} = \eta ~\cdot~\left(\frac{1}{1~+~\text{decay}~\cdot~t}\right)
$$

### `post_update_params(self)`:
Called after all parameter updates. Increments the iteration counter
used for learning rate `lr_decay` and bias correction.

```python
def post_update_params(self):
    self.iterations += 1
```

## Adam Optimizer

Adam optimizer. Combines momentum and RMSprop-style adaptive learning rates,
with bias correction applied to both the first and second moment estimates.
Well suited as a general-purpose optimizer for most neural network tasks.

### `Adam` `__init__(...)`:

| Args | Type | Description |
| --- | --- | --- | 
| `lr` | (float, optional) | Initial learning rate. Default is 0.001. |
| `lr_decay` | (float, optional) | Learning rate `lr_decay` applied each step. Default is 0. |
| `epsilon` | (float, optional) | Small constant for numerical stability in the denominator. Default is 1e-7. |
| `beta_1` | (float, optional) | Exponential `lr_decay` rate for the first moment (momentum) estimate. Default is 0.9. |
| `beta_2` | (float, optional) | Exponential `lr_decay` rate for the second moment (cache) estimate. Default is 0.999. |

### Adam Update Rule `update_params(self, layer)`:

Applies the Adam update rule to a layer's `weights` and `biases`.

Initializes per-layer momentum and cache arrays on the first call.
Computes bias-corrected first and second moment estimates, then
updates parameters using the scaled gradient.

At step $t$, given gradient $g_t$ and parameters $\theta_{t-1}$:

$$
m_t = \underbrace{\beta_1}_{\text{1st moment decay}} \cdot m_{t-1} + (1 - \underbrace{\beta_1}_{\text{beta\_1}}) \cdot \underbrace{g_t}_{\text{gradient at step } t}
$$

For example, setting `beta_1` to a value of 0.9 means we trust a "historical average" 90% of the gradient $g_t$ and the new gradient 10%.

$$
v_t = \underbrace{\beta_2}_{\text{2nd moment decay}} \cdot v_{t-1} + (1 - \underbrace{\beta_2}_{\text{beta\_2}}) \cdot g_t^2
$$

The `beta_2` controls how much history to keep for the squared gradients. For example, having `beta_2` at a value of 0.999 means we trust 99.9% of the running estimate of squared-gradient. 0.01% weight on the newest $g_t^2$.

Example: 
$$m_t = 0.1 g_t + 0.9(0.9 m_{t-2} + 0.1 g_{t-2}) = 0.1 g_t + 0.09 g_{t-1} + 0.081 g_{t-2} + \ldots$$

$$
\underbrace{\hat{m}_t}_{\text{bias-corrected 1st moment}} = \frac{m_t}{1 - \beta_1^t}
$$

$$
\underbrace{\hat{v}_t}_{\text{bias-corrected 2nd moment}} = \frac{v_t}{1 - \beta_2^t}
$$

Bias correction compensates for the fact that the initial estimates of the first and second moments are biased towards zero.

$$
\theta_t = \theta_{t-1} - \underbrace{\eta}_{\text{learning rate}} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \underbrace{\epsilon}_{\text{numerical stability}}}
$$

**Where:**
- $m_t$ - exponential moving average of gradients (momentum)
- $v_t$ - exponential moving average of squared gradients (adaptive scale)
- $\theta_t$ - model parameters at step $t$

**Typical defaults:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

## AdamW Optimizer

AdamW optimizer. A variant of Adam that decouples weight decay from the
gradient update. Instead of folding L2 regularization into the gradient
(which interacts poorly with Adam's adaptive learning rate), AdamW applies
weight decay directly to the parameters as a separate term.

Well suited for training deep networks where generalization matters,
often the default choice over plain Adam in modern practice.

### `AdamW` `__init__(...)`:

| Args | Type | Description |
| --- | --- | --- | 
| `lr` | (float, optional) | Initial learning rate. Default is 0.001. |
| `lr_decay` | (float, optional) | Learning rate `lr_decay` applied each step. Default is 0. |
| `weight_decay` | (float, optional) | Decoupled weight decay coefficient applied each step. Default is 0. |
| `epsilon` | (float, optional) | Small constant for numerical stability in the denominator. Default is 1e-7. |
| `beta_1` | (float, optional) | Exponential `lr_decay` rate for the first moment (momentum) estimate. Default is 0.9. |
| `beta_2` | (float, optional) | Exponential `lr_decay` rate for the second moment (cache) estimate. Default is 0.999. |

### AdamW Update Rule `update_params(self, layer)`:

Applies the AdamW update rule to a layer's `weights` and `biases`.

Initializes per-layer momentum and cache arrays on the first call.
Computes bias-corrected first and second moment estimates, then updates
parameters using the scaled gradient plus a decoupled weight decay term
applied only to weights (not biases).

At step $t$, given gradient $g_t$ and parameters $\theta_{t-1}$:

**First and second moments (same as Adam):**

$$
m_t = \underbrace{\beta_1}_{\text{1st moment decay}} \cdot m_{t-1} + (1 - \beta_1) \cdot \underbrace{g_t}_{\text{gradient at step } t}
$$

$$
v_t = \underbrace{\beta_2}_{\text{2nd moment decay}} \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

**Bias-corrected estimates:**

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
\qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**Parameter update (weights - with decoupled weight decay):**

$$
\theta_t = \theta_{t-1} - \eta \cdot \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \underbrace{\lambda}_{\text{weight\_decay}} \cdot \theta_{t-1} \right)
$$

**Parameter update (biases - no weight decay):**

$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

The key difference from Adam: the $\lambda \cdot \theta_{t-1}$ term is applied
directly to the parameters rather than added to the gradient $g_t$ before
the moment updates. This keeps weight decay independent of the adaptive
scaling by $\sqrt{\hat{v}_t}$, which is why AdamW generalizes better than
Adam with L2 regularization in practice.

Weight decay is only applied to weights, not biases, decaying biases
rarely helps and can hurt performance.

---

**Where:**
- $m_t$ - exponential moving average of gradients (momentum)
- $v_t$ - exponential moving average of squared gradients (adaptive scale)
- $\theta_t$ - model parameters at step $t$
- $\lambda$ - weight decay coefficient (`weight_decay`)

**Typical defaults:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\lambda = 0.01$.

**Reference:** Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"* (ICLR 2019).

## SGD Optimizer

Stochastic Gradient Descent optimizer with optional momentum and learning rate `lr_decay`.

Without momentum, parameters are updated by subtracting the scaled gradient directly.
With momentum, a velocity term accumulates and smooths updates across steps.

### `SGD` `__init__(...)`:

| Args | Type | Description |
| --- | --- | --- | 
| `lr` | (float, optional) | Initial learning rate. Default is 0.1. |
| `lr_decay` | (float, optional) | Learning rate `lr_decay` applied each step. Default is 0. |
| `momentum` | (float, optional) | Momentum coefficient. Set to 0 to disable. Default is 0. |

### SGD Update Rule `update_params(self, layer)`:

Applies the SGD update rule to a layer's weights and biases.

If momentum is enabled, initializes per-layer velocity arrays on the
first call and accumulates gradients over steps. Otherwise performs a
standard gradient descent step.

$$
\theta_t = \theta_{t-1} - \eta ~\cdot~ g_t
$$

### SGD Update Rule with Momentum:

$$
m_{\theta} = m ~\cdot~ m_{\theta_{t-1}} - \eta ~\cdot~ g_t
$$

$$
\theta_t = \theta_{t-1} + m_{\theta}
$$

**Where:**
- $m_{\theta}$ - current parameter's momentum
- $\theta_t$ - model parameters at step $t$
- $\eta$ - learning rate

## AdaGrad Optimizer

AdaGrad optimizer. Adapts the learning rate per parameter by accumulating
the sum of squared gradients. Parameters with large historical gradients
receive smaller updates over time.

Well suited for sparse data but may cause the learning rate to shrink too
aggressively on long training runs.

### `AdaGrad` `__init__(...)`:

| Args | Type | Description |
| --- | --- | --- | 
| `lr` | (float, optional) | Initial learning rate. Default is 0.01. |
| `lr_decay` | (float, optional) | Learning rate `lr_decay` applied each step. Default is 0. |
| `epsilon` | (float, optional) | Small constant for numerical stability in the denominator. Default is 1e-7. |

### AdaGrad Update Rule `update_params(self, layer)`:

Applies the AdaGrad update rule to a layer's weights and biases.

Initializes per-layer cache arrays on the first call. Accumulates
squared gradients into the cache and scales the update accordingly.

Initializes `weight_cache` and `bias_cache` when `AdaGrad` is set with `Model.set(optimizer=AdaGrad(...), ...)`. 

$$
v_t = v_{t-1} + g_t^2
$$

$$
\theta_t = \theta_{t-1} - \underbrace{\eta}_{\text{learning rate}} \cdot \frac{g_t}{\sqrt{v_t} + \underbrace{\epsilon}_{\text{numerical stability}}}
$$

**Where:**
- $v_t$ - accumulated sum of squared gradients (adaptive scale)
- $g_t$ - gradient of the loss w.r.t. $\theta_{t-1}$
- $\theta_t$ - model parameters at step $t$

## RMSprop Optimizer

RMSprop optimizer. Maintains a decaying average of squared gradients
to normalize the update step, preventing the learning rate from
shrinking indefinitely as AdaGrad does.

### `RMSprop` `__init__(...)`:

| Args | Type | Description |
| --- | --- | --- | 
| `lr` | (float, optional) | Initial learning rate. Default is 0.001. |
| `lr_decay` | (float, optional) | Learning rate `lr_decay` applied each step. Default is 0. |
| `epsilon` | (float, optional) | Small constant for numerical stability in the denominator. Default is 1e-7. |
| `beta` | (float, optional) | Exponential `lr_decay` rate for the squared-gradient cache. Default is 0.9. |

### RMSprop Update Rule `update_params(self, layer)`:

Applies the RMSprop update rule to a layer's weights and biases.

Initializes per-layer cache arrays on the first call. Updates the
exponentially decaying average of squared gradients and scales the
parameter update by the root of the cache.

Initializes `weight_cache` and `bias_cache` when `RMSprop` is set with `Model.set(optimizer=RMSprop(...), ...)`. 


At step $t$, given gradient $g_t$ and parameters $\theta_{t-1}$:

**Cache update (exponential moving average of squared gradients):**

$$
v_t = \underbrace{\beta}_{\text{decay}} \cdot v_{t-1} + (1 - \beta) \cdot \underbrace{g_t^2}_{\text{squared gradient}}
$$

Setting $\beta = 0.9$ means we trust the historical average of squared
gradients 90% and the new squared gradient 10%. Expanding the recursion
shows the exponential decay:

$$
v_t = 0.1\,g_t^2 + 0.09\,g_{t-1}^2 + 0.081\,g_{t-2}^2 + \ldots
$$

**Parameter update:**

$$
\theta_t = \theta_{t-1} - \underbrace{\eta}_{\text{learning rate}} \cdot \frac{g_t}{\sqrt{v_t} + \underbrace{\epsilon}_{\text{numerical stability}}}
$$

Dividing by $\sqrt{v_t}$ gives each parameter its own adaptive step size:
parameters with consistently large gradients get smaller updates, and
parameters with small gradients get larger ones. Unlike AdaGrad, the
exponential decay prevents the effective learning rate from shrinking to
zero over time.

---

**Where:**
- $v_t$ - exponential moving average of squared gradients (the cache)
- $g_t$ - gradient of the loss w.r.t. $\theta_{t-1}$
- $\theta_t$ - model parameters at step $t$

**Typical defaults:** $\beta = 0.9$, $\epsilon = 10^{-8}$.