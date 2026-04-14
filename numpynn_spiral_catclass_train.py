import numpy as np
import matplotlib.pyplot as plt
from numpynn import *

def create_spiral(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype=np.int64)
    for c in range(classes):
        ix = range(samples * c, samples * (c + 1))
        r = np.linspace(0, 1, samples)
        t = np.linspace(c * 4, (c + 1) * 4, samples) + np.random.randn(samples) * 0.5
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X.astype(np.float32), y

# small dataset to make overfitting easier
X, y = create_spiral(samples=200, classes=3)
X += np.random.randn(*X.shape).astype(np.float32) * 0.1

# shuffle
keys = np.random.permutation(X.shape[0])
X = X[keys]
y = y[keys]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

"""
Intentionally overfitting the data.
"""

neurons = 256

model = Model()
model.add(Dense(X_train.shape[1], neurons))
model.add(ReLU())
model.add(Dense(neurons, neurons))
model.add(ReLU())
model.add(Dense(neurons, neurons))
model.add(ReLU())
model.add(Dense(neurons, neurons))
model.add(ReLU())
model.add(Dense(neurons, 3))
model.add(Softmax())

model.set(loss=CategoricalCrossEntropy(),
          optimizer=Adam(learning_rate=0.01,
                         decay=0,
                         epsilon=1e-7,
                         beta_1=0.9,
                         beta_2=0.999),
          accuracy=Accuracy_Categorical())

model.finalize()

model.train(X_train, y_train, validation_data=(X_test, y_test),
            batch_size=32, print_every=50, epochs=100)

############################ Plot

x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
probs = model.predict(grid)
zz = np.argmax(probs, axis=1).reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, zz, alpha=0.3, cmap='brg')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', s=10, alpha=0.8)
plt.title('Spiral Decision Boundary (Overfit)')
plt.tight_layout()
plt.savefig('plots/spiral_overfit.png', dpi=150, bbox_inches='tight')
plt.show()