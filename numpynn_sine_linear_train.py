import numpy as np
import matplotlib.pyplot as plt
from numpynn import *

X = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1).astype(np.float32)
y = np.sin(X).astype(np.float32)

# shuffle
keys = np.random.permutation(X.shape[0])
X = X[keys]
y = y[keys]

# train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

neurons = 128

model = Model()
model.add(Dense(X_train.shape[1], neurons))
model.add(ReLU())
model.add(Dense(neurons, neurons))
model.add(ReLU())
model.add(Dense(neurons, 1))
model.add(Linear())

model.set(loss=MeanSquareError(),
          optimizer=RMSprop(learning_rate=0.01,
                            decay=5e-5,
                            epsilon=1e-7,
                            beta=0.9),
          accuracy=Accuracy_Regression())

model.finalize()

model.train(X_train, y_train, validation_data=(X_test, y_test),
            batch_size=64, print_every=100, epochs=1000)

############################ Plot fitted data

X_plot = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1).astype(np.float32)
y_true = np.sin(X_plot)
y_pred = model.predict(X_plot)

plt.figure(figsize=(10, 5))
plt.plot(X_plot, y_true, label='true sin(x)', color='red', linewidth=2)
plt.plot(X_plot, y_pred, label='predicted',   color='blue',    linewidth=2, linestyle='--')
plt.scatter(X_test, y_test, s=5, alpha=0.3, color='gray', label='test points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/sine_regression.png', dpi=150, bbox_inches='tight')
plt.show()
