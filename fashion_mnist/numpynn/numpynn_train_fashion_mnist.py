import os
import numpy as np
from torchvision import datasets, transforms
from numpynn import *

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root='datasets', train=True,  download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='datasets', train=False, download=True, transform=transform)

X = train_dataset.data.numpy().astype(np.float32)
y = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy().astype(np.float32)
y_test = test_dataset.targets.numpy()

# shuffle data
keys = np.random.permutation(X.shape[0])
X = X[keys]
y = y[keys]

# normalize between -1 and 1
X = (X.reshape(X.shape[0], -1) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1) - 127.5) / 127.5

neurons = 128

############################ Initialize model

model = Sequential(
    Dense(X.shape[1], neurons),
    ReLU(),
    Dense(neurons, neurons),
    ReLU(),
    Dense(neurons, 10),
    Softmax(),
)

model.set(loss=CategoricalCrossEntropy(),
          optimizer=Adam(learning_rate=0.001
                         , decay=5e-5
                         , epsilon=1e-7
                         , beta_1=0.9
                         , beta_2=0.999),
          accuracy=Accuracy_Categorical())

model.finalize()

model.train(X, y, validation_data=(X_test, y_test),
            batch_size=128, print_every=100, epochs=5)

model.evaluate(X_test, y_test)

# save model
if os.environ("SAVE") == 1:
    model.save("models/numpynn_fashion_mnist.model")
    os.makedirs('data_history', exist_ok=True)
    np.savez('data_history/numpynn_fashion_mnist_history.npz',
            train_loss=np.array(model.history['train_loss']),
            train_acc=np.array(model.history['train_acc']),
            val_loss=np.array(model.history['val_loss']),
            val_acc=np.array(model.history['val_acc']),
            lr=np.array(model.history['lr']))
