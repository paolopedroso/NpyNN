from numpynn import *
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./datasets', train=True,  download=True, transform=transform)
test_dataset = datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)

# keep only 0 and 1
train_mask = train_dataset.targets <= 1
test_mask = test_dataset.targets <= 1

X = train_dataset.data[train_mask].numpy().astype(np.float32)
y = train_dataset.targets[train_mask].numpy()
X_test = test_dataset.data[test_mask].numpy().astype(np.float32)
y_test = test_dataset.targets[test_mask].numpy()

# shuffle
keys = np.random.permutation(X.shape[0])
X = X[keys]
y = y[keys]

# normalize between -1 and 1
X = (X.reshape(X.shape[0], -1) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1) - 127.5) / 127.5

neurons = 16

model = Sequential(
    Dense(X.shape[1], neurons),
    ReLU(),
    Dense(neurons, 1),
    Sigmoid()
)

model.set(loss=BinaryCrossEntropy(),
          optimizer=AdaGrad(learning_rate=0.01,
                         decay=5e-5),
          accuracy=Accuracy_Categorical(binary=True))
model.finalize()

model.train(X, y, validation_data=(X_test, y_test),
            batch_size=128, print_every=100, epochs=3)
