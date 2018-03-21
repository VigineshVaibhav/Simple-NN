import numpy as np


class Neural_Network(object):
    def __init__(self):
        # Define Hyperparameters
        self.yHat = 0
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = self.sigmoid(self.z3)
        return self.yHat

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        return (np.exp(-z)) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self, x, y):
        self.yHat = self.forward(x)
        J = 0.5 * sum((y - self.yHat) ** 2)
        return J

    def costFunctionPrime(self, x, y):
        self.yHat = self.forward(x)
        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        print("dJdW2=", dJdW2)

        # print(delta3.shape, self.W2.shape)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        x = np.array(x).T.tolist()
        # print(np.shape(x))
        dJdW1 = np.dot(x, delta2)
        print("dJdW1=", dJdW1)
        return dJdW1, dJdW2


X = [[0 for x in range(2)] for y in range(3)]

print(X)
NN = Neural_Network()
for x in range(3):
    for y in range(2):
        print("Enter value for row ", x + 1, " and column ", y + 1, ":")
        X[x][y] = float(input())
yHat = NN.forward(X)
print("yHat is:", yHat)
Y = [[0.75], [0.82], [0.93]]

oldCost = NN.costFunction(X, Y)
print("Old Cost is:", oldCost)

factor = 3
for i in range(100):
    dJdW1, dJdW2 = NN.costFunctionPrime(X, Y)
    print("dJdW1 =", dJdW1)
    print("dJdW2 =", dJdW2)
    NN.W1 = NN.W1 - factor * dJdW1
    NN.W2 = NN.W2 - factor * dJdW2
    newCost = NN.costFunction(X, Y)

print("Final new cost:", newCost)

print("Predicted output is:", NN.forward(X))
print("Actual output is:", Y)
