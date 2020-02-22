from NeuralNetwork import *

if __name__ == '__main__':
    # A neural network with a input layer of 784 neurons, 1 hidden layer of 100 neurons, and output layer of 10 neurons
    # The network uses cross entropy cost, sigmoid activation function, a dropout retention rate of 0.9 for input layer 
    # ... and 0.5 for hidden layers. 
    # For training, parameters of 1000 epochs, a mini batch size of 10, a learning rate of 0.1, and a l2 value 1.0 were used here.
    NN = NeuralNetwork([784, 100, 10], cost='cross_entropy', activation_fn='sigmoid', dropout=(0.9, 0.5))
    NN.train('mnist_train.csv', 'mnist_test.csv', 1000, 10, 0.1, lmbda=1.0)
