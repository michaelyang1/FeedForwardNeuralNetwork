# FeedForwardNeuralNetwork
This is a multi-layer perceptron/feed-forward neural network library complete with various optimizers and regularizers. 

Prerequisites

Before you continue, ensure you have met the following requirements: 

* You have installed Python 3
* You have installed NumPy 
* You have downloaded the MNIST database (only for demo in test.py to run; you can skip this step and use your own dataset if you so desire) 
* You have a basic understanding of feed forward neural networks 

Installation 

1) Install Python with the NumPy library 
2) Download the mnist_train.csv and mnist_test.csv files from https://www.kaggle.com/oddrationale/mnist-in-csv 
3) Place both files in the project folder 

Usage 

* To build the network, create an object instance of the NeuralNetwork class 
* To train the network, call the train function from the NeuralNetwork object instance 
* The cost functions supported by this library are cross entropy and mean squared error cost 
* The activation functions supported are rectified linear units and sigmoids 
* The regularizers supported are L2 regularization and dropout 
* The optimizers supported are adam (adaptive moment estimation), momentum, nesterov acclerated momentum, and learning rate decay 

Contact Information 

* Email : myang1394@gmail.com
