# Simple Neural Network Using C++

This project is a multiclass neural network implemented completely from scratch in C++ using only the standard library.

Features:
- ReLU hidden activation
- Softmax output layer
- Cross entropy loss
- Backpropagation
- Feature normalization
- Iris classification

# Architecture:
- 4 input features
- 1 hidden layer composed of 3 neurons
- 3 output neurons
  - setosa
  - versicolor
  - virginica
 
# How it works:
Input features are normalized using min-max normalization.  
The normalized inputs are passed through a hidden layer where each neuron computes:
 - weighted sum
 - bias addition
 - ReLU activation

The hidden layer outputs are passed into the output layer.  
The output layer computes class scores, then applies softmax to convert them into probabilities.  
Cross entropy loss measures how far the predictions are from the expected outputs.  
Backpropagation computes gradients for each weight and bias.  
Gradient descent updates the parameters to reduce loss over time.  
After training, the network predicts the Iris flower class with the highest probability.  

# Example Output:

```txt
Epoch 0 Loss: 1.13686  
Epoch 500 Loss: 0.308032  
Epoch 1000 Loss: 0.10051  
Epoch 1500 Loss: 0.0451959  
Epoch 2000 Loss: 0.0266198  
Epoch 2500 Loss: 0.0181159  
Epoch 2999 Loss: 0.0134324  
0.995891 0.00410894 6.63628e-13  
setosa  
0.00165148 0.959784 0.0385648  
versicolor  
1.1422e-08 0.00331291 0.996687  
virginica  
4.05229e-05 0.244594 0.755365  
virginica
```

# Concepts Implemented:
- Forward pass
- Backpropagation
- ReLU
- Softmax
- Cross entropy loss
- Multiclass classification
- Gradient descent

This project was built for learning purposes to understand neural networks internally before using prebuilt libraries.
