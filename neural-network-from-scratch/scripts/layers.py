import numpy as np
from activation_functions import *

class Layer:
    """Classe de base pour toutes les couches"""
    def forward_propagation(self, input_data):
        raise NotImplementedError
    
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

class FullyConnectedLayer(Layer):
    """Couche entièrement connectée (Dense Layer)"""
    
    def __init__(self, input_size, output_size):
        # Initialisation des poids avec Xavier/Glorot
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.bias = np.zeros((1, output_size))
        
    def forward_propagation(self, input_data):
        """Propagation avant: Y = X * W + B"""
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        """
        Propagation arrière selon les formules du cours:
        - ∂E/∂W = X^T * ∂E/∂Y
        - ∂E/∂B = ∂E/∂Y
        - ∂E/∂X = ∂E/∂Y * W^T
        """
        # Calcul des gradients
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)
        
        # Mise à jour des paramètres
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        
        return input_error

class ActivationLayer(Layer):
    """Couche d'activation"""
    
    def __init__(self, activation_function):
        self.activation_function = activation_function
    
    def forward_propagation(self, input_data):
        """Propagation avant: Y = f(X)"""
        self.input = input_data
        self.output = self.activation_function.activation(input_data)
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        """
        Propagation arrière: ∂E/∂X = ∂E/∂Y * f'(X)
        """
        return output_error * self.activation_function.derivative(self.input)
