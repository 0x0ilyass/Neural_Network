import numpy as np

class ActivationFunction:
    """Classe de base pour les fonctions d'activation"""
    def activation(self, x):
        raise NotImplementedError
    
    def derivative(self, x):
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    def activation(self, x):
        # Éviter l'overflow en limitant x
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        s = self.activation(x)
        return s * (1 - s)

class Tanh(ActivationFunction):
    def activation(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

class ReLU(ActivationFunction):
    def activation(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return (x > 0).astype(float)

class Linear(ActivationFunction):
    def activation(self, x):
        return x
    
    def derivative(self, x):
        return np.ones_like(x)
