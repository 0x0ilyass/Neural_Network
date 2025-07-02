import numpy as np

class LossFunction:
    """Classe de base pour les fonctions de perte"""
    def loss(self, y_true, y_pred):
        raise NotImplementedError
    
    def derivative(self, y_true, y_pred):
        raise NotImplementedError

class MeanSquaredError(LossFunction):
    """Fonction de perte MSE (Mean Squared Error)"""
    
    def loss(self, y_true, y_pred):
        """MSE = (1/n) * Σ(y_true - y_pred)²"""
        return np.mean(np.power(y_true - y_pred, 2))
    
    def derivative(self, y_true, y_pred):
        """∂MSE/∂y_pred = 2(y_pred - y_true) / n"""
        return 2 * (y_pred - y_true) / y_true.size

class BinaryCrossEntropy(LossFunction):
    """Fonction de perte pour classification binaire"""
    
    def loss(self, y_true, y_pred):
        # Éviter log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def derivative(self, y_true, y_pred):
        # Éviter division par 0
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.size
