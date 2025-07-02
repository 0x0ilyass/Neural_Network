import numpy as np
from layers import *
from loss_functions import *
from activation_functions import *

class Network:
    """Classe principale du réseau de neurones"""
    
    def __init__(self):
        self.layers = []
        self.loss_function = None
    
    def add_layer(self, layer):
        """Ajouter une couche au réseau"""
        self.layers.append(layer)
    
    def set_loss_function(self, loss_function):
        """Définir la fonction de perte"""
        self.loss_function = loss_function
    
    def predict(self, input_data):
        """Prédiction (propagation avant uniquement)"""
        samples = len(input_data)
        result = []
        
        # Propagation avant pour chaque échantillon
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        
        return result
    
    def fit(self, x_train, y_train, epochs, learning_rate, verbose=True):
        """Entraînement du réseau"""
        samples = len(x_train)
        
        for epoch in range(epochs):
            total_error = 0
            
            for j in range(samples):
                # Propagation avant
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                # Calcul de l'erreur
                total_error += self.loss_function.loss(y_train[j], output)
                
                # Propagation arrière
                error = self.loss_function.derivative(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            
            # Affichage des résultats
            if verbose and (epoch + 1) % 100 == 0:
                avg_error = total_error / samples
                print(f'Epoch {epoch + 1}/{epochs}, Error: {avg_error:.6f}')
    
    def evaluate(self, x_test, y_test):
        """Évaluation du réseau"""
        predictions = self.predict(x_test)
        total_error = 0
        
        for i in range(len(x_test)):
            total_error += self.loss_function.loss(y_test[i], predictions[i])
        
        avg_error = total_error / len(x_test)
        return avg_error, predictions
