"""
Fonctionnalités avancées pour le réseau de neurones
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import numpy as np
import json
import pickle
from network import Network
from layers import FullyConnectedLayer, ActivationLayer
from activation_functions import Sigmoid, Tanh, ReLU
from loss_functions import MeanSquaredError

class AdvancedNetwork(Network):
    """Version avancée du réseau avec sauvegarde et métriques"""
    
    def __init__(self):
        super().__init__()
        self.training_history = {
            'epochs': [],
            'errors': [],
            'accuracies': []
        }
    
    def fit_advanced(self, x_train, y_train, x_val=None, y_val=None, 
                    epochs=1000, learning_rate=0.5, patience=50, verbose=True):
        """Entraînement avancé avec validation et early stopping"""
        samples = len(x_train)
        best_val_error = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            total_error = 0
            
            # Entraînement
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                total_error += self.loss_function.loss(y_train[j], output)
                
                error = self.loss_function.derivative(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            
            avg_error = total_error / samples
            
            # Validation
            val_error = avg_error
            if x_val is not None and y_val is not None:
                val_error = self.evaluate(x_val, y_val)[0]
            
            # Enregistrer l'historique
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['errors'].append(avg_error)
            
            # Calculer la précision
            predictions = self.predict(x_train)
            correct = 0
            for i in range(len(x_train)):
                pred_class = 1 if predictions[i].flatten()[0] > 0.5 else 0
                true_class = int(y_train[i].flatten()[0])
                correct += (pred_class == true_class)
            accuracy = correct / len(x_train) * 100
            self.training_history['accuracies'].append(accuracy)
            
            # Early stopping
            if val_error < best_val_error:
                best_val_error = val_error
                patience_counter = 0
                self.save_best_weights()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping à l'époque {epoch + 1}")
                    break
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Error: {avg_error:.6f}, '
                      f'Val Error: {val_error:.6f}, Accuracy: {accuracy:.1f}%')
    
    def save_best_weights(self):
        """Sauvegarder les meilleurs poids"""
        self.best_weights = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                self.best_weights.append({
                    'weights': layer.weights.copy(),
                    'bias': layer.bias.copy()
                })
    
    def load_best_weights(self):
        """Charger les meilleurs poids"""
        if hasattr(self, 'best_weights'):
            weight_idx = 0
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    layer.weights = self.best_weights[weight_idx]['weights'].copy()
                    layer.bias = self.best_weights[weight_idx]['bias'].copy()
                    weight_idx += 1
    
    def save_model(self, filename):
        """Sauvegarder le modèle complet"""
        model_data = {
            'architecture': [],
            'weights': [],
            'training_history': self.training_history
        }
        
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                model_data['architecture'].append({
                    'type': 'FullyConnected',
                    'input_size': layer.weights.shape[0],
                    'output_size': layer.weights.shape[1]
                })
                model_data['weights'].append({
                    'weights': layer.weights.tolist(),
                    'bias': layer.bias.tolist()
                })
            elif hasattr(layer, 'activation_function'):
                model_data['architecture'].append({
                    'type': 'Activation',
                    'function': layer.activation_function.__class__.__name__
                })
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✅ Modèle sauvegardé dans {filename}")
    
    def load_model(self, filename):
        """Charger un modèle sauvegardé"""
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        # Reconstruire l'architecture
        self.layers = []
        weight_idx = 0
        
        activation_map = {
            'Sigmoid': Sigmoid(),
            'Tanh': Tanh(),
            'ReLU': ReLU()
        }
        
        for layer_info in model_data['architecture']:
            if layer_info['type'] == 'FullyConnected':
                layer = FullyConnectedLayer(
                    layer_info['input_size'], 
                    layer_info['output_size']
                )
                layer.weights = np.array(model_data['weights'][weight_idx]['weights'])
                layer.bias = np.array(model_data['weights'][weight_idx]['bias'])
                self.layers.append(layer)
                weight_idx += 1
            elif layer_info['type'] == 'Activation':
                activation = activation_map[layer_info['function']]
                self.layers.append(ActivationLayer(activation))
        
        self.training_history = model_data['training_history']
        print(f"✅ Modèle chargé depuis {filename}")

def create_iris_dataset():
    """Créer un dataset Iris simplifié pour test"""
    # Dataset Iris simplifié (2 classes, 2 features)
    np.random.seed(42)
    
    # Classe 0: Setosa (petites valeurs)
    class0 = np.random.normal([4.5, 2.5], [0.3, 0.2], (25, 2))
    labels0 = np.zeros((25, 1))
    
    # Classe 1: Versicolor (grandes valeurs)  
    class1 = np.random.normal([6.0, 3.5], [0.4, 0.3], (25, 2))
    labels1 = np.ones((25, 1))
    
    # Combiner
    X = np.vstack([class0, class1])
    y = np.vstack([labels0, labels1])
    
    # Normaliser
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Convertir au format du réseau
    X_formatted = X.reshape(50, 1, 2)
    y_formatted = y.reshape(50, 1, 1)
    
    return X_formatted, y_formatted

def test_advanced_features():
    """Tester les fonctionnalités avancées"""
    print("🚀 TEST DES FONCTIONNALITÉS AVANCÉES")
    print("=" * 60)
    
    # 1. Test avec dataset Iris
    print("\n1️⃣ Test avec dataset Iris simplifié")
    X, y = create_iris_dataset()
    
    # Split train/validation
    train_size = 40
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Créer le réseau avancé
    network = AdvancedNetwork()
    network.add_layer(FullyConnectedLayer(2, 6))
    network.add_layer(ActivationLayer(Sigmoid()))
    network.add_layer(FullyConnectedLayer(6, 3))
    network.add_layer(ActivationLayer(Sigmoid()))
    network.add_layer(FullyConnectedLayer(3, 1))
    network.add_layer(ActivationLayer(Sigmoid()))
    network.set_loss_function(MeanSquaredError())
    
    print("Architecture: 2 → 6 → 3 → 1")
    print("Dataset: 40 train, 10 validation")
    
    # Entraînement avancé
    network.fit_advanced(X_train, y_train, X_val, y_val, 
                        epochs=1000, learning_rate=0.3, patience=100)
    
    # Évaluation
    train_error, train_preds = network.evaluate(X_train, y_train)
    val_error, val_preds = network.evaluate(X_val, y_val)
    
    print(f"\n📊 Résultats Iris:")
    print(f"   Erreur train: {train_error:.4f}")
    print(f"   Erreur validation: {val_error:.4f}")
    
    # 2. Sauvegarde et chargement
    print("\n2️⃣ Test de sauvegarde/chargement")
    network.save_model("iris_model.json")
    
    # Créer un nouveau réseau et charger
    new_network = AdvancedNetwork()
    new_network.set_loss_function(MeanSquaredError())
    new_network.load_model("iris_model.json")
    
    # Vérifier que les prédictions sont identiques
    original_preds = network.predict(X_val)
    loaded_preds = new_network.predict(X_val)
    
    diff = np.mean(np.abs(np.array(original_preds) - np.array(loaded_preds)))
    print(f"   Différence après chargement: {diff:.8f}")
    print("   ✅ Sauvegarde/chargement OK" if diff < 1e-10 else "   ❌ Problème de sauvegarde")
    
    # 3. Comparaison d'architectures
    print("\n3️⃣ Comparaison d'architectures sur XOR")
    architectures = [
        ([2, 3, 1], "Minimal"),
        ([2, 4, 1], "Standard"),
        ([2, 6, 1], "Large"),
        ([2, 4, 2, 1], "Deep")
    ]
    
    x_xor = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_xor = np.array([[[0]], [[1]], [[1]], [[0]]])
    
    results = {}
    
    for arch, name in architectures:
        net = AdvancedNetwork()
        
        for i in range(len(arch) - 1):
            net.add_layer(FullyConnectedLayer(arch[i], arch[i+1]))
            net.add_layer(ActivationLayer(Sigmoid()))
        
        net.set_loss_function(MeanSquaredError())
        net.fit_advanced(x_xor, y_xor, epochs=1500, learning_rate=0.5, verbose=False)
        
        # Calculer précision
        preds = net.predict(x_xor)
        correct = sum(1 for i in range(4) 
                     if (preds[i][0][0] > 0.5) == bool(y_xor[i][0][0]))
        accuracy = correct / 4 * 100
        
        results[name] = accuracy
        print(f"   {name:10} ({str(arch):15}): {accuracy:5.1f}%")
    
    print(f"\n🏆 Meilleure architecture: {max(results, key=results.get)}")
    
    return network

def benchmark_comparison():
    """Comparer les performances avec différents paramètres"""
    print("\n⚡ BENCHMARK DE PERFORMANCE")
    print("=" * 50)
    
    import time
    
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    
    configs = [
        (0.1, "Lent"),
        (0.5, "Standard"), 
        (1.0, "Rapide"),
        (2.0, "Très rapide")
    ]
    
    print("Taux d'apprentissage | Temps | Époques | Précision")
    print("-" * 55)
    
    for lr, name in configs:
        network = AdvancedNetwork()
        network.add_layer(FullyConnectedLayer(2, 4))
        network.add_layer(ActivationLayer(Sigmoid()))
        network.add_layer(FullyConnectedLayer(4, 1))
        network.add_layer(ActivationLayer(Sigmoid()))
        network.set_loss_function(MeanSquaredError())
        
        start_time = time.time()
        network.fit_advanced(x_train, y_train, epochs=2000, 
                           learning_rate=lr, verbose=False)
        training_time = time.time() - start_time
        
        # Calculer précision finale
        preds = network.predict(x_train)
        correct = sum(1 for i in range(4) 
                     if (preds[i][0][0] > 0.5) == bool(y_train[i][0][0]))
        accuracy = correct / 4 * 100
        
        epochs_trained = len(network.training_history['epochs'])
        
        print(f"{lr:15.1f} | {training_time:5.2f}s | {epochs_trained:7d} | {accuracy:8.1f}%")

if __name__ == "__main__":
    # Test complet des fonctionnalités avancées
    network = test_advanced_features()
    
    # Benchmark
    benchmark_comparison()
    
    print("\n" + "="*60)
    print("🎉 TOUTES LES FONCTIONNALITÉS AVANCÉES TESTÉES !")
    print("="*60)
