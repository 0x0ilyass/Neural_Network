"""
Visualiseur simple du réseau (sans matplotlib pour éviter les dépendances)
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import numpy as np
from network import Network
from layers import FullyConnectedLayer, ActivationLayer
from activation_functions import Sigmoid
from loss_functions import MeanSquaredError

def print_network_architecture(network):
    """Afficher l'architecture du réseau en ASCII"""
    print("🏗️ ARCHITECTURE DU RÉSEAU")
    print("=" * 40)
    
    layer_info = []
    for i, layer in enumerate(network.layers):
        if hasattr(layer, 'weights'):
            input_size, output_size = layer.weights.shape
            layer_info.append(f"FC({input_size}→{output_size})")
        elif hasattr(layer, 'activation_function'):
            func_name = layer.activation_function.__class__.__name__
            layer_info.append(f"Act({func_name})")
    
    # Affichage ASCII
    print("Input")
    for i, info in enumerate(layer_info):
        print("  ↓")
        print(f"Layer {i+1}: {info}")
    print("  ↓")
    print("Output")
    print()

def visualize_weights_ascii(network):
    """Visualiser les poids en ASCII"""
    print("⚖️ VISUALISATION DES POIDS")
    print("=" * 40)
    
    layer_num = 1
    for i, layer in enumerate(network.layers):
        if hasattr(layer, 'weights'):
            print(f"\n🔗 Couche {layer_num} - Poids:")
            weights = layer.weights
            
            # Normaliser pour l'affichage
            w_min, w_max = weights.min(), weights.max()
            if w_max != w_min:
                weights_norm = (weights - w_min) / (w_max - w_min)
            else:
                weights_norm = weights
            
            # Affichage ASCII
            chars = " .-+*#"
            for row in weights_norm:
                line = ""
                for val in row:
                    char_idx = int(val * (len(chars) - 1))
                    line += chars[char_idx] + " "
                print(f"   {line}")
            
            print(f"   Min: {w_min:.3f}, Max: {w_max:.3f}")
            layer_num += 1

def trace_forward_pass(network, input_data):
    """Tracer une passe avant étape par étape"""
    print("🔍 TRACE DE LA PROPAGATION AVANT")
    print("=" * 50)
    
    current_data = input_data
    print(f"📥 Entrée: {current_data.flatten()}")
    
    for i, layer in enumerate(network.layers):
        if hasattr(layer, 'weights'):
            # Couche fully connected
            output = layer.forward_propagation(current_data)
            print(f"🔗 Couche FC {i+1}: {current_data.flatten()} → {output.flatten()}")
            current_data = output
        elif hasattr(layer, 'activation_function'):
            # Couche d'activation
            output = layer.forward_propagation(current_data)
            func_name = layer.activation_function.__class__.__name__
            print(f"⚡ Activation {func_name}: {current_data.flatten()} → {output.flatten()}")
            current_data = output
    
    print(f"📤 Sortie finale: {current_data.flatten()}")
    return current_data

def analyze_trained_network():
    """Analyser un réseau entraîné"""
    print("🔬 ANALYSE DÉTAILLÉE DU RÉSEAU ENTRAÎNÉ")
    print("=" * 60)
    
    # Créer et entraîner le réseau
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    
    network = Network()
    network.add_layer(FullyConnectedLayer(2, 4))
    network.add_layer(ActivationLayer(Sigmoid()))
    network.add_layer(FullyConnectedLayer(4, 1))
    network.add_layer(ActivationLayer(Sigmoid()))
    network.set_loss_function(MeanSquaredError())
    
    print("🔄 Entraînement en cours...")
    network.fit(x_train, y_train, epochs=2000, learning_rate=0.5, verbose=False)
    print("✅ Entraînement terminé!\n")
    
    # Analyser l'architecture
    print_network_architecture(network)
    
    # Visualiser les poids
    visualize_weights_ascii(network)
    
    # Tracer quelques exemples
    print("\n" + "="*50)
    test_cases = [
        np.array([[[0, 0]]]),
        np.array([[[0, 1]]]),
        np.array([[[1, 0]]]),
        np.array([[[1, 1]]])
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📋 Test case {i+1}:")
        result = trace_forward_pass(network, test_case)
        expected = y_train[i].flatten()[0]
        predicted_class = 1 if result.flatten()[0] > 0.5 else 0
        print(f"   Attendu: {expected}, Prédit: {predicted_class} {'✅' if predicted_class == expected else '❌'}")

if __name__ == "__main__":
    analyze_trained_network()
