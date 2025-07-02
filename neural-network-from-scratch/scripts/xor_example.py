import numpy as np
from network import Network
from layers import FullyConnectedLayer, ActivationLayer
from activation_functions import Sigmoid
from loss_functions import MeanSquaredError

def solve_xor():
    """Résoudre le problème XOR avec un réseau de neurones"""
    
    print("=== RÉSOLUTION DU PROBLÈME XOR ===\n")
    
    # Données d'entraînement XOR
    x_train = np.array([
        [[0, 0]],
        [[0, 1]],
        [[1, 0]],
        [[1, 1]]
    ])
    
    y_train = np.array([
        [[0]],  # 0 XOR 0 = 0
        [[1]],  # 0 XOR 1 = 1
        [[1]],  # 1 XOR 0 = 1
        [[0]]   # 1 XOR 1 = 0
    ])
    
    print("Données d'entraînement:")
    for i in range(len(x_train)):
        print(f"Input: {x_train[i].flatten()}, Expected Output: {y_train[i].flatten()}")
    print()
    
    # Construction du réseau
    # Architecture: 2 -> 4 -> 1 (avec activation sigmoid)
    network = Network()
    
    # Couche cachée: 2 entrées -> 4 neurones
    network.add_layer(FullyConnectedLayer(2, 4))
    network.add_layer(ActivationLayer(Sigmoid()))
    
    # Couche de sortie: 4 neurones -> 1 sortie
    network.add_layer(FullyConnectedLayer(4, 1))
    network.add_layer(ActivationLayer(Sigmoid()))
    
    # Fonction de perte
    network.set_loss_function(MeanSquaredError())
    
    print("Architecture du réseau:")
    print("Input Layer: 2 neurones")
    print("Hidden Layer: 4 neurones (Sigmoid)")
    print("Output Layer: 1 neurone (Sigmoid)")
    print("Loss Function: Mean Squared Error")
    print()
    
    # Entraînement
    print("Début de l'entraînement...")
    network.fit(x_train, y_train, epochs=2000, learning_rate=0.5)
    print()
    
    # Test du réseau
    print("=== RÉSULTATS APRÈS ENTRAÎNEMENT ===")
    predictions = network.predict(x_train)
    
    print("Prédictions:")
    for i in range(len(x_train)):
        input_val = x_train[i].flatten()
        expected = y_train[i].flatten()[0]
        predicted = predictions[i].flatten()[0]
        print(f"Input: {input_val}, Expected: {expected}, Predicted: {predicted:.4f}")
    
    # Évaluation
    error, _ = network.evaluate(x_train, y_train)
    print(f"\nErreur moyenne finale: {error:.6f}")
    
    # Test de classification
    print("\n=== TEST DE CLASSIFICATION ===")
    threshold = 0.5
    correct = 0
    
    for i in range(len(x_train)):
        input_val = x_train[i].flatten()
        expected = y_train[i].flatten()[0]
        predicted = predictions[i].flatten()[0]
        predicted_class = 1 if predicted > threshold else 0
        
        is_correct = predicted_class == expected
        correct += is_correct
        
        print(f"Input: {input_val}, Expected: {int(expected)}, "
              f"Predicted: {predicted_class}, Correct: {is_correct}")
    
    accuracy = correct / len(x_train) * 100
    print(f"\nPrecision: {accuracy:.1f}%")
    
    return network

def test_other_examples():
    """Tester d'autres exemples avec le réseau entraîné"""
    print("\n=== TEST AVEC D'AUTRES ARCHITECTURES ===")
    
    # Test avec une architecture plus simple: 2 -> 3 -> 1
    print("\nTest avec architecture 2 -> 3 -> 1:")
    
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    
    network2 = Network()
    network2.add_layer(FullyConnectedLayer(2, 3))
    network2.add_layer(ActivationLayer(Sigmoid()))
    network2.add_layer(FullyConnectedLayer(3, 1))
    network2.add_layer(ActivationLayer(Sigmoid()))
    network2.set_loss_function(MeanSquaredError())
    
    network2.fit(x_train, y_train, epochs=1500, learning_rate=0.7, verbose=False)
    
    predictions2 = network2.predict(x_train)
    for i in range(len(x_train)):
        input_val = x_train[i].flatten()
        expected = y_train[i].flatten()[0]
        predicted = predictions2[i].flatten()[0]
        print(f"Input: {input_val}, Expected: {expected}, Predicted: {predicted:.4f}")

if __name__ == "__main__":
    # Résoudre XOR
    trained_network = solve_xor()
    
    # Tester d'autres architectures
    test_other_examples()
    
    print("\n=== RÉSEAU DE NEURONES IMPLÉMENTÉ AVEC SUCCÈS ===")
    print("Le réseau peut maintenant résoudre le problème XOR !")
