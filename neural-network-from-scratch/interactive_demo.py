"""
Démonstration interactive du réseau de neurones
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import numpy as np
from network import Network
from layers import FullyConnectedLayer, ActivationLayer
from activation_functions import Sigmoid
from loss_functions import MeanSquaredError

def create_trained_xor_network():
    """Créer et entraîner un réseau XOR"""
    # Données XOR
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    
    # Créer le réseau
    network = Network()
    network.add_layer(FullyConnectedLayer(2, 4))
    network.add_layer(ActivationLayer(Sigmoid()))
    network.add_layer(FullyConnectedLayer(4, 1))
    network.add_layer(ActivationLayer(Sigmoid()))
    network.set_loss_function(MeanSquaredError())
    
    # Entraîner
    print("🔄 Entraînement du réseau...")
    network.fit(x_train, y_train, epochs=2000, learning_rate=0.5, verbose=False)
    print("✅ Entraînement terminé!")
    
    return network

def interactive_test():
    """Test interactif du réseau"""
    print("🎮 DÉMONSTRATION INTERACTIVE")
    print("=" * 50)
    
    # Créer le réseau entraîné
    network = create_trained_xor_network()
    
    while True:
        print("\n" + "-" * 30)
        print("Testez votre réseau XOR !")
        print("Entrez deux valeurs (0 ou 1) ou 'quit' pour quitter")
        
        try:
            user_input = input("Entrée (format: x y): ").strip().lower()
            
            if user_input == 'quit' or user_input == 'q':
                print("👋 Au revoir !")
                break
            
            # Parser l'entrée
            values = user_input.split()
            if len(values) != 2:
                print("❌ Veuillez entrer exactement 2 valeurs")
                continue
            
            x1, x2 = float(values[0]), float(values[1])
            
            # Faire la prédiction
            input_data = np.array([[[x1, x2]]])
            prediction = network.predict(input_data)[0].flatten()[0]
            predicted_class = 1 if prediction > 0.5 else 0
            
            # Calculer la vraie valeur XOR
            true_xor = int(bool(x1) ^ bool(x2))
            
            # Afficher les résultats
            print(f"📊 Résultats:")
            print(f"   Entrée: [{x1}, {x2}]")
            print(f"   XOR réel: {true_xor}")
            print(f"   Prédiction brute: {prediction:.4f}")
            print(f"   Classe prédite: {predicted_class}")
            print(f"   Correct: {'✅' if predicted_class == true_xor else '❌'}")
            
        except ValueError:
            print("❌ Veuillez entrer des nombres valides")
        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

def benchmark_network():
    """Benchmark de performance du réseau"""
    print("⚡ BENCHMARK DE PERFORMANCE")
    print("=" * 50)
    
    import time
    
    # Test de vitesse d'entraînement
    print("🏃‍♂️ Test de vitesse d'entraînement...")
    
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    
    network = Network()
    network.add_layer(FullyConnectedLayer(2, 4))
    network.add_layer(ActivationLayer(Sigmoid()))
    network.add_layer(FullyConnectedLayer(4, 1))
    network.add_layer(ActivationLayer(Sigmoid()))
    network.set_loss_function(MeanSquaredError())
    
    start_time = time.time()
    network.fit(x_train, y_train, epochs=1000, learning_rate=0.5, verbose=False)
    training_time = time.time() - start_time
    
    # Test de vitesse de prédiction
    print("🎯 Test de vitesse de prédiction...")
    start_time = time.time()
    for _ in range(1000):
        network.predict(x_train)
    prediction_time = time.time() - start_time
    
    # Résultats
    print(f"\n📈 Résultats du benchmark:")
    print(f"   Temps d'entraînement (1000 époques): {training_time:.3f}s")
    print(f"   Temps de prédiction (1000 × 4 échantillons): {prediction_time:.3f}s")
    print(f"   Prédictions par seconde: {4000/prediction_time:.0f}")

def main_demo():
    """Démonstration principale"""
    print("🚀 DÉMONSTRATION COMPLÈTE DU RÉSEAU DE NEURONES")
    print("=" * 70)
    
    while True:
        print("\n🎯 Que voulez-vous faire ?")
        print("1. Test interactif XOR")
        print("2. Benchmark de performance")
        print("3. Entraînement complet avec détails")
        print("4. Quitter")
        
        choice = input("\nVotre choix (1-4): ").strip()
        
        if choice == '1':
            interactive_test()
        elif choice == '2':
            benchmark_network()
        elif choice == '3':
            from main_runner import main
            main()
        elif choice == '4':
            print("👋 Au revoir !")
            break
        else:
            print("❌ Choix invalide, veuillez entrer 1, 2, 3 ou 4")

if __name__ == "__main__":
    main_demo()
