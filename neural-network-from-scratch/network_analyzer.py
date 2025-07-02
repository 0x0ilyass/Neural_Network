import numpy as np
import matplotlib.pyplot as plt
from network import Network
from layers import FullyConnectedLayer, ActivationLayer
from activation_functions import Sigmoid, Tanh, ReLU
from loss_functions import MeanSquaredError

def analyze_network_weights(network, title="Analyse des Poids"):
    """Analyser et visualiser les poids du réseau"""
    fig, axes = plt.subplots(1, len([l for l in network.layers if hasattr(l, 'weights')]), 
                            figsize=(15, 4))
    
    if len([l for l in network.layers if hasattr(l, 'weights')]) == 1:
        axes = [axes]
    
    layer_idx = 0
    for i, layer in enumerate(network.layers):
        if hasattr(layer, 'weights'):
            im = axes[layer_idx].imshow(layer.weights, cmap='RdBu', aspect='auto')
            axes[layer_idx].set_title(f'Couche {i+1} - Poids')
            axes[layer_idx].set_xlabel('Neurone de sortie')
            axes[layer_idx].set_ylabel('Neurone d\'entrée')
            plt.colorbar(im, ax=axes[layer_idx])
            layer_idx += 1
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def compare_activation_functions():
    """Comparer différentes fonctions d'activation"""
    print("🔬 COMPARAISON DES FONCTIONS D'ACTIVATION")
    print("=" * 60)
    
    # Données XOR
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    
    activations = {
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh(),
        'ReLU': ReLU()
    }
    
    results = {}
    
    for name, activation in activations.items():
        print(f"\n🧪 Test avec {name}...")
        
        # Créer le réseau
        network = Network()
        network.add_layer(FullyConnectedLayer(2, 4))
        network.add_layer(ActivationLayer(activation))
        network.add_layer(FullyConnectedLayer(4, 1))
        network.add_layer(ActivationLayer(Sigmoid()))  # Sigmoid en sortie pour XOR
        network.set_loss_function(MeanSquaredError())
        
        # Entraîner
        network.fit(x_train, y_train, epochs=1000, learning_rate=0.5, verbose=False)
        
        # Évaluer
        predictions = network.predict(x_train)
        correct = 0
        
        for i in range(len(x_train)):
            expected = y_train[i].flatten()[0]
            predicted = predictions[i].flatten()[0]
            predicted_class = 1 if predicted > 0.5 else 0
            correct += (predicted_class == expected)
        
        accuracy = correct / len(x_train) * 100
        results[name] = accuracy
        
        print(f"   Précision: {accuracy:.1f}%")
    
    # Visualiser les résultats
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accuracies = list(results.values())
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    bars = plt.bar(names, accuracies, color=colors, alpha=0.8)
    plt.ylabel('Précision (%)')
    plt.title('Comparaison des Fonctions d\'Activation sur XOR')
    plt.ylim(0, 110)
    
    # Ajouter les valeurs sur les barres
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results

def test_different_architectures():
    """Tester différentes architectures de réseau"""
    print("🏗️ TEST DE DIFFÉRENTES ARCHITECTURES")
    print("=" * 60)
    
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    
    architectures = [
        ([2, 3, 1], "Simple: 2→3→1"),
        ([2, 4, 1], "Standard: 2→4→1"),
        ([2, 6, 1], "Large: 2→6→1"),
        ([2, 4, 2, 1], "Deep: 2→4→2→1"),
        ([2, 8, 4, 1], "Very Deep: 2→8→4→1")
    ]
    
    results = {}
    
    for arch, name in architectures:
        print(f"\n🔧 Test architecture: {name}")
        
        # Créer le réseau
        network = Network()
        
        for i in range(len(arch) - 1):
            network.add_layer(FullyConnectedLayer(arch[i], arch[i+1]))
            network.add_layer(ActivationLayer(Sigmoid()))
        
        network.set_loss_function(MeanSquaredError())
        
        # Entraîner
        network.fit(x_train, y_train, epochs=1500, learning_rate=0.5, verbose=False)
        
        # Évaluer
        predictions = network.predict(x_train)
        correct = 0
        
        for i in range(len(x_train)):
            expected = y_train[i].flatten()[0]
            predicted = predictions[i].flatten()[0]
            predicted_class = 1 if predicted > 0.5 else 0
            correct += (predicted_class == expected)
        
        accuracy = correct / len(x_train) * 100
        results[name] = accuracy
        
        print(f"   Précision: {accuracy:.1f}%")
    
    # Visualiser
    plt.figure(figsize=(12, 6))
    names = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(range(len(names)), accuracies, color='lightcoral', alpha=0.8)
    plt.ylabel('Précision (%)')
    plt.title('Comparaison des Architectures de Réseau')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylim(0, 110)
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results

def main_analysis():
    """Fonction principale d'analyse"""
    print("🚀 ANALYSE COMPLÈTE DU RÉSEAU DE NEURONES")
    print("=" * 70)
    
    # 1. Entraînement avec visualisation
    print("\n1️⃣ ENTRAÎNEMENT AVEC VISUALISATION")
    from visualize_training import visualize_xor_training
    trained_network = visualize_xor_training()
    
    # 2. Analyse des poids
    print("\n2️⃣ ANALYSE DES POIDS")
    analyze_network_weights(trained_network, "Poids après entraînement XOR")
    
    # 3. Comparaison des fonctions d'activation
    print("\n3️⃣ COMPARAISON DES FONCTIONS D'ACTIVATION")
    activation_results = compare_activation_functions()
    
    # 4. Test des architectures
    print("\n4️⃣ TEST DES ARCHITECTURES")
    architecture_results = test_different_architectures()
    
    print("\n" + "="*70)
    print("✅ ANALYSE TERMINÉE!")
    print("Toutes les fenêtres graphiques ont été générées.")
    print("="*70)

if __name__ == "__main__":
    main_analysis()
