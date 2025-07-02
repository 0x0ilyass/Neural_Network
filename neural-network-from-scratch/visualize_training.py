import numpy as np
import matplotlib.pyplot as plt
from network import Network
from layers import FullyConnectedLayer, ActivationLayer
from activation_functions import Sigmoid
from loss_functions import MeanSquaredError

class VisualNetwork(Network):
    """Version du réseau avec visualisation de l'entraînement"""
    
    def __init__(self):
        super().__init__()
        self.training_errors = []
        self.epoch_numbers = []
    
    def fit_with_visualization(self, x_train, y_train, epochs, learning_rate, plot_interval=100):
        """Entraînement avec visualisation en temps réel"""
        samples = len(x_train)
        
        # Configuration du graphique
        plt.ion()  # Mode interactif
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
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
            
            # Enregistrer l'erreur
            avg_error = total_error / samples
            self.training_errors.append(avg_error)
            self.epoch_numbers.append(epoch + 1)
            
            # Mise à jour des graphiques
            if (epoch + 1) % plot_interval == 0 or epoch == epochs - 1:
                self.update_plots(ax1, ax2, x_train, y_train)
                plt.pause(0.01)
                
                print(f'Epoch {epoch + 1}/{epochs}, Error: {avg_error:.6f}')
        
        plt.ioff()
        plt.show()
    
    def update_plots(self, ax1, ax2, x_train, y_train):
        """Mettre à jour les graphiques"""
        # Graphique 1: Évolution de l'erreur
        ax1.clear()
        ax1.plot(self.epoch_numbers, self.training_errors, 'b-', linewidth=2)
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Erreur MSE')
        ax1.set_title('Évolution de l\'erreur pendant l\'entraînement')
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Prédictions actuelles
        ax2.clear()
        predictions = self.predict(x_train)
        
        inputs = []
        expected = []
        predicted = []
        
        for i in range(len(x_train)):
            input_str = f"{x_train[i].flatten()}"
            inputs.append(input_str)
            expected.append(y_train[i].flatten()[0])
            predicted.append(predictions[i].flatten()[0])
        
        x_pos = np.arange(len(inputs))
        width = 0.35
        
        ax2.bar(x_pos - width/2, expected, width, label='Attendu', alpha=0.8, color='green')
        ax2.bar(x_pos + width/2, predicted, width, label='Prédit', alpha=0.8, color='orange')
        
        ax2.set_xlabel('Entrées XOR')
        ax2.set_ylabel('Sortie')
        ax2.set_title('Prédictions actuelles vs Attendues')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(inputs)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

def visualize_xor_training():
    """Entraîner et visualiser le réseau XOR"""
    print("🧠 ENTRAÎNEMENT VISUEL DU RÉSEAU XOR")
    print("=" * 50)
    
    # Données XOR
    x_train = np.array([
        [[0, 0]],
        [[0, 1]],
        [[1, 0]],
        [[1, 1]]
    ])
    
    y_train = np.array([
        [[0]],
        [[1]],
        [[1]],
        [[0]]
    ])
    
    # Créer le réseau avec visualisation
    network = VisualNetwork()
    
    # Architecture: 2 -> 4 -> 1
    network.add_layer(FullyConnectedLayer(2, 4))
    network.add_layer(ActivationLayer(Sigmoid()))
    network.add_layer(FullyConnectedLayer(4, 1))
    network.add_layer(ActivationLayer(Sigmoid()))
    network.set_loss_function(MeanSquaredError())
    
    print("Architecture: 2 → 4 → 1 (Sigmoid)")
    print("Fonction de perte: MSE")
    print("\nDémarrage de l'entraînement avec visualisation...")
    print("Fermez la fenêtre graphique pour continuer.\n")
    
    # Entraînement avec visualisation
    network.fit_with_visualization(x_train, y_train, epochs=2000, learning_rate=0.5, plot_interval=50)
    
    # Résultats finaux
    print("\n" + "="*50)
    print("RÉSULTATS FINAUX")
    print("="*50)
    
    predictions = network.predict(x_train)
    correct = 0
    
    for i in range(len(x_train)):
        input_val = x_train[i].flatten()
        expected = y_train[i].flatten()[0]
        predicted = predictions[i].flatten()[0]
        predicted_class = 1 if predicted > 0.5 else 0
        is_correct = predicted_class == expected
        correct += is_correct
        
        print(f"Input: {input_val}, Expected: {int(expected)}, "
              f"Predicted: {predicted:.4f} → Class: {predicted_class}, "
              f"✓" if is_correct else "✗")
    
    accuracy = correct / len(x_train) * 100
    print(f"\n🎯 Précision finale: {accuracy:.1f}%")
    
    return network

if __name__ == "__main__":
    visualize_xor_training()
