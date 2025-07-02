"""
Interface graphique simple pour tester le réseau
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from network import Network
from layers import FullyConnectedLayer, ActivationLayer
from activation_functions import Sigmoid
from loss_functions import MeanSquaredError

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Réseau de Neurones - Démonstration XOR")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        self.network = None
        self.setup_ui()
        
    def setup_ui(self):
        """Configurer l'interface utilisateur"""
        # Titre
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(title_frame, text="🧠 Réseau de Neurones XOR", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack()
        
        # Section d'entraînement
        train_frame = tk.LabelFrame(self.root, text="Entraînement", 
                                   font=('Arial', 12, 'bold'), bg='#f0f0f0')
        train_frame.pack(pady=10, padx=20, fill='x')
        
        # Paramètres d'entraînement
        params_frame = tk.Frame(train_frame, bg='#f0f0f0')
        params_frame.pack(pady=5)
        
        tk.Label(params_frame, text="Époques:", bg='#f0f0f0').grid(row=0, column=0, padx=5)
        self.epochs_var = tk.StringVar(value="2000")
        tk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)
        
        tk.Label(params_frame, text="Taux d'apprentissage:", bg='#f0f0f0').grid(row=0, column=2, padx=5)
        self.lr_var = tk.StringVar(value="0.5")
        tk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=0, column=3, padx=5)
        
        # Bouton d'entraînement
        self.train_button = tk.Button(train_frame, text="🚀 Entraîner le Réseau", 
                                     command=self.train_network, bg='#4CAF50', fg='white',
                                     font=('Arial', 10, 'bold'))
        self.train_button.pack(pady=10)
        
        # Barre de progression
        self.progress = ttk.Progressbar(train_frame, mode='indeterminate')
        self.progress.pack(pady=5, padx=20, fill='x')
        
        # Status
        self.status_var = tk.StringVar(value="Prêt à entraîner")
        status_label = tk.Label(train_frame, textvariable=self.status_var, 
                               bg='#f0f0f0', font=('Arial', 9))
        status_label.pack()
        
        # Section de test
        test_frame = tk.LabelFrame(self.root, text="Test Interactif", 
                                  font=('Arial', 12, 'bold'), bg='#f0f0f0')
        test_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Entrées de test
        input_frame = tk.Frame(test_frame, bg='#f0f0f0')
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Entrée X1:", bg='#f0f0f0').grid(row=0, column=0, padx=5)
        self.x1_var = tk.StringVar(value="0")
        x1_entry = tk.Entry(input_frame, textvariable=self.x1_var, width=10)
        x1_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(input_frame, text="Entrée X2:", bg='#f0f0f0').grid(row=0, column=2, padx=5)
        self.x2_var = tk.StringVar(value="0")
        x2_entry = tk.Entry(input_frame, textvariable=self.x2_var, width=10)
        x2_entry.grid(row=0, column=3, padx=5)
        
        # Bouton de test
        test_button = tk.Button(input_frame, text="🎯 Tester", 
                               command=self.test_network, bg='#2196F3', fg='white',
                               font=('Arial', 10, 'bold'))
        test_button.grid(row=0, column=4, padx=10)
        
        # Boutons de test rapide XOR
        quick_frame = tk.Frame(test_frame, bg='#f0f0f0')
        quick_frame.pack(pady=10)
        
        tk.Label(quick_frame, text="Tests rapides XOR:", bg='#f0f0f0').pack()
        
        buttons_frame = tk.Frame(quick_frame, bg='#f0f0f0')
        buttons_frame.pack(pady=5)
        
        for i, (x1, x2) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            btn = tk.Button(buttons_frame, text=f"[{x1},{x2}]", 
                           command=lambda a=x1, b=x2: self.quick_test(a, b),
                           width=8)
            btn.grid(row=0, column=i, padx=2)
        
        # Zone de résultats
        self.result_text = tk.Text(test_frame, height=10, width=60, 
                                  font=('Courier', 10), bg='#ffffff')
        self.result_text.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Scrollbar pour les résultats
        scrollbar = tk.Scrollbar(test_frame, command=self.result_text.yview)
        self.result_text.config(yscrollcommand=scrollbar.set)
        
    def train_network(self):
        """Entraîner le réseau de neurones"""
        try:
            epochs = int(self.epochs_var.get())
            learning_rate = float(self.lr_var.get())
            
            self.train_button.config(state='disabled')
            self.progress.start()
            self.status_var.set("Entraînement en cours...")
            self.root.update()
            
            # Créer le réseau
            self.network = Network()
            self.network.add_layer(FullyConnectedLayer(2, 4))
            self.network.add_layer(ActivationLayer(Sigmoid()))
            self.network.add_layer(FullyConnectedLayer(4, 1))
            self.network.add_layer(ActivationLayer(Sigmoid()))
            self.network.set_loss_function(MeanSquaredError())
            
            # Données XOR
            x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
            y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
            
            # Entraîner
            self.network.fit(x_train, y_train, epochs=epochs, 
                           learning_rate=learning_rate, verbose=False)
            
            # Test automatique
            predictions = self.network.predict(x_train)
            correct = 0
            
            result_text = "🎉 ENTRAÎNEMENT TERMINÉ !\n"
            result_text += "=" * 40 + "\n"
            result_text += f"Époques: {epochs}\n"
            result_text += f"Taux d'apprentissage: {learning_rate}\n\n"
            result_text += "Tests automatiques XOR:\n"
            result_text += "-" * 30 + "\n"
            
            for i in range(4):
                input_val = x_train[i].flatten()
                expected = y_train[i].flatten()[0]
                predicted = predictions[i].flatten()[0]
                predicted_class = 1 if predicted > 0.5 else 0
                is_correct = predicted_class == expected
                correct += is_correct
                
                result_text += f"[{input_val[0]},{input_val[1]}] → "
                result_text += f"Attendu: {int(expected)}, "
                result_text += f"Prédit: {predicted:.4f} ({predicted_class}) "
                result_text += "✅\n" if is_correct else "❌\n"
            
            accuracy = correct / 4 * 100
            result_text += f"\n🎯 Précision: {accuracy:.1f}%\n"
            result_text += "=" * 40 + "\n\n"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            
            self.status_var.set(f"✅ Entraînement terminé - Précision: {accuracy:.1f}%")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'entraînement: {e}")
            self.status_var.set("❌ Erreur d'entraînement")
        finally:
            self.progress.stop()
            self.train_button.config(state='normal')
    
    def test_network(self):
        """Tester le réseau avec des valeurs personnalisées"""
        if self.network is None:
            messagebox.showwarning("Attention", "Veuillez d'abord entraîner le réseau !")
            return
        
        try:
            x1 = float(self.x1_var.get())
            x2 = float(self.x2_var.get())
            
            # Faire la prédiction
            input_data = np.array([[[x1, x2]]])
            prediction = self.network.predict(input_data)[0].flatten()[0]
            predicted_class = 1 if prediction > 0.5 else 0
            
            # XOR réel
            true_xor = int(bool(x1) ^ bool(x2)) if x1 in [0,1] and x2 in [0,1] else "?"
            
            # Afficher le résultat
            result = f"🔍 TEST PERSONNALISÉ\n"
            result += f"Entrée: [{x1}, {x2}]\n"
            result += f"Prédiction brute: {prediction:.6f}\n"
            result += f"Classe prédite: {predicted_class}\n"
            if true_xor != "?":
                result += f"XOR réel: {true_xor}\n"
                result += f"Correct: {'✅' if predicted_class == true_xor else '❌'}\n"
            result += "-" * 30 + "\n\n"
            
            self.result_text.insert(tk.END, result)
            self.result_text.see(tk.END)
            
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des nombres valides !")
    
    def quick_test(self, x1, x2):
        """Test rapide avec valeurs prédéfinies"""
        self.x1_var.set(str(x1))
        self.x2_var.set(str(x2))
        self.test_network()

def main():
    """Lancer l'interface graphique"""
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
