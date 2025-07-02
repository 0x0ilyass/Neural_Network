"""
Script principal pour exécuter et tester le réseau de neurones
"""

from xor_example import solve_xor, test_other_examples
import numpy as np

def main():
    """Fonction principale"""
    print("🧠 RÉSEAU DE NEURONES FROM SCRATCH")
    print("=" * 50)
    
    # Définir la graine pour la reproductibilité
    np.random.seed(42)
    
    try:
        # Résoudre le problème XOR
        network = solve_xor()
        
        # Tests supplémentaires
        test_other_examples()
        
        print("\n✅ Tous les tests ont été exécutés avec succès !")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        raise

if __name__ == "__main__":
    main()
