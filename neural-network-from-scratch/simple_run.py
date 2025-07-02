"""
Script simple pour exécuter rapidement le réseau
"""
import numpy as np
from xor_example import solve_xor

def quick_test():
    """Test rapide du réseau"""
    print("🚀 TEST RAPIDE DU RÉSEAU DE NEURONES")
    print("=" * 50)
    
    # Fixer la graine pour des résultats reproductibles
    np.random.seed(42)
    
    # Exécuter XOR
    network = solve_xor()
    
    print("\n✅ Test terminé avec succès!")
    return network

if __name__ == "__main__":
    quick_test()
