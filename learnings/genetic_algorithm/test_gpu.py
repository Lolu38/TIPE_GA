import torch

try:
    # On crée deux gros tenseurs directement sur la RTX 5070
    a = torch.randn(10000, 10000, device='cuda')
    b = torch.randn(10000, 10000, device='cuda')
    
    # On fait une multiplication de matrice (l'opération de base de l'IA et de ton AG)
    c = torch.matmul(a, b)
    
    print("✅ Calcul réussi sur le GPU !")
    print(f"Architecture utilisée : {torch.cuda.get_device_capability(0)}")
except Exception as e:
    print("❌ Le calcul a échoué. Détails de l'erreur :")
    print(e)