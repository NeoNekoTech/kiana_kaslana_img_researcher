import os

# Spécifiez le chemin du dossier principal
dossier_principal = './imgs_test/autre/'

# Liste des extensions d'image à traiter
extensions_image = ['.jpg', '.jpeg', '.png', '.webp']

# Parcourir tous les dossiers et sous-dossiers
for dossier_racine, sous_dossiers, fichiers in os.walk(dossier_principal):
    # Récupérer le nom du dossier actuel
    nom_dossier = os.path.basename(dossier_racine)
    
    # Filtrer les fichiers pour ne garder que les images
    images = [f for f in fichiers if any(f.lower().endswith(ext) for ext in extensions_image)]
    
    # Renommer les images
    for i, image in enumerate(images, 1):
        # Création du nouveau nom pour l'image (nom du dossier + numéro)
        nouveau_nom = f'{nom_dossier}_{i}{os.path.splitext(image)[1]}'
        
        # Chemins complets pour l'ancien et le nouveau fichier
        ancien_chemin = os.path.join(dossier_racine, image)
        nouveau_chemin = os.path.join(dossier_racine, nouveau_nom)
        
        # Renommer le fichier
        os.rename(ancien_chemin, nouveau_chemin)
        print(f'{image} renome en {nouveau_nom}')
