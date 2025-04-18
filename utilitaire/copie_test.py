import os
import shutil
import random

def copier_images(source_dossier_kiana, source_dossier_autres, base_destination, tailles=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]):
    # Liste des fichiers dans le dossier "kiana"
    images_kiana = [os.path.join(source_dossier_kiana, f) for f in os.listdir(source_dossier_kiana) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Liste des sous-dossiers dans "source_dossier_autres" pour "autres"
    sous_dossiers_autres = [os.path.join(source_dossier_autres, d) for d in os.listdir(source_dossier_autres) if os.path.isdir(os.path.join(source_dossier_autres, d))]
    
    # Liste pour stocker les images de tous les sous-dossiers
    images_autres = []
    
    # Parcours chaque sous-dossier et ajoute les images
    for sous_dossier in sous_dossiers_autres:
        images_autres.extend([os.path.join(sous_dossier, f) for f in os.listdir(sous_dossier) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Vérifie qu'il y a suffisamment d'images pour chaque taille demandée
    if len(images_kiana) < max(tailles) or len(images_autres) < max(tailles):
        print("Il n'y a pas assez d'images dans l'un des dossiers.")
        return

    # Boucle sur les tailles demandées (100, 200, 300, etc.)
    for taille in tailles:
        # Sélectionne aléatoirement les images à copier pour cette taille
        images_kiana_selectionnees = random.sample(images_kiana, taille)
        images_autres_selectionnees = random.sample(images_autres, taille)

        # Crée les dossiers de destination pour cette taille
        destination_dossier_kiana = os.path.join(base_destination, f'tests_{taille}', 'kiana')
        destination_dossier_autres = os.path.join(base_destination, f'tests_{taille}', 'autres')
        os.makedirs(destination_dossier_kiana, exist_ok=True)
        os.makedirs(destination_dossier_autres, exist_ok=True)

        # Copie les images sélectionnées dans les dossiers de destination
        for image in images_kiana_selectionnees:
            chemin_source = image  # Utilisation du chemin complet de l'image
            chemin_destination = os.path.join(destination_dossier_kiana, os.path.basename(image))
            shutil.copy(chemin_source, chemin_destination)
            print(f"Image {os.path.basename(image)} copiée dans le dossier Kiana (tests_{taille}).")

        for image in images_autres_selectionnees:
            chemin_source = image  # Utilisation du chemin complet de l'image
            chemin_destination = os.path.join(destination_dossier_autres, os.path.basename(image))
            shutil.copy(chemin_source, chemin_destination)
            print(f"Image {os.path.basename(image)} copiée dans le dossier Autres (tests_{taille}).")

# Utilisation du script
source_dossier_kiana = "./imgs_test/kiana"  # Remplace par le chemin de ton dossier contenant les images de Kiana
source_dossier_autres = "./imgs_test/autre"  # Remplace par le chemin du dossier contenant plusieurs sous-dossiers pour les images "autres"
base_destination = "Tests"  # Dossier de base où les sous-dossiers seront créés (tests_100, tests_200, etc.)

copier_images(source_dossier_kiana, source_dossier_autres, base_destination)