import os
import re
from PIL import Image

def convert_and_rename_images(directory):
    # Liste tous les fichiers du répertoire spécifié
    for filename in os.listdir(directory):
        # On ne prend que les fichiers avec une extension d'image
        if filename.lower().endswith(('.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            # Création du chemin complet du fichier
            filepath = os.path.join(directory, filename)
            
            # Ouverture de l'image
            with Image.open(filepath) as img:
                # On crée le nouveau nom de fichier avec l'extension PNG
                base_filename = os.path.splitext(filename)[0]
                new_filename = base_filename + ".png"
                new_filepath = os.path.join(directory, new_filename)
                
                # Conversion de l'image en PNG et sauvegarde
                img.save(new_filepath, "PNG")
                print(f"Image {filename} convertie et sauvegardée sous {new_filename}")
                
                # Utilisation d'une expression régulière pour extraire le numéro du fichier
                match = re.search(r'(\d+)', base_filename)  # Cherche une séquence de chiffres
                if match:
                    number = int(match.group(1))  # Convertit le numéro trouvé en entier
                    new_number = number + 1  # Incrémente le numéro
                    
                    # Nouvelle nomenclature
                    new_base_filename = f"images{new_number:05d}.png"
                    new_filepath_renamed = os.path.join(directory, new_base_filename)

                    # Vérification si le fichier existe déjà, et incrémentation si nécessaire
                    while os.path.exists(new_filepath_renamed):
                        new_number += 1
                        new_base_filename = f"images{new_number:05d}.png"
                        new_filepath_renamed = os.path.join(directory, new_base_filename)
                    
                    # Renommer le fichier
                    os.rename(new_filepath, new_filepath_renamed)
                    print(f"Image renommée en {new_base_filename}")
                    
                    # Supprimer l'ancien fichier (source)
                    os.remove(filepath)
                    print(f"Ancien fichier supprimé : {filename}")
                else:
                    print(f"Aucun numéro trouvé dans {filename}, pas de renommage effectué.")

# Spécifie le répertoire contenant tes images
directory = "./imgs_test/kiana"
convert_and_rename_images(directory)