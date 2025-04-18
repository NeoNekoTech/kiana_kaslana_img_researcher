import os
import re
import numpy as np 
from PIL import Image

def resize_image(img, size=(780, 780)):
    """
    Redimensionne une image à une taille spécifique en gardant les proportions
    @param:
        img: Image PIL à redimensionner
        size: tuple (width, height) taille désirée
    @return: Image PIL redimensionnée
    """
    ratio = min(size[0]/img.size[0], size[1]/img.size[1])
    new_size = tuple([int(x*ratio) for x in img.size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Créer une nouvelle image avec fond blanc
    new_img = Image.new("RGB", size, (255, 255, 255))
    # Coller l'image redimensionnée au centre
    offset = ((size[0] - new_size[0]) // 2,
             (size[1] - new_size[1]) // 2)
    new_img.paste(img, offset)
    return new_img

def convert_and_rename_images(directory):
    """
    @param:
        directory :str: chemin vers le dossier ou se trouve les images a rename
    """
    # Obtenir le nom du dossier
    folder_name = os.path.basename(directory).replace('_crop', '')
    number = 0
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
            filepath = os.path.join(directory, filename)
            
            try:
                with Image.open(filepath) as img:
                    # Convertir en RGB si nécessaire
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Redimensionner l'image
                    img = resize_image(img, (780, 780))
                    
                    # Créer le nouveau nom de fichier avec le nom du dossier
                    new_number = number + 1
                    new_base_filename = f"{folder_name}_{new_number:05d}.png"
                    new_filepath = os.path.join(directory, new_base_filename)
                    
                    # Vérifier si le fichier existe déjà
                    while os.path.exists(new_filepath):
                        new_number += 1
                        new_base_filename = f"{folder_name}_{new_number:05d}.png"
                        new_filepath = os.path.join(directory, new_base_filename)
                    
                    # Sauvegarder l'image redimensionnée
                    img.save(new_filepath, "PNG", quality=95)
                    print(f"Image {filename} convertie et sauvegardée sous {new_base_filename}")
                    
                    # Supprimer l'ancien fichier si différent
                    if filepath != new_filepath:
                        os.remove(filepath)
                        print(f"Ancien fichier supprimé : {filename}")
                    
                    number = new_number
                    
            except Exception as e:
                print(f"Erreur lors du traitement de {filename}: {str(e)}")
                continue

def anti_doublon(directory):
    """
    @param:
        directory :str: chemin du dossier où l'on veut enlever les doublons
    """
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for fileref in files:
        img_ref_path = os.path.join(directory, fileref)
        for filetest in files:
            if fileref != filetest:
                img_test_path = os.path.join(directory, filetest)
                try:
                    with open(img_ref_path, "rb") as img1, open(img_test_path, "rb") as img2:
                        if img1.read() == img2.read():
                            print("del: ", img_test_path)
                            img1.close()
                            img2.close()
                            os.remove(img_test_path)
                except (FileNotFoundError, PermissionError):
                    pass

def process_all_directories(base_directory, convert=False):
    """
    Traite tous les dossiers dans imgs_test et ses sous-dossiers
    @param:
        base_directory :str: chemin du dossier racine (imgs_test)
        convert :bool: si True, convertit aussi les images en PNG
    """
    # Vérifier si le dossier existe
    if not os.path.exists(base_directory):
        print(f"Le dossier {base_directory} n'existe pas.")
        return

    # Traiter tous les sous-dossiers
    for root, dirs, files in os.walk(base_directory):
        print(f"Traitement du dossier: {root}")
        if files:  # Ne traiter que s'il y a des fichiers
            if convert:
                convert_and_rename_images(root)
            anti_doublon(root)

# Pour utiliser avec la conversion :
base_directory = "./faces_detected"
process_all_directories(base_directory, convert=True)