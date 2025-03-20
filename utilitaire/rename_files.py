import os

def rename_files(directory):
    for root, _, files in os.walk(directory):
        for index, file in enumerate(files, start=1):
            ancien_chemin = os.path.join(root, file)
            extension = os.path.splitext(file)[1]
            nouveau_nom = f"{os.path.basename(root)}_{index}{extension}"
            nouveau_chemin = os.path.join(root, nouveau_nom)

            # Vérifier si le fichier existe déjà
            compteur = 1
            while os.path.exists(nouveau_chemin):
                nouveau_nom = f"{os.path.basename(root)}_{index}_{compteur}{extension}"
                nouveau_chemin = os.path.join(root, nouveau_nom)
                compteur += 1

            os.rename(ancien_chemin, nouveau_chemin)
            print(f"{ancien_chemin} renommé en {nouveau_chemin}")