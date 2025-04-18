from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import os

# Télécharger le modèle depuis Hugging Face
model_path = hf_hub_download(repo_id="Bingsu/adetailer", filename="face_yolov8s.pt")

# Charger le modèle de détection de visages anime
model = YOLO(model_path)

# Dossier racine contenant le dossier "autre"
base_input_dir = './imgs_test/autre'

# Parcourir chaque dossier de personnage dans le dossier "autre"
for character_folder in os.listdir(base_input_dir):
    character_path = os.path.join(base_input_dir, character_folder)
    
    # Vérifier si c'est un dossier
    if os.path.isdir(character_path):
        # Créer le dossier de sortie pour ce personnage
        output_dir = f'./faces_detected/{character_folder}_crop'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Traitement du dossier: {character_folder}")
        
        # Parcourir toutes les images du personnage
        for filename in os.listdir(character_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Chemin complet de l'image source
                image_path = os.path.join(character_path, filename)
                
                try:
                    # Ouvrir l'image originale d'abord pour vérifier qu'elle est valide
                    original_image = Image.open(image_path)
                    original_image.verify()  # Vérifier que l'image est valide
                    
                    # Réouvrir l'image car verify() la ferme
                    original_image = Image.open(image_path)
                    
                    # Faire la détection des visages
                    results = model.predict(
                        source=image_path,
                        save=False,
                        conf=0.3,
                        iou=0.5
                    )

                    # Pour chaque visage détecté
                    for i, result in enumerate(results):
                        boxes = result.boxes
                        for j, box in enumerate(boxes):
                            try:
                                # Récupérer les coordonnées
                                coordinates = box.xyxy[0].tolist()
                                x1, y1, x2, y2 = map(int, coordinates)
                                
                                # Recadrer l'image et convertir en RGB
                                face_image = original_image.crop((x1, y1, x2, y2)).convert('RGB')
                                
                                # Créer le nouveau nom de fichier
                                name_without_ext = os.path.splitext(filename)[0]
                                extension = '.jpg'
                                new_filename = f"{name_without_ext}_crop_{j}{extension}"
                                output_path = os.path.join(output_dir, new_filename)
                                
                                # Sauvegarder l'image recadrée
                                face_image.save(output_path, 'JPEG')
                                
                                print(f"Visage détecté dans autre/{character_folder}/{filename} -> sauvegardé comme {new_filename}")
                                print(f"Dimensions: {x2-x1}x{y2-y1} pixels")
                            except Exception as e:
                                print(f"Erreur lors du traitement du visage dans {filename}: {str(e)}")
                                continue
                                
                except Exception as e:
                    print(f"ERREUR: Impossible de traiter l'image {filename}: {str(e)}")
                    continue

        print(f"Traitement terminé pour {character_folder}")
print("Traitement terminé pour tous les personnages")