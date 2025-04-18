from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Télécharger le modèle depuis Hugging Face
model_path = hf_hub_download(repo_id="Bingsu/adetailer", filename="face_yolov8s.pt")

# Charger le modèle de détection de visages anime
model = YOLO(model_path)

# Faire la détection des visages
results = model.predict(
    source='./imgs_test/kiana/kiana_1027.png',
    save=True,
    conf=0.3,
    iou=0.5
)