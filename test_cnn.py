from PIL import Image
import torch
import time
import torch.nn as nn 
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image

from test_image import Kclassifier


def  get_head(image_path):
    model_path = hf_hub_download(repo_id="Bingsu/adetailer", filename="face_yolov8s.pt")
    # Charger le modèle de détection de visages anime
    model = YOLO(model_path)
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
    
    all_faces = []

    for i, result in enumerate(results):
        boxes = result.boxes
        for j, box in enumerate(boxes):
            try:
                # Récupérer les coordonnées
                coordinates = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coordinates)
                                
                # Recadrer l'image et convertir en RGB
                face_image = original_image.crop((x1, y1, x2, y2)).convert('RGB')
                all_faces.append(face_image)
            except Exception as e:
                print(f"Erreur lors du traitement du visage dans {image_path}: {str(e)}")
                continue
    return all_faces

def test_single_image(image_path):
    # Configuration des transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Chargement du modèle
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Kclassifier(num_classes=14) 
    model.load_state_dict(torch.load("model_1.0.1.pth", map_location=device))
    model = model.to(device)  # Déplacer le modèle sur le device
    model.eval()
    
    all_faces = get_head(image_path)
    # Chargement et transformation de l'image
    for image in all_faces:
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        # Prédiction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            
        # Chargement des noms de classes
        dataset = datasets.ImageFolder("dataset")
        class_names = dataset.classes
        
        # Affichage du résultat
        predicted_class = class_names[predicted.item()]
        print(f"L'image est classée comme : {predicted_class}")
        
        # Affichage des probabilités pour chaque classe
        probabilities = F.softmax(output, dim=1)[0]
        for i, prob in enumerate(probabilities):
            print(f"{class_names[i]}: {prob.item()*100:.2f}%")
            
        # Affichage de l'image
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"Prédiction: {predicted_class}")
        plt.axis('off')
        plt.show()

def plot_training_stats(training_stats):
    epochs = training_stats['epochs']
    train_loss = training_stats['train_loss']
    val_loss = training_stats['val_loss']
    train_acc = training_stats['train_acc']
    val_acc = training_stats['val_acc']

    plt.figure(figsize=(15,10))
    
    # Plot des pertes
    plt.subplot(2,1,1)
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red')
    plt.title("Loss Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot des précisions
    plt.subplot(2,1,2)
    plt.plot(epochs, train_acc, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')
    plt.title("Accuracy Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.close()

def create_model_and_train():
    # Configuration des transformations
    transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAutocontrast(),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    # Configuration des chemins de sauvegarde
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    last_checkpoint = checkpoint_dir / "last_checkpoint.pth"
    training_stats_file = checkpoint_dir / "training_stats.json"

    # Chargement des données
    dataset = datasets.ImageFolder("dataset", transform=transform)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [0.8, 0.2], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Nombre d'échantillons d'entraînement: {len(train_dataset)}")
    print(f"Nombre d'échantillons de test: {len(test_dataset)}")
    print(f"Nombre de classes: {len(dataset.classes)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64,  # Augmentation de la taille du batch
        shuffle=True, 
        pin_memory=True,
        num_workers=4  # Augmentation du nombre de workers si votre CPU le permet
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        pin_memory=True,
        num_workers=0
    )

    class Kclassifier(nn.Module):
        def __init__(self, num_classes=14):
            super().__init__()
            # Feature extraction layers
            self.features = nn.Sequential(
                # Premier bloc
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                # Deuxième bloc
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                # Troisième bloc
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                # Quatrième bloc
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            
            # Classification layers
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(512 * 4 * 4, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # Configuration du device et CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"CUDA est disponible : {torch.cuda.is_available()}")
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
        print(f"Nombre de GPUs : {torch.cuda.device_count()}")

    print(f"Utilisation du device: {device}")

    # Création et déplacement du modèle sur GPU
    model = Kclassifier(num_classes=len(dataset.classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    num_epochs = 100  # Augmentation du nombre d'époques
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    # Variables pour le suivi de l'entraînement
    start_epoch = 0
    best_val_acc = 0
    training_stats = {
        'epochs': [], 
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': []
    }

    # Chargement du dernier checkpoint si existant
    if last_checkpoint.exists():
        print("Chargement du dernier checkpoint...")
        checkpoint = torch.load(last_checkpoint)
        
        model = Kclassifier(num_classes=len(dataset.classes))
        model = model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Initialisation du scheduler ici pour tous les cas
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        # Variables pour le suivi de l'entraînement
        start_epoch = 0
        best_val_acc = 0
        training_stats = {
            'epochs': [], 
            'train_loss': [], 
            'train_acc': [], 
            'val_loss': [], 
            'val_acc': []
        }
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        
        if training_stats_file.exists():
            with open(training_stats_file, 'r') as f:
                training_stats = json.load(f)
        
        print(f"Reprise de l'entraînement depuis l'époque {start_epoch}")

    print(f"Modèle sur GPU: {next(model.parameters()).is_cuda}")
    print(f"Mémoire GPU utilisée: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    # Boucle d'entraînement
    try:
        start_time = time.time()
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            total_batches = len(train_loader)
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Calculer le temps estimé restant
                elapsed_time = time.time() - start_time
                progress = (epoch * total_batches + batch_idx + 1) / (num_epochs * total_batches)
                if progress > 0:
                    estimated_total_time = elapsed_time / progress
                    remaining_time = estimated_total_time - elapsed_time
                    
                    # Convertir en heures:minutes:secondes
                    hours = int(remaining_time // 3600)
                    minutes = int((remaining_time % 3600) // 60)
                    seconds = int(remaining_time % 60)
                    
                    print(f'\rÉpoque {epoch+1}/{num_epochs} '
                          f'[{batch_idx+1}/{total_batches}] '
                          f'Temps restant estimé: {hours:02d}:{minutes:02d}:{seconds:02d}', 
                          end='')
            
            epoch_time = time.time() - epoch_start_time
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            print(f'\nÉpoque {epoch+1}/{num_epochs}, '
                  f'Loss: {epoch_loss:.4f}, '
                  f'Accuracy: {epoch_acc:.2f}%, '
                  f'Temps: {epoch_time:.2f}s')
            
            # Sauvegarde des statistiques
            training_stats['epochs'].append(epoch+1)
            training_stats['train_loss'].append(epoch_loss)
            training_stats['train_acc'].append(epoch_acc)

            # Validation tous les 5 epochs
            if (epoch + 1) % 5 == 0:
                model.eval()
                
                with torch.no_grad():
                    val_loss = 0.0
                    correct = 0
                    total = 0
                    
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    val_loss = val_loss / len(test_loader)
                    val_acc = 100 * correct / total
                    
                    scheduler.step(val_loss)
                            
                    # Enregistrement des statistiques de validation
                    training_stats['val_loss'].append(val_loss)
                    training_stats['val_acc'].append(val_acc)
                    
                    print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
                
                model.train()
                # Sauvegarde du checkpoint à chaque 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': max(best_val_acc, val_acc)
                }, last_checkpoint)
                
                # Sauvegarde des statistiques
                with open(training_stats_file, 'w') as f:
                    json.dump(training_stats, f)
                
                print(f"Checkpoint sauvegardé à l'époque {epoch+1}")

    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur. Sauvegarde du dernier état...")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }, last_checkpoint)
        with open(training_stats_file, 'w') as f:
            json.dump(training_stats, f)
        print("État sauvegardé. Vous pourrez reprendre l'entraînement plus tard.")
        return

    print("Entraînement terminé!")

    # Sauvegarde du modèle final
    torch.save(model, "CNN.pth")
    print("Modèle final sauvegardé!")
    
    plot_training_stats(training_stats)
    print("Graphiques sauvegardés dans 'training_plots.png'")

    # Test final
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f'Résultats finaux - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')

if __name__ == '__main__':
    mp.freeze_support()
    create_model_and_train()

    # test_single_image(r"C:\Users\Alexis\OneDrive\Images\ChatGPT Image 5 avr. 2025, 18_50_36.png")
    #get_head(r"jeu_de_validation\autre\Bell cranel\Bell cranel_14.png")
    # OU
    #test_single_image("imgs_test/autre/kiana/kiana_4.jpeg")   
    # OU
    #test_single_image("imgs_test\\autre\\kiana\\kiana_4.jpeg")