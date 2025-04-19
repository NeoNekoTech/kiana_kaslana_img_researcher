import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import json
from pathlib import Path

def create_model_and_train():
    # Configuration des transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Configuration des chemins de sauvegarde
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    last_checkpoint = checkpoint_dir / "last_checkpoint.pth"
    training_stats_file = checkpoint_dir / "training_stats.json"

    # Chargement des données
    dataset = datasets.ImageFolder("faces_detected", transform=transform)
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
        batch_size=32, 
        shuffle=True, 
        pin_memory=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        pin_memory=True,
        num_workers=0
    )

    class Kclassifier(nn.Module):
        def __init__(self, num_classes=13):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=9, padding=4)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=7, padding=3)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            
            self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))
            
            self.fc1 = nn.Linear(128 * 3 * 3, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, num_classes)
            
            self.dropout = nn.Dropout(0.5)
            self.flatten = nn.Flatten()

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            
            x = self.adaptive_pool(x)
            x = self.flatten(x)
            
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    num_epochs = 50
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
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
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
            
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
                print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
                
                # Sauvegarde des statistiques de validation
                training_stats['val_loss'].append(val_loss)
                training_stats['val_acc'].append(val_acc)
                
                # Sauvegarde du checkpoint
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
    torch.save(model.state_dict(), "face_classifier_model.pth")
    print("Modèle final sauvegardé!")

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