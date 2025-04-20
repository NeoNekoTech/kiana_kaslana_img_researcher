import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

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

def test_single_image(image_path):
    # Configuration des transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Chargement du modèle
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Kclassifier(num_classes=13) 
    model.load_state_dict(torch.load("face_classifier_model.pth"))
    model = model.to(device)
    model.eval()
    
    # Chargement et transformation de l'image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        
    # Chargement des noms de classes
    dataset = datasets.ImageFolder("faces_detected")
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
    
if __name__ == '__main__':

    # Pour tester une image
    test_single_image("imgs_test\autre\kiana\kiana_2.jpg")  