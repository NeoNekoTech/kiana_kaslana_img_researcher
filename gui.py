import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from test_image import Kclassifier
from pathlib import Path

class ClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificateur d'Images")
        self.root.geometry("800x600")

        # Frame principale
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bouton pour sélectionner une image
        self.select_btn = tk.Button(self.main_frame, text="Sélectionner une image", command=self.select_image)
        self.select_btn.pack(pady=10)

        # Label pour afficher l'image
        self.image_label = tk.Label(self.main_frame)
        self.image_label.pack(pady=10)

        # Frame pour les prédictions
        self.pred_frame = tk.Frame(self.main_frame)
        self.pred_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Label pour la prédiction principale
        self.pred_label = tk.Label(self.pred_frame, text="", font=("Arial", 12, "bold"))
        self.pred_label.pack()

        # Text widget pour les probabilités
        self.prob_text = tk.Text(self.pred_frame, height=10, width=40)
        self.prob_text.pack(pady=5)

        # Chargement du modèle
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Kclassifier(num_classes=14)
        self.model.load_state_dict(torch.load("face_classifier_model.pth"))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Configuration des transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Chargement des noms de classes
        self.dataset = datasets.ImageFolder("faces_detected")
        self.class_names = self.dataset.classes

    def select_image(self):
        # Ouvrir le sélecteur de fichiers
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )

        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        # Charger et afficher l'image
        image = Image.open(image_path).convert('RGB')
        # Redimensionner l'image pour l'affichage
        display_image = image.copy()
        display_image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

        # Prédiction
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            probabilities = F.softmax(output, dim=1)[0]

        # Afficher la prédiction principale
        predicted_class = self.class_names[predicted.item()]
        self.pred_label.config(text=f"Prédiction: {predicted_class}")

        # Afficher toutes les probabilités
        self.prob_text.delete(1.0, tk.END)
        for i, prob in enumerate(probabilities):
            self.prob_text.insert(tk.END, f"{self.class_names[i]}: {prob.item()*100:.2f}%\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClassifierGUI(root)
    root.mainloop()