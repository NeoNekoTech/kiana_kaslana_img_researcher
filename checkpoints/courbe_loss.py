import matplotlib.pyplot as plt 
import json
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
stats_file = os.path.join(current_dir, "training_stats.json")

with open(stats_file, "r") as file:
    data = json.loads(file.read())

trainloss = data["train_loss"]
val_loss = data["val_loss"]
epoch = data["epochs"]
acc_train = data["train_acc"]
acc_test = data["val_acc"]

# Calcul des indices pour les données de validation (tous les 5 epochs)
val_indices = np.arange(5, 51, 5)[:len(val_loss)]

plt.figure(figsize=(15, 10))

# Courbe de perte d'entraînement
plt.subplot(2, 2, 1)
plt.plot(epoch, trainloss, 'b-', label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.grid(True)
plt.legend()

# Courbe de perte de validation
plt.subplot(2, 2, 2)
plt.plot(val_indices, val_loss, 'r-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.grid(True)
plt.legend()

# Courbe de précision d'entraînement
plt.subplot(2, 2, 3)
plt.plot(epoch, acc_train, 'g-', label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Train Accuracy')
plt.grid(True)
plt.legend()

# Courbe de précision de validation
plt.subplot(2, 2, 4)
plt.plot(val_indices, acc_test, 'm-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()