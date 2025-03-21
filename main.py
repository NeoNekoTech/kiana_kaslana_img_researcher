import tkinter as tk
from tkinter import filedialog, messagebox
from utilitaire.rename_files import rename_files
from utilitaire.copie_test import copier_images
from NNT_tools import convert_and_rename_images, anti_doublon

class UtilitaireApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Utilitaire d'Images")

        # Bouton pour renommer les fichiers
        self.rename_button = tk.Button(root, text="Renommer les fichiers", command=self.rename_files)
        self.rename_button.pack(pady=10)

        # Bouton pour copier les images
        self.copy_button = tk.Button(root, text="Copier les images", command=self.copy_images)
        self.copy_button.pack(pady=10)

        # Bouton pour supprimer les doublons
        self.deduplicate_button = tk.Button(root, text="Supprimer les doublons", command=self.remove_duplicates)
        self.deduplicate_button.pack(pady=10)

    def rename_files(self):
        directory = filedialog.askdirectory(title="Sélectionnez un dossier pour renommer les fichiers")
        if directory:
            try:
                convert_and_rename_images(directory)
                messagebox.showinfo("Succès", "Les fichiers ont été renommés avec succès.")
            except Exception as e:
                messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

    def copy_images(self):
        source_kiana = filedialog.askdirectory(title="Sélectionnez le dossier source pour Kiana")
        source_autres = filedialog.askdirectory(title="Sélectionnez le dossier source pour Autres")
        destination = filedialog.askdirectory(title="Sélectionnez le dossier de destination")
        if source_kiana and source_autres and destination:
            try:
                copier_images(source_kiana, source_autres, destination)
                messagebox.showinfo("Succès", "Les images ont été copiées avec succès.")
            except Exception as e:
                messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

    def remove_duplicates(self):
        directory = filedialog.askdirectory(title="Sélectionnez un dossier pour supprimer les doublons")
        if directory:
            try:
                anti_doublon(directory)
                messagebox.showinfo("Succès", "Les doublons ont été supprimés avec succès.")
            except Exception as e:
                messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = UtilitaireApp(root)
    root.mainloop()