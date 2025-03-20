import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from utilitaire.rename_files import rename_files
from utilitaire.copie_test import copier_images
from NNT_tools import convert_and_rename_images, anti_doublon

class UtilitaireApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Utilitaire d'Images")

        # Bouton pour choisir une commande
        self.command_button = tk.Button(root, text="Choisir une commande", command=self.choose_command)
        self.command_button.pack(pady=10)

    def choose_command(self):
        # Liste des commandes disponibles
        commands = {
            "Renommer les fichiers": self.rename_files,
            "Copier les images": self.copy_images,
            "Supprimer les doublons": self.remove_duplicates
        }

        # Demander à l'utilisateur de choisir une commande
        command_name = simpledialog.askstring(
            "Choisir une commande",
            "Entrez une commande :\n- Renommer les fichiers\n- Copier les images\n- Supprimer les doublons"
        )

        if command_name in commands:
            # Exécuter la commande choisie
            commands[command_name]()
        else:
            messagebox.showerror("Erreur", "Commande invalide. Veuillez réessayer.")

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