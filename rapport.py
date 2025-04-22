import os
from PIL import Image
import torch
import torch.nn.functional as F 
from torchvision import datasets, transforms
from test_image import Kclassifier
from datetime import datetime
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Modification de la partie résultats et formatage
def format_probability(prob_str):
    return f"{float(prob_str.strip('%')):.2f}%"

def create_test_report(test_folder):
    # Modification du chemin des rapports pour être dans le dossier du site
    reports_dir = Path("site/static/rapports")
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    # Création d'un sous-dossier avec une date plus lisible
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y_%Hh%M")
    report_name = f"rapport_analyse_images_{date_str}"
    current_report_dir = reports_dir / report_name
    current_report_dir.mkdir(exist_ok=True)
    
    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Chargement du modèle
    model = Kclassifier(num_classes=14)
    model.load_state_dict(torch.load("CNN.pth", weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    
    # Chargement des noms de classes
    dataset = datasets.ImageFolder("dataset")
    class_names = dataset.classes
    
    results = []
    for root, _, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                
                # Plus besoin de copier l'image car elles sont déjà dans static/images
                # Extraction du label attendu depuis le nom du dossier parent
                expected_label = os.path.basename(os.path.dirname(image_path))
                
                # Prédiction
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = F.softmax(output, dim=1)[0]
                    _, predicted = torch.max(output.data, 1)
                    
                predicted_label = class_names[predicted.item()]
                
                # Création d'un dictionnaire pour les probabilités
                probs_dict = {}
                for i, class_name in enumerate(class_names):
                    prob = probabilities[i].item() * 100
                    if class_name == expected_label:
                        probs_dict[class_name] = {
                            'attendu': '100%',
                            'obtenu': f"{prob:.2f}%",
                            'difference': f"{prob - 100:.2f}%",
                            'is_target': True
                        }
                    else:
                        probs_dict[class_name] = {
                            'attendu': '0%',
                            'obtenu': f"{prob:.2f}%",
                            'difference': f"{prob:.2f}%",
                            'is_target': False
                        }
                
                # Ajout des résultats simplifiés
                results.append({
                    'Image': file,
                    'Label_Attendu': expected_label,
                    'Prediction': predicted_label,
                    'Correct': expected_label.lower() == predicted_label.lower(),
                    'Probabilites': probs_dict,
                })

    
    # Sauvegarde en Excel
    report_name = current_report_dir / "rapport"
    
    # Sauvegarde en Excel
    df = pd.DataFrame(results)
    
    # Sauvegarde en Excel avec données simplifiées
    excel_path = report_name.with_suffix('.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        main_data = df[['Image', 'Label_Attendu', 'Prediction', 'Correct']].copy()
        main_data['Status'] = main_data['Correct'].map({True: '✅', False: '❌'})
        main_data.to_excel(writer, sheet_name='Résumé', index=False)
        
        probs_df = pd.DataFrame([r['Probabilites'] for r in results])
        probs_df.index = df['Image']
        probs_df.to_excel(writer, sheet_name='Probabilités détaillées')
    
    # Création des visualisations
    plt.figure(figsize=(15, 10))
    
    # Premier subplot : graphique en barres
    plt.subplot(2, 2, 1)
    accuracy = (df['Correct'].sum() / len(df)) * 100
    plt.bar(['Corrects', 'Incorrects'], 
            [df['Correct'].sum(), len(df) - df['Correct'].sum()],
            color=['#2ecc71', '#e74c3c'])
    plt.title(f"Résultats des prédictions\nPrécision: {accuracy:.2f}%")
    
    # Deuxième subplot : camembert des prédictions par classe
    plt.subplot(2, 2, 2)
    predictions_count = df['Prediction'].value_counts()
    plt.pie(predictions_count, labels=predictions_count.index, autopct='%1.1f%%')
    plt.title("Répartition des prédictions")
    
    # Sauvegarde du graphique
    stats_path = report_name.with_suffix('.png')
    plt.savefig(stats_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    def format_probabilities(probs_dict):
        # Trier les probabilités par valeur obtenue (décroissant)
        sorted_probs = sorted(
            probs_dict.items(),
            key=lambda x: float(x[1]['obtenu'].strip('%')),
            reverse=True
        )[:3]  # Ne garder que les 3 premiers
        
        html = "<div class='probs-container'>"
        for class_name, values in sorted_probs:
            bg_color = '#e8f5e9' if values['is_target'] else '#ffffff'
            html += f"""
                <div class='prob-item' style='background-color: {bg_color}; padding: 5px; margin: 2px; border-radius: 3px;'>
                    <strong>{class_name}</strong><br>
                    <span class="prob-value">Probabilité: {values['obtenu']}</span><br>
                    {'<span class="prob-target">(Classe attendue)</span>' if values['is_target'] else ''}
                </div>
            """
        html += "</div>"
        return html
    
    css_styles = """
    /* Styles de base */
    body {
        font-family: 'Segoe UI', Arial, sans-serif;
        margin: 0;
        padding: 20px;
        background-color: #f5f5f5;
    }

    /* Container principal */
    .container {
        max-width: 1200px;
        margin: 0 auto;
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
    }

    /* En-têtes */
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 15px;
        border-bottom: 3px solid #3498db;
    }

    h2 {
        color: #34495e;
        margin-top: 30px;
        padding-left: 10px;
        border-left: 4px solid #3498db;
    }

    /* Dashboard */
    .dashboard {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }

    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }

    .big-number {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        margin: 10px 0;
    }

    .stat-label {
        color: #7f8c8d;
        font-size: 14px;
        text-transform: uppercase;
    }

    /* Boîte de résumé */
    .summary-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid #dee2e6;
    }

    .accuracy {
        font-size: 24px;
        color: #2c3e50;
        text-align: center;
        margin: 20px 0;
    }

    .stats-img {
        max-width: 600px;
        display: block;
        margin: 30px auto;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
    }

    /* Grille de prédictions */
    .prediction-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }

    .prediction-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .prediction-card.correct {
        border-left: 4px solid #2ecc71;
    }

    .prediction-card.incorrect {
        border-left: 4px solid #e74c3c;
    }

    /* Images */
    .image-container {
        max-width: 150px;
        margin: 10px auto;
        text-align: center;
    }

    .result-image {
        max-width: 100%;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* Probabilités */
    .probs-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 10px;
        padding: 10px;
    }

    .prob-item {
        font-size: 0.9em;
        border: 1px solid #dee2e6;
        padding: 8px;
        border-radius: 5px;
        transition: all 0.3s ease;
        background-color: white;
    }

    .prob-item:hover {
        transform: scale(1.02);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .prob-item strong {
        color: #2c3e50;
        display: block;
        margin-bottom: 5px;
    }

    /* Tableau */
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        background-color: white;
    }

    th {
        background-color: #3498db;
        color: white;
        padding: 12px;
        text-align: left;
    }

    td {
        padding: 10px;
        border-bottom: 1px solid #dee2e6;
        max-width: 300px;
    }

    tr:hover {
        background-color: #f8f9fa;
    }

    /* États */
    .correct {
        background-color: #d4edda;
        color: #155724;
    }

    .incorrect {
        background-color: #f8d7da;
        color: #721c24;
    }

    /* Boutons */
    .analyze-btn {
        display: inline-block;
        padding: 10px 20px;
        background-color: #3498db;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        margin: 20px 0;
        transition: background-color 0.3s;
    }

    .analyze-btn:hover {
        background-color: #2980b9;
    }
    
    .prob-value {
        font-size: 1.2em;
        color: #2c3e50;
        font-weight: bold;
    }
    
    .prob-target {
        color: #27ae60;
        font-size: 0.9em;
        font-style: italic;
    }
    
    .probs-container {
        display: flex;
        gap: 10px;
        justify-content: center;
        padding: 10px;
    }
    
    .prob-item {
        flex: 1;
        max-width: 150px;
        text-align: center;
    }
"""
    
    
    df_html = df.to_html(
        classes='table', 
        escape=False,
        formatters={
            'Image': lambda x: f'<div class="image-container"><img src="/static/images/{x}" class="result-image"><br>{x}</div>',
            'Correct': lambda x: '✅' if x else '❌',
            'Probabilites': format_probabilities
        },
        columns=['Image', 'Label_Attendu', 'Prediction', 'Correct', 'Probabilites']
    )
    
    date_affichage = now.strftime("%d/%m/%Y à %H:%M")
    # Génération du rapport HTML
    html_content = f"""
<html>
<head>
    <style>
    {css_styles}
    </style>
</head>
<body>
    <div class="container">
        <h1>Rapport d'Analyse d'Images - {date_affichage}</h1>
        
        <div class="dashboard">
            <div class="stat-card">
                <div class="stat-label">Précision Globale</div>
                <div class="big-number">{accuracy:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Images Analysées</div>
                <div class="big-number">{len(df)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Prédictions Correctes</div>
                <div class="big-number">{df['Correct'].sum()}</div>
            </div>
        </div>

        <div class="summary-box">
                <h2>Visualisations</h2>
                <img src="/static/rapports/{report_name}/rapport.png" class="stats-img" alt="Statistiques">
            </div>

        <h2>Résultats Détaillés</h2>
        {df_html}
    </div>
</body>
</html>
"""
    
    html_path = report_name.with_suffix('.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nRapport généré avec succès dans le dossier: {current_report_dir}")
    print(f"Excel: rapport.xlsx")
    print(f"HTML: rapport.html")
    print(f"Statistiques: rapport.png")

if __name__ == '__main__':
    test_folder = "test" 
    create_test_report(test_folder)