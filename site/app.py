import shutil
import sys
from flask import Flask, render_template, send_from_directory, request, redirect, url_for
from werkzeug.utils import secure_filename
import json
import os
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "static" / "rapports"
TEST_DIR = BASE_DIR / "test"
TEST_IMAGES_DIR = BASE_DIR / "images_test"

sys.path.append(str(BASE_DIR.parent))

@app.route('/')
def index():
    """Page d'accueil listant tous les rapports"""
    reports = get_reports_list()
    print(f"Affichage de {len(reports)} rapports")
    return render_template('index.html', reports=reports)

def get_reports_list():
    """Récupère la liste des rapports disponibles"""
    reports = []
    
    # Création du dossier s'il n'existe pas
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Recherche des rapports dans : {REPORTS_DIR}")
    
    if REPORTS_DIR.exists():
        for report_dir in REPORTS_DIR.iterdir():
            if report_dir.is_dir():
                try:
                    print(f"Dossier trouvé : {report_dir}")
                    parts = report_dir.name.split('_')
                    if len(parts) >= 3:
                        report_info = {
                            'id': report_dir.name,
                            'date': parts[1],
                            'time': parts[2],
                            'path': str(report_dir)
                        }
                        reports.append(report_info)
                        print(f"Rapport ajouté : {report_info}")
                    else:
                        print(f"Format de dossier invalide : {report_dir.name}")
                except Exception as e:
                    print(f"Erreur lors du traitement du dossier {report_dir}: {str(e)}")
    
    reports_sorted = sorted(reports, key=lambda x: x['id'], reverse=True)
    print(f"Nombre total de rapports trouvés : {len(reports_sorted)}")
    return reports_sorted

@app.route('/report/<report_id>')
def view_report(report_id):
    """Affiche un rapport spécifique"""
    report_dir = REPORTS_DIR / f"rapport_{report_id}"
    if not report_dir.exists():
        return "Rapport non trouvé", 404
    
    return send_from_directory(report_dir, 'rapport.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)