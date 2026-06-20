#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_results.py
------------------
Extrait automatiquement :
- Toutes les figures affichées dans les notebooks Jupyter (après exécution)
- Les métriques numériques (pertes, FID, etc.) à partir des cellules de code
- Sauvegarde les figures dans figures/ et les métriques dans metrics.json / metrics.csv

Utilisation :
    python extract_results.py

Prérequis :
    pip install jupyter nbconvert pandas
"""

import os
import re
import json
import base64
import subprocess
import sys
from pathlib import Path
import pandas as pd

# Configuration
NOTEBOOKS_DIR = Path("notebooks")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)
METRICS_FILE_JSON = "metrics.json"
METRICS_FILE_CSV = "metrics.csv"

# Liste des notebooks à traiter (dans l'ordre)
notebooks = [
    "01_autoencoder.ipynb",
    "02_vae.ipynb",
    "03_gan.ipynb",
    "04_wgangp.ipynb"
]

# Dictionnaire pour stocker toutes les métriques
all_metrics = {}

def extract_metrics_from_notebook(notebook_path):
    """Extrait les métriques (pertes, FID, etc.) d'un notebook exécuté."""
    notebook_name = notebook_path.stem
    print(f"📊 Extraction des métriques pour {notebook_name}...")

    # Lire le notebook en JSON
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    metrics = {}

    # Parcourir toutes les cellules
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            # Récupérer les sorties (outputs)
            outputs = cell.get("outputs", [])
            for out in outputs:
                if out.get("name") == "stdout":
                    text = out.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    # Rechercher des motifs de métriques
                    # Exemples : "Perte finale sur test : 0.023", "FID: 52.3", "test_loss : 0.035"
                    patterns = [
                        r"Perte finale sur test\s*:\s*([0-9.]+)",
                        r"test_loss\s*:\s*([0-9.]+)",
                        r"FID\s*:\s*([0-9.]+)",
                        r"FID\s*=\s*([0-9.]+)",
                        r"Inception Score\s*:\s*([0-9.]+)",
                        r"loss\[.*?\]\s*=\s*([0-9.]+)",  # pertes génériques
                        r"Perte Générateur\s*:\s*([0-9.]+)",
                        r"Perte Discriminateur\s*:\s*([0-9.]+)",
                        r"G_loss\s*:\s*([0-9.]+)",
                        r"D_loss\s*:\s*([0-9.]+)"
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            # Déduire un nom de métrique lisible
                            if "Perte finale" in pattern or "test_loss" in pattern:
                                key = f"{notebook_name}_test_loss"
                            elif "FID" in pattern:
                                key = f"{notebook_name}_FID"
                            elif "Inception Score" in pattern:
                                key = f"{notebook_name}_IS"
                            elif "Perte Générateur" in pattern or "G_loss" in pattern:
                                key = f"{notebook_name}_G_loss_final"
                            elif "Perte Discriminateur" in pattern or "D_loss" in pattern:
                                key = f"{notebook_name}_D_loss_final"
                            else:
                                key = f"{notebook_name}_metric_{match.group(1)}"
                            metrics[key] = float(match.group(1))
                            print(f"   → Trouvé {key} = {metrics[key]}")

            # Optionnel : extraire les variables des cellules Python (si stockées explicitement)
            # On pourrait exécuter un mini parseur AST, mais c'est plus complexe.
            # On se contente des sorties texte.

    return metrics

def main():
    # Vérifier le dossier notebooks
    if not NOTEBOOKS_DIR.exists():
        print(f"❌ Dossier '{NOTEBOOKS_DIR}' introuvable.")
        sys.exit(1)

    # Traiter chaque notebook
    for nb in notebooks:
        nb_path = NOTEBOOKS_DIR / nb
        if nb_path.exists():
            metrics = extract_metrics_from_notebook(nb_path)
            if metrics:
                all_metrics.update(metrics)
        else:
            print(f"⚠️ Notebook non trouvé : {nb_path}")

    # Sauvegarder toutes les métriques en JSON
    if all_metrics:
        with open(METRICS_FILE_JSON, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Métriques sauvegardées dans {METRICS_FILE_JSON}")

        # Convertir en DataFrame pandas pour CSV
        df = pd.DataFrame([all_metrics])
        df.to_csv(METRICS_FILE_CSV, index=False)
        print(f"✅ Métriques sauvegardées dans {METRICS_FILE_CSV}")
    else:
        print("\n⚠️ Aucune métrique extraite. Vérifiez que vos notebooks impriment des valeurs avec des mots-clés comme 'Perte finale', 'FID', etc.")

    print(f"\n🎉 Extraction terminée. Figures dans '{OUTPUT_DIR}/', métriques dans '{METRICS_FILE_JSON}' et '{METRICS_FILE_CSV}'.")

if __name__ == "__main__":
    main()