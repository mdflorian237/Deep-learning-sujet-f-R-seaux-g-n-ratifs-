#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_presentation.py
------------------------
Génère automatiquement un diaporama PowerPoint (PPTX) à partir des figures
et des métriques extraites, via Pandoc.

Utilisation :
    python generate_presentation.py

Prérequis :
    - Pandoc installé
    - Les figures dans le dossier 'figures/'
    - Le fichier 'metrics.json' (généré par extract_results.py)
"""

import os
import json
import subprocess
from pathlib import Path

# Configuration
FIGURES_DIR = Path("figures")
METRICS_FILE = Path("metrics.json")
OUTPUT_MD = Path("presentation.md")
OUTPUT_PPTX = Path("diaporama.pptx")

# Ordre des modèles
NOTEBOOKS_ORDER = ["01_autoencoder", "02_vae", "03_gan", "04_wgangp"]
MODEL_NAMES = {
    "01_autoencoder": "Autoencodeur simple",
    "02_vae": "Autoencodeur variationnel (VAE)",
    "03_gan": "GAN classique",
    "04_wgangp": "WGAN-GP (plus stable)"
}

def load_metrics():
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_figure_paths(model_prefix):
    """Retourne les chemins des figures pour un modèle donné"""
    return sorted(FIGURES_DIR.glob(f"{model_prefix}_fig*.png"))

def generate_presentation_md():
    """Génère le fichier Markdown structuré pour le diaporama"""
    metrics = load_metrics()
    content = []

    # --- Slide 1 : Titre ---
    content.append("""% Diaporama – Projet Deep Learning
% Sujet F : Autoencodeurs, VAE et GAN
% Groupe [vos noms] – [Date]

# Plan de la présentation

1. Introduction et contexte
2. Autoencodeur simple
3. Autoencodeur variationnel (VAE)
4. GAN classique
5. WGAN-GP
6. Comparaison et métriques
7. Conclusion & perspectives

---
""")

    # --- Slide 2 : Introduction ---
    content.append("""# Introduction

**Objectif** : Générer des images de chiffres manuscrits (MNIST) avec différentes architectures Deep Learning.

**Modèles étudiés** :
- Autoencodeur (reconstruction)
- VAE (génération probabiliste)
- GAN (génération adversarial)
- WGAN-GP (amélioration stable)

**Métriques d'évaluation** :
- Perte de reconstruction (MSE) pour AE/VAE
- FID (Fréchet Inception Distance) pour les GANs
- Qualité visuelle

---
""")

    # --- Slides pour chaque modèle ---
    for model_prefix in NOTEBOOKS_ORDER:
        model_name = MODEL_NAMES[model_prefix]
        fig_paths = get_figure_paths(model_prefix)

        # Titre du slide
        content.append(f"# {model_name}\n")

        # Description rapide
        if model_prefix == "01_autoencoder":
            content.append("""
- **Principe** : Encodeur → Espace latent → Décodeur
- **Objectif** : Minimiser l'erreur de reconstruction (MSE)
- **Avantage** : Simple et rapide à entraîner
- **Limite** : Ne génère pas de nouvelles images (seulement reconstruction)
""")
        elif model_prefix == "02_vae":
            content.append("""
- **Principe** : Encodeur probabiliste (moyenne + variance) + reparametrization trick
- **Objectif** : Maximiser l'ELBO (reconstruction + régularisation KL)
- **Avantage** : Génération contrôlable par échantillonnage
- **Limite** : Images parfois floues
""")
        elif model_prefix == "03_gan":
            content.append("""
- **Principe** : Compétition Générateur vs Discriminateur
- **Objectif** : Le générateur trompe le discriminateur
- **Avantage** : Images nettes
- **Limite** : Instabilité, mode collapse
""")
        elif model_prefix == "04_wgangp":
            content.append("""
- **Principe** : Distance de Wasserstein + pénalité de gradient
- **Objectif** : Stabilité et convergence assurée
- **Avantage** : Qualité élevée, diversité
- **Limite** : Coût de calcul légèrement supérieur
""")

        # Ajouter les figures
        if fig_paths:
            content.append("\n**Résultats** :\n")
            # Courbe de perte (généralement fig01)
            if len(fig_paths) >= 1:
                content.append(f"\n![Courbe d'apprentissage]({fig_paths[0]})\n")
            # Reconstructions ou générations (fig02 ou plus)
            if len(fig_paths) >= 2:
                content.append(f"\n![Exemples de sorties]({fig_paths[1]})\n")
            # Si plus de figures, on peut en ajouter une troisième (ex: espace latent VAE)
            if len(fig_paths) >= 3 and "vae" in model_prefix:
                content.append(f"\n![Espace latent (VAE)]({fig_paths[2]})\n")
        else:
            content.append("\n⚠️ Aucune figure disponible.\n")

        content.append("\n---\n")

    # --- Slide : Comparaison des modèles ---
    content.append("""
# Comparaison des performances

| Modèle | Perte de test (reconstruction) | FID (↓) | Stabilité | Qualité |
|--------|-------------------------------|---------|-----------|---------|
""")

    for model_prefix in NOTEBOOKS_ORDER:
        model_name = MODEL_NAMES[model_prefix]
        # Récupération sécurisée des métriques
        if model_prefix in ["01_autoencoder", "02_vae"]:
            loss = metrics.get(f"{model_prefix}_test_loss", "N/A")
            if isinstance(loss, float):
                loss = f"{loss:.6f}"
            fid = "N/A"
            stable = "✅ Élevée"
            quality = "Reconstruction fidèle"
        else:
            loss = "N/A"
            fid = metrics.get(f"{model_prefix}_FID", "N/A")
            if isinstance(fid, float):
                fid = f"{fid:.2f}"
            if "wgangp" in model_prefix:
                stable = "✅ Très stable"
                quality = "⭐ Haute qualité"
            else:
                stable = "⚠️ Instable"
                quality = "Bonne (mode collapse possible)"

        content.append(f"| {model_name} | {loss} | {fid} | {stable} | {quality} |\n")

    content.append("""
---
""")

    # --- Slide : Conclusion ---
    content.append("""
# Conclusion

**Principaux enseignements** :
- L'autoencodeur simple est parfait pour la reconstruction, mais limité pour la génération.
- Le VAE permet une génération contrôlée et variée.
- Le GAN classique offre des images nettes mais souffre d'instabilité.
- Le **WGAN-GP** est le meilleur compromis : stable, diversifié, haute qualité visuelle.

**Perspectives** :
- Utiliser des datasets plus complexes (CIFAR-10, visages).
- Explorer d'autres architectures (StyleGAN, Diffusion models).
- Déployer le modèle dans une application Streamlit pour la génération interactive.

---
**Merci pour votre attention !**

Questions ?
""")

    # Écrire le fichier Markdown
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
    print(f"✅ Fichier de présentation généré : {OUTPUT_MD}")

def generate_pptx():
    """Convertit le Markdown en PowerPoint via Pandoc"""
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
    except:
        print("❌ Pandoc non trouvé.")
        return

    cmd = [
        "pandoc",
        str(OUTPUT_MD),
        "-o", str(OUTPUT_PPTX),
        "--to", "pptx"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Diaporama PowerPoint généré : {OUTPUT_PPTX}")
        print("\n📌 Conseil : Ouvrez le fichier dans PowerPoint et appliquez un thème (onglet 'Design') pour un rendu professionnel.")
        print("📌 Pensez à passer en mode 16:9 (Design → Taille du diaporama → Widescreen (16:9)).")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de la génération du PPTX : {e}")

def main():
    generate_presentation_md()
    generate_pptx()

if __name__ == "__main__":
    main()