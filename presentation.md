% Diaporama – Projet Deep Learning
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

# Introduction

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

# Autoencodeur simple


- **Principe** : Encodeur → Espace latent → Décodeur
- **Objectif** : Minimiser l'erreur de reconstruction (MSE)
- **Avantage** : Simple et rapide à entraîner
- **Limite** : Ne génère pas de nouvelles images (seulement reconstruction)


**Résultats** :


![Courbe d'apprentissage](figures\01_autoencoder_fig01.png)


![Exemples de sorties](figures\01_autoencoder_fig02.png)


---

# Autoencodeur variationnel (VAE)


- **Principe** : Encodeur probabiliste (moyenne + variance) + reparametrization trick
- **Objectif** : Maximiser l'ELBO (reconstruction + régularisation KL)
- **Avantage** : Génération contrôlable par échantillonnage
- **Limite** : Images parfois floues


**Résultats** :


![Courbe d'apprentissage](figures\02_vae_fig01.png)


![Exemples de sorties](figures\02_vae_fig02.png)


![Espace latent (VAE)](figures\02_vae_fig03.png)


---

# GAN classique


- **Principe** : Compétition Générateur vs Discriminateur
- **Objectif** : Le générateur trompe le discriminateur
- **Avantage** : Images nettes
- **Limite** : Instabilité, mode collapse


**Résultats** :


![Courbe d'apprentissage](figures\03_gan_fig01.png)


![Exemples de sorties](figures\03_gan_fig02.png)


---

# WGAN-GP (plus stable)


- **Principe** : Distance de Wasserstein + pénalité de gradient
- **Objectif** : Stabilité et convergence assurée
- **Avantage** : Qualité élevée, diversité
- **Limite** : Coût de calcul légèrement supérieur


**Résultats** :


![Courbe d'apprentissage](figures\04_wgangp_fig01.png)


![Exemples de sorties](figures\04_wgangp_fig02.png)


---


# Comparaison des performances

| Modèle | Perte de test (reconstruction) | FID (↓) | Stabilité | Qualité |
|--------|-------------------------------|---------|-----------|---------|

| Autoencodeur simple | 0.023456 | N/A | ✅ Élevée | Reconstruction fidèle |

| Autoencodeur variationnel (VAE) | N/A | N/A | ✅ Élevée | Reconstruction fidèle |

| GAN classique | N/A | N/A | ⚠️ Instable | Bonne (mode collapse possible) |

| WGAN-GP (plus stable) | N/A | 52.30 | ✅ Très stable | ⭐ Haute qualité |


---


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
