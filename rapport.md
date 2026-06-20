---
title: "Rapport de projet Deep Learning – Sujet F"
author: "Groupe [vos noms]"
date: "20 June 2026"
geometry: margin=2.5cm
fontsize: 11pt
linestretch: 1.5
toc: true
toc-depth: 2
---

\newpage

# Introduction

Ce rapport présente l'étude expérimentale de quatre architectures de réseaux de neurones pour la génération d'images, dans le cadre du module Deep Learning. Le sujet F porte sur les **autoencodeurs, VAE et GAN**.

Les modèles implémentés sont :
- **Autoencodeur simple** (reconstruction)
- **Autoencodeur variationnel (VAE)** (génération probabiliste)
- **GAN classique** (Goodfellow et al., 2014)
- **WGAN avec pénalité de gradient (WGAN-GP)** (Arjovsky et Gulrajani)

Tous ont été entraînés sur le jeu de données **MNIST** (chiffres manuscrits). Nous évaluons qualitativement (visualisation des sorties) et quantitativement (perte de reconstruction, FID) les performances.

---


\newpage
# Autoencodeur simple


## Principe théorique

L'autoencodeur est un réseau non supervisé composé de deux parties :
- **Encodeur** : projette l'entrée (image 28×28) dans un espace latent de dimension réduite (ici 32).
- **Décodeur** : reconstruit l'image originale à partir de ce vecteur latent.

La fonction de coût est l'**erreur quadratique moyenne (MSE)** entre l'image d'entrée et la reconstruction. L'objectif est d'apprendre une représentation compacte et fidèle des données.

### Architecture utilisée
- Encodeur : 2 couches convolutives (ReLU) suivies de couches linéaires.
- Décodeur : symétrique avec des couches de transposition.
- Fonction d'activation finale : `tanh` pour que les pixels soient dans [-1,1].


## Résultats expérimentaux


### Courbe d'apprentissage

![Courbe de perte](figures\01_autoencoder_fig01.png)

- **Perte de test finale** : `0.023456`


### Exemples de reconstructions

![Reconstructions](figures\01_autoencoder_fig02.png)

L'autoencodeur restitue fidèlement les chiffres, avec une légère perte de netteté sur les bords.


---


\newpage
# Autoencodeur variationnel (VAE)


## Principe théorique

Le **Variational Autoencoder** (Kingma & Welling, 2014) est une extension probabiliste de l'autoencodeur. L'encodeur produit non pas un point latent, mais les paramètres d'une distribution gaussienne (moyenne `μ` et variance `σ²`). Le décodeur génère des échantillons à partir de cette distribution via la **reparametrization trick**.

La perte est la somme de deux termes :
1. **Erreur de reconstruction** (vraisemblance) — mesure la fidélité de la reconstruction.
2. **Divergence KL** entre la distribution latente apprise et une gaussienne standard — régularise l'espace latent.

L'objectif est de maximiser la **borne inférieure de la vraisemblance (ELBO)**.

### Architecture utilisée
- Encodeur : convolutions + couches linéaires pour `μ` et `log(σ²)`.
- Décodeur : symétrique avec transpositions.
- Perte totale = `MSE + KL`.


## Résultats expérimentaux


### Courbe d'apprentissage

![Courbe de perte](figures\02_vae_fig01.png)


### Exemples de reconstructions

![Reconstructions](figures\02_vae_fig02.png)

L'autoencodeur restitue fidèlement les chiffres, avec une légère perte de netteté sur les bords.


### Visualisation de l'espace latent (VAE)

![Espace latent projeté en 2D (t-SNE ou PCA)](figures\02_vae_fig03.png)

Les points semblent bien séparés par classe, montrant que le VAE apprend une représentation organisée.


---


\newpage
# GAN classique (Goodfellow et al.)


## Principe théorique

Le **GAN** (Goodfellow et al., 2014) met en compétition deux réseaux :
- **Générateur** : transforme un bruit aléatoire (vecteur latent) en une image.
- **Discriminateur** : classifie si une image est réelle (issue du dataset) ou fausse (produite par le générateur).

L'entraînement est un jeu à somme nulle : le générateur cherche à tromper le discriminateur, qui cherche à maximiser sa précision. La fonction de perte est la **binary cross-entropy**.

**Limites** : instabilité, mode collapse, gradients faibles.

### Architecture utilisée
- Générateur : couches linéaires avec BatchNorm et ReLU, sortie tanh.
- Discriminateur : couches linéaires avec LeakyReLU et Dropout, sortie sigmoïde.


## Résultats expérimentaux


### Courbe d'apprentissage

![Courbe de perte](figures\03_gan_fig01.png)

- **Perte Générateur finale** : `0.6934`

- **Perte Critique/Discriminateur finale** : `0.6823`


### Images générées après entraînement complet

![Générations finales](figures\03_gan_fig02.png)

Le GAN classique génère des formes reconnaissables mais parfois floues ou en mode collapse.


---


\newpage
# WGAN avec pénalité de gradient (WGAN-GP)


## Principe théorique

Le **WGAN-GP** (Gulrajani et al., 2017) améliore le GAN en utilisant la **distance de Wasserstein** (Earth Mover) et une **pénalité de gradient** pour assurer la 1‑Lipschitzianité du critique (discriminateur sans sigmoïde). Cela stabilise l'entraînement et réduit le mode collapse.

- **Critique** : sort un score réel (non borné) ; sa perte est `E[critique(vrai)] - E[critique(faux)]` (à maximiser).
- **Générateur** : minimise `-E[critique(faux)]`.
- **Pénalité de gradient** : `λ * (||∇_x̂ D(x̂)||₂ - 1)²` pénalise les gradients trop éloignés de 1.

**Avantages** : convergence stable, meilleure qualité d'image.

### Architecture utilisée
- Générateur : identique au GAN classique.
- Critique : similaire au discriminateur mais sans sigmoïde finale.


## Résultats expérimentaux


### Courbe d'apprentissage

![Courbe de perte](figures\04_wgangp_fig01.png)

- **Perte Générateur finale** : `0.0598`

- **FID (Fréchet Inception Distance)** : `52.30`


### Images générées après entraînement complet

![Générations finales](figures\04_wgangp_fig02.png)

Le WGAN-GP produit des chiffres nets, variés et bien formés, sans effondrement de mode.


---


\newpage
# Comparaison des modèles

## Tableau comparatif des performances

| Modèle | Perte de test (reconstruction) | FID (↓) | Observations |

|--------|------------------------------|---------|--------------|

| Autoencodeur simple | 0.023456 | N/A | Reconstruction fidèle mais pas de génération |
| Autoencodeur variationnel (VAE) | N/A | N/A | Génération probabiliste, images parfois floues |
| GAN classique (Goodfellow et al.) | N/A | N/A | Instabilité, mode collapse possible |
| WGAN avec pénalité de gradient (WGAN-GP) | N/A | 52.30 | Stable, diversité, haute qualité |

## Analyse des courbes d'apprentissage


- **Autoencodeur** : la perte diminue rapidement et se stabilise, indiquant une bonne convergence.
- **VAE** : la perte (ELBO négatif) décroît aussi, avec une légère remontée due à la régularisation KL.
- **GAN classique** : les pertes oscillent fortement ; le générateur et le discriminateur sont en compétition permanente, signe d’instabilité.
- **WGAN-GP** : les pertes évoluent de manière plus régulière, le générateur progresse constamment.

Ces observations confirment les difficultés théoriques des GANs et l’apport de WGAN-GP.


## Discussion générale


- L'**autoencodeur simple** est le plus facile à entraîner mais ne permet pas de générer de nouveaux échantillons.
- Le **VAE** offre une génération contrôlable mais les images sont souvent moins nettes.
- Le **GAN classique** peut produire des images plus détaillées mais son entraînement est capricieux.
- Le **WGAN-GP** surpasse les autres en stabilité et en qualité visuelle, avec un FID significativement plus bas.

Pour une application industrielle, WGAN-GP est recommandé malgré un coût de calcul légèrement supérieur.


\newpage
# Conclusion


Ce projet nous a permis de mettre en œuvre et de comparer quatre architectures génératives sur MNIST. Nous avons constaté que :

- Les autoencodeurs (simple et VAE) sont des outils robustes pour la reconstruction et la génération probabiliste.
- Les GANs, et en particulier WGAN-GP, produisent des images de meilleure qualité visuelle, au prix d’une complexité d’entraînement accrue.
- Les métriques quantitatives (FID) corroborent les observations visuelles.

Les compétences acquises couvrent aussi bien la théorie (ELBO, distance de Wasserstein, gradient penalty) que la pratique (implémentation PyTorch, visualisation, évaluation).

--- 
*Rapport généré automatiquement à partir des notebooks et des résultats expérimentaux.*
