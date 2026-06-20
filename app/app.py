# app/app.py
# ------------------------------------------------------------------
# Application Streamlit – Projet Deep Learning Sujet F
# Auteurs : MAFFOUO DONGMO FLORIAN & SONA NAUDEM SINTHIA
# Description : Visualisation des résultats des modèles génératifs
#              (Autoencodeur, VAE, GAN, WGAN-GP) sur MNIST.
# ------------------------------------------------------------------

# === 1. IMPORTS ===================================================
import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import des fonctions et classes depuis utils.py
from utils import load_model, to_image, Autoencoder, VAE, Generator, transforms

# === 2. CONFIGURATION DE LA PAGE ==================================
st.set_page_config(
    page_title="Projet Deep Learning • Sujet F : Réseaux génératifs",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === 3. STYLES CSS PERSONNALISÉS (thème clair professionnel) ======
st.markdown("""
<style>
    /* --- Fond général --- */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        box-shadow: 2px 0 8px rgba(0,0,0,0.04);
        padding: 1rem 0.5rem;
    }
    [data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }

    /* --- Titres --- */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 600;
    }

    /* --- Texte courant --- */
    p, li, .stMarkdown, .stText {
        color: #1e293b !important;
    }

    /* --- Cartes images --- */
    .image-card {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        transition: 0.2s;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .image-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.10);
    }

    /* --- Boutons --- */
    .stButton button {
        background: #3b82f6;
        color: white !important;
        font-weight: 600;
        border-radius: 40px;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background: #2563eb;
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }

    /* --- Onglets --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.2rem;
        background: #f1f5f9;
        padding: 0.3rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        color: #334155;
        transition: 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        color: #0f172a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* --- Pied de page --- */
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid #e2e8f0;
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# === 4. SIDEBAR COMMUNE ===========================================
with st.sidebar:
    # Logo ou titre visuel
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/main/examples/data/logo.png", width=70)
    st.markdown("## 🧠 Projet Sujet F : Réseaux génératifs")
    st.markdown("**Autoencodeurs, VAE, GAN, WGAN-GP**")
    st.markdown("---")

    # Métriques globales (lues depuis metrics.json)
    st.markdown("### 📊 Métriques globales")
    metrics_path = Path("metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        for k, v in metrics.items():
            # Nettoyage du label pour l'affichage
            label = k.replace("01_", "").replace("02_", "").replace("03_", "").replace("04_", "")
            label = label.replace("_test_loss", " (test loss)").replace("_G_loss_final", " (G loss)")
            label = label.replace("_D_loss_final", " (D loss)").replace("_FID", " FID")
            st.metric(label, f"{v:.4f}" if isinstance(v, float) else v)
    else:
        st.caption("Fichier metrics.json introuvable")

    st.markdown("---")
    st.caption("Auteurs : MAFFOUO DONGMO FLORIAN & SONA NAUDEM SINTHIA")

# === 5. FONCTION DE CHARGEMENT DES MODÈLES (avec cache) ===========
@st.cache_resource
def get_model(model_key):
    """Charge le modèle demandé (autoencoder, vae, gan, wgangp) avec mise en cache."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_key, device)
    return model, device

# === 6. ONGLETS PRINCIPAUX =========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Accueil",
    "📘 Autoencodeur",
    "📗 VAE",
    "📙 GAN",
    "📕 WGAN-GP",
    "👤 À propos"
])

# --- 6.1 ONGLET ACCUEIL --------------------------------------------
with tab1:
    st.markdown("""
                # Bienvenue sur le Projet Deep Learning : Réseaux génératifs 
                """
                )
    st.markdown("""
    Cette application présente les résultats de l'étude expérimentale de quatre architectures génératives
    sur le dataset **MNIST** (chiffres manuscrits).
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Objectifs
        - Implémenter un **Autoencodeur**, un **VAE**, un **GAN** et un **WGAN-GP**.
        - Comparer leurs performances en reconstruction et génération.
        - Visualiser les courbes d'apprentissage et les échantillons générés.
        """)
    with col2:
        st.markdown("""
        ### 📌 Modèles étudiés
        - **Autoencodeur** : reconstruction fidèle, pas de génération.
        - **VAE** : génération probabiliste, espace latent régularisé.
        - **GAN** : compétition Générateur/Discriminateur, images nettes.
        - **WGAN-GP** : distance de Wasserstein + pénalité de gradient, stable.
        """)

    st.markdown("---")
    st.markdown("### 🧭 Utilisation")
    st.markdown("""
    Utilisez les onglets ci-dessus pour explorer chaque modèle :
    - Consultez les **courbes d'apprentissage** et les **exemples de sorties**.
    - Pour les modèles génératifs (VAE, GAN, WGAN-GP), testez la **génération interactive**.
    - Pour l'autoencodeur, téléchargez une image MNIST pour **reconstruction**.
    """)

# --- 6.2 ONGLET AUTOENCODEUR ---------------------------------------
with tab2:
    st.markdown("# 📘 Autoencodeur ")
    st.markdown("""
    L'autoencodeur apprend une représentation compacte des données via un goulot d'étranglement (espace latent).
    Il est entraîné à minimiser l'erreur de reconstruction (MSE).
    """)

    # Affichage des figures disponibles
    fig_paths = sorted(Path("figures").glob("01_autoencoder_fig*.png"))
    if fig_paths:
        cols = st.columns(2)
        captions = {
            "01_autoencoder_fig01": "Courbe d'apprentissage",
            "01_autoencoder_fig02": "Exemples de reconstructions",
        }
        for idx, fig_path in enumerate(fig_paths):
            with cols[idx % 2]:
                caption = captions.get(fig_path.stem, fig_path.stem)
                st.image(str(fig_path), caption=caption, use_container_width=True)
    else:
        st.warning("Aucune figure trouvée pour l'autoencodeur.")

    st.markdown("---")
    st.subheader("🔄 Tester la reconstruction")
    uploaded_file = st.file_uploader("Téléchargez une image MNIST (28x28)", type=["png", "jpg", "jpeg"], key="ae_upload")
    if uploaded_file is not None:
        model, device = get_model("autoencoder")
        if model:
            img = Image.open(uploaded_file).convert("L").resize((28,28))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                recon, _ = model(tensor)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                st.image(img, caption="Originale", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                st.image(to_image(recon), caption="Reconstruite", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

# --- 6.3 ONGLET VAE -------------------------------------------------
with tab3:
    st.markdown("# 📗 VAE (Variational Autoencoder)")
    st.markdown("""
    Le VAE étend l'autoencodeur en rendant l'espace latent probabiliste. Il maximise l'ELBO (reconstruction + régularisation KL).
    Il permet de **générer de nouvelles images** en échantillonnant l'espace latent.
    """)

    fig_paths = sorted(Path("figures").glob("02_vae_fig*.png"))
    if fig_paths:
        cols = st.columns(2)
        captions = {
            "02_vae_fig01": "Courbe d'apprentissage",
            "02_vae_fig02": "Exemples de reconstructions",
            "02_vae_fig03": "Visualisation de l'espace latent",
            "02_vae_fig04": "Générations aléatoires",
        }
        for idx, fig_path in enumerate(fig_paths):
            with cols[idx % 2]:
                caption = captions.get(fig_path.stem, fig_path.stem)
                st.image(str(fig_path), caption=caption, use_container_width=True)
    else:
        st.warning("Aucune figure trouvée pour le VAE.")

    st.markdown("---")
    st.subheader("🎲 Génération interactive")
    model, device = get_model("vae")
    if model:
        col1, col2 = st.columns([1, 3])
        with col1:
            num_samples = st.slider("Nombre d'images", 1, 16, 4, step=1, key="vae_count")
            temperature = st.slider("Température (écart-type du bruit)", 0.5, 2.0, 1.0, 0.1, key="vae_temp")
            generate = st.button("🎲 Générer", key="gen_vae")
        if generate:
            with st.spinner("Génération en cours..."):
                z = torch.randn(num_samples, 20, device=device) * temperature
                with torch.no_grad():
                    gen_imgs = model.decode(z)
            cols = st.columns(min(4, num_samples))
            for i, col in enumerate(cols):
                if i < num_samples:
                    with col:
                        st.markdown('<div class="image-card">', unsafe_allow_html=True)
                        st.image(to_image(gen_imgs[i]), use_container_width=True, caption=f"Échantillon {i+1}")
                        st.markdown('</div>', unsafe_allow_html=True)
            # Téléchargement en mosaïque
            if num_samples > 0:
                images = [to_image(gen_imgs[i]) for i in range(num_samples)]
                w, h = images[0].size
                grid_cols = min(4, num_samples)
                grid_rows = (num_samples + grid_cols - 1) // grid_cols
                mosaic = Image.new('L', (grid_cols * w, grid_rows * h))
                for idx, img in enumerate(images):
                    x = (idx % grid_cols) * w
                    y = (idx // grid_cols) * h
                    mosaic.paste(img, (x, y))
                st.download_button(
                    label="📥 Télécharger la mosaïque",
                    data=mosaic.tobytes(),
                    file_name="vae_mosaique.png",
                    mime="image/png"
                )

# --- 6.4 ONGLET GAN -------------------------------------------------
with tab4:
    st.markdown("# 📙 GAN classique (Goodfellow et al.)")
    st.markdown("""
    Le GAN met en compétition un générateur et un discriminateur. Il produit des images nettes mais
    souffre d'instabilité (effondrement de mode, gradients faibles).
    """)

    fig_paths = sorted(Path("figures").glob("03_gan_fig*.png"))
    if fig_paths:
        # Courbe de perte
        if len(fig_paths) >= 1:
            st.image(str(fig_paths[0]), caption="Courbes d'apprentissage", use_container_width=True)
        # Générations finales
        if len(fig_paths) >= 2:
            st.image(str(fig_paths[1]), caption="Générations finales", use_container_width=True)
        # Évolution des générations
        if len(fig_paths) >= 3:
            st.markdown("---")
            st.subheader("📈 Évolution des générations au fil des époques")
            epoch_figs = fig_paths[2:]
            if epoch_figs:
                epoch_idx = st.slider("Choisissez une époque", 0, len(epoch_figs)-1, 0, key="gan_epoch")
                st.image(str(epoch_figs[epoch_idx]), caption=f"Époque {epoch_idx+1} (sur {len(epoch_figs)} époques)", use_container_width=True)
    else:
        st.warning("Aucune figure trouvée pour le GAN.")

    st.markdown("---")
    st.subheader("🎲 Génération interactive")
    model, device = get_model("gan")
    if model:
        col1, col2 = st.columns([1, 3])
        with col1:
            num_samples = st.slider("Nombre d'images", 1, 16, 4, step=1, key="gan_count")
            temperature = st.slider("Température", 0.5, 2.0, 1.0, 0.1, key="gan_temp")
            generate = st.button("🎲 Générer", key="gen_gan")
        if generate:
            with st.spinner("Génération en cours..."):
                z = torch.randn(num_samples, 100, device=device) * temperature
                with torch.no_grad():
                    gen_imgs = model(z)
            cols = st.columns(min(4, num_samples))
            for i, col in enumerate(cols):
                if i < num_samples:
                    with col:
                        st.markdown('<div class="image-card">', unsafe_allow_html=True)
                        st.image(to_image(gen_imgs[i]), use_container_width=True, caption=f"Faux {i+1}")
                        st.markdown('</div>', unsafe_allow_html=True)
            # Téléchargement
            if num_samples > 0:
                images = [to_image(gen_imgs[i]) for i in range(num_samples)]
                w, h = images[0].size
                grid_cols = min(4, num_samples)
                grid_rows = (num_samples + grid_cols - 1) // grid_cols
                mosaic = Image.new('L', (grid_cols * w, grid_rows * h))
                for idx, img in enumerate(images):
                    x = (idx % grid_cols) * w
                    y = (idx // grid_cols) * h
                    mosaic.paste(img, (x, y))
                st.download_button(
                    label="📥 Télécharger la mosaïque",
                    data=mosaic.tobytes(),
                    file_name="gan_mosaique.png",
                    mime="image/png"
                )

# --- 6.5 ONGLET WGAN-GP --------------------------------------------
with tab5:
    st.markdown("# 📕 WGAN avec pénalité de gradient (WGAN-GP)")
    st.markdown("""
    Le WGAN-GP améliore le GAN en utilisant la distance de Wasserstein et une pénalité de gradient.
    Il est **plus stable**, **moins sujet à l'effondrement de mode** et produit des images de meilleure qualité.
    """)

    fig_paths = sorted(Path("figures").glob("04_wgangp_fig*.png"))
    if fig_paths:
        if len(fig_paths) >= 1:
            st.image(str(fig_paths[0]), caption="Courbes d'apprentissage", use_container_width=True)
        if len(fig_paths) >= 2:
            st.image(str(fig_paths[1]), caption="Générations finales", use_container_width=True)
        if len(fig_paths) >= 3:
            st.markdown("---")
            st.subheader("📈 Évolution des générations au fil des époques")
            epoch_figs = fig_paths[2:]
            if epoch_figs:
                epoch_idx = st.slider("Choisissez une époque", 0, len(epoch_figs)-1, 0, key="wgan_epoch")
                st.image(str(epoch_figs[epoch_idx]), caption=f"Époque {epoch_idx+1} (sur {len(epoch_figs)} époques)", use_container_width=True)
    else:
        st.warning("Aucune figure trouvée pour WGAN-GP.")

    st.markdown("---")
    st.subheader("🎲 Génération interactive")
    model, device = get_model("wgangp")
    if model:
        col1, col2 = st.columns([1, 3])
        with col1:
            num_samples = st.slider("Nombre d'images", 1, 16, 4, step=1, key="wgan_count")
            temperature = st.slider("Température", 0.5, 2.0, 1.0, 0.1, key="wgan_temp")
            generate = st.button("🎲 Générer", key="gen_wgan")
        if generate:
            with st.spinner("Génération en cours..."):
                z = torch.randn(num_samples, 100, device=device) * temperature
                with torch.no_grad():
                    gen_imgs = model(z)
            cols = st.columns(min(4, num_samples))
            for i, col in enumerate(cols):
                if i < num_samples:
                    with col:
                        st.markdown('<div class="image-card">', unsafe_allow_html=True)
                        st.image(to_image(gen_imgs[i]), use_container_width=True, caption=f"Faux {i+1}")
                        st.markdown('</div>', unsafe_allow_html=True)
            if num_samples > 0:
                images = [to_image(gen_imgs[i]) for i in range(num_samples)]
                w, h = images[0].size
                grid_cols = min(4, num_samples)
                grid_rows = (num_samples + grid_cols - 1) // grid_cols
                mosaic = Image.new('L', (grid_cols * w, grid_rows * h))
                for idx, img in enumerate(images):
                    x = (idx % grid_cols) * w
                    y = (idx // grid_cols) * h
                    mosaic.paste(img, (x, y))
                st.download_button(
                    label="📥 Télécharger la mosaïque",
                    data=mosaic.tobytes(),
                    file_name="wgangp_mosaique.png",
                    mime="image/png"
                )

# --- 6.6 ONGLET À PROPOS --------------------------------------------
with tab6:
    st.markdown("# 👤 À propos de ce projet")
    st.markdown("""
    Ce projet a été réalisé dans le cadre du cours **Deep Learning** à l'**École Normale Supérieure (ENS) de Yaoundé**.
    Il porte sur l'étude et l'implémentation d'architectures génératives pour la génération d'images numériques (dataset MNIST).
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 👨‍🎓 Auteurs
        - **MAFFOUO DONGMO FLORIAN**
        - **SONA NAUDEM SINTHIA**

        **4eme annee Informatique Fondamentale**          
        **ENS Yaoundé**  
        **Année académique 2025-2026**
        """)
    with col2:
        st.markdown("""
        ### 📚 Contexte académique
        - **Enseignant** : **Stéphane C. K. TEKOUABOU (PhD & Ing.)**
        - **Module** : Deep Learning
        - **Sujet** : Réseaux génératifs : Autoencodeurs, VAE et GAN (Sujet F)
        - **Livrables** : Rapport, diaporama, code, application Streamlit

        ### 🔗 Ressources utilisées
        - Dataset : MNIST (28x28, 10 classes)
        - Bibliothèques : PyTorch, Streamlit, Matplotlib, NumPy
     
        """)

    st.markdown("---")
    

# === 7. FOOTER =====================================================
st.markdown("""
<div class="footer">
    Réseaux génératifs :  &nbsp;·&nbsp; Autoencodeurs, VAE, GAN, WGAN-GP &nbsp;·&nbsp; MNIST
</div>
""", unsafe_allow_html=True)