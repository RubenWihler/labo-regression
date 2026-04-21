#!/bin/bash

# Nom du dossier de l'environnement virtuel
VENV_DIR=".venv"

echo "===================================================="
echo "🚀 Préparation du Laboratoire de Mathématiques..."
echo "===================================================="

# 1. Vérifier si Python 3 est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Erreur : Python 3 n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# 2. Création de l'environnement virtuel (s'il n'existe pas)
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Création de l'environnement virtuel ($VENV_DIR)..."
    python3 -m venv $VENV_DIR
else
    echo "✅ Environnement virtuel déjà existant."
fi

# 3. Activation de l'environnement virtuel
echo "🔄 Activation de l'environnement virtuel..."
source $VENV_DIR/bin/activate

# 4. Mise à jour de pip pour éviter les warnings
echo "⬇️  Mise à jour de pip..."
pip install --upgrade pip --quiet

# 5. Installation des dépendances
echo "📥 Installation des dépendances (Streamlit, Numpy, Plotly)..."
pip install streamlit numpy plotly --quiet

echo "===================================================="
echo "🎉 Lancement de l'application..."
echo "===================================================="

# 6. Lancer le dashboard Streamlit
streamlit run src/labo.py
