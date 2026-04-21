# 🧪 Labo Régression & Descente de Gradient

Bienvenue dans le **Laboratoire Interactif d'Optimisation**. Ce projet éducatif permet de visualiser en temps réel comment les algorithmes de Machine Learning (Descente de gradient) apprennent et naviguent dans des espaces mathématiques complexes.

La particularité de ce projet ? **Le calcul des dérivées est exact.** Il n'utilise aucune approximation numérique, mais repose sur une implémentation maison des **Nombres Duaux** (Différenciation Automatique).

## ✨ Fonctionnalités

L'application propose 3 modes d'exploration principaux :

- 📈 **Régression Linéaire :** Ajustement d'une droite avec minimisation de l'Erreur Quadratique Moyenne (MSE).
- 🔮 **Régression Logistique :** Classification binaire avec maximisation de la Log-Vraisemblance (BCE Cost).
- 🏔️ **Laboratoire de Descente (1D & 2D) :** Observez la bille dévaler des fonctions mathématiques célèbres (Vallée de Rosenbrock, Fonction de Beale, Double Puits, etc.).

### 🤖 Algorithmes supportés

Comparez visuellement les performances des algorithmes suivants :

1. **Simple Descent** : La méthode classique.
2. **Momentum** : Ajoute de l'inertie physique pour traverser les plateaux.
3. **Nesterov** : Anticipe la pente pour éviter l'overshooting.
4. **Adam** : Adapte le *Learning Rate* dynamiquement pour chaque paramètre.

## 🚀 Démarrage Rapide

### Sur macOS & Linux (Méthode Automatique)

Un script se charge de tout (création de l'environnement virtuel, installation des dépendances, et lancement) :

```
# 1. Cloner le dépôt
git clone git@github.com:RubenWihler/labo-regression.git
cd labo-regression

# 2. Donner les droits d'exécution au script et le lancer
chmod +x run.sh && ./run.sh
```

### Sur Windows (Méthode Manuelle)

Ouvrez PowerShell ou l'Invite de commandes dans le dossier du projet :

```
# 1. Créer et activer l'environnement virtuel
python -m venv .venv
.\.venv\Scripts\activate

# 2. Installer les bibliothèques requises
pip install streamlit numpy plotly

# 3. Lancer l'interface
streamlit run labo.py
```

## 📂 Structure du Projet

Le projet est divisé en deux parties : l'interface utilisateur et le moteur mathématique.

```
📦 labo-regression
 ┣ 📜 labo.py        # Interface utilisateur (Dashboard Streamlit)
 ┣ 📜 gradient.py    # Algorithme de calcul du gradient
 ┣ 📜 dual.py        # Implémentation des Nombres Duaux (Différenciation Auto)
 ┣ 📜 run.sh         # Script d'exécution automatisé
 ┗ 📜 README.md      # Ce fichier
```

## 🛠️ Astuces d'utilisation

- **Animations Fluides :** Les graphiques 3D et les vues topographiques utilisent le moteur natif de Plotly. Vous pouvez zoomer et tourner la caméra pendant que l'algorithme cherche le minimum !
- **Rappels Mathématiques :** Ne manquez pas l'onglet "📚 Rappels Théoriques" situé au-dessus des graphiques. Il contient toutes les formules, les pseudo-codes des algorithmes et des conseils sur le choix des hyperparamètres.
- **Early Stopping :** L'entraînement s'arrête automatiquement s'il détecte que la bille a atteint un minimum absolu. Essayez d'utiliser **Adam** sur la fonction *Fond Plat (1D)* pour voir sa puissance.

*Projet réalisé dans le cadre du cours de mathématiques / algorithmique.*

Wihler Ruben