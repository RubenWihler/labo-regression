import streamlit as st
import numpy as np
import plotly.graph_objects as go

# === IMPORT DU BACKEND MATHÉMATIQUE ===
from gradient import compute_gradient

# ==========================================
# 1. MODÈLES ET FONCTIONS DE COÛT (Régression)
# ==========================================
def model_lin(x, a, b): return a * x + b
def mse_cost(a, b, X, Y):
    cost = 0
    for xi, yi in zip(X, Y): cost = cost + ((a * xi + b) - yi)**2
    return cost / len(X)

def model_log(x, a, b):
    z = -(a * x + b)
    return 1 / (1 + (z.exp() if hasattr(z, 'exp') else np.exp(z)))

def bce_cost(a, b, X, Y):
    cost = 0
    for xi, yi in zip(X, Y):
        p = model_log(xi, a, b) * 0.999998 + 0.000001
        log_p = p.log() if hasattr(p, 'log') else np.log(p)
        log_p_inv = (1 - p).log() if hasattr(1 - p, 'log') else np.log(1 - p)
        cost = cost - (log_p if yi == 1 else log_p_inv)
    return cost / len(X)

# ==========================================
# 2. FONCTIONS DE TESTS (Descente Pure)
# ==========================================
def fn_quad(x): return x**2
def fn_puits(x): return x**4 - 4*(x**2) + x 
def fn_asym(x): return 0.25*(x**4) - 0.333*(x**3) - (x**2) + 1
def fn_plat(x): return 0.05*(x**6)
def fn_ondul(x): return 0.01*(x**6) - 0.1*(x**4) + 0.2*(x**2) - 0.05*x

def fn_sphere(x, y): return x**2 + y**2
def fn_rosenbrock(x, y): return (1 - x)**2 + 100 * (y - x**2)**2
def fn_beale(x, y): return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
def fn_booth(x, y): return (x + 2*y - 7)**2 + (2*x + y - 5)**2

# ==========================================
# 3. MOTEUR D'OPTIMISATION (Avec Early Stopping)
# ==========================================
def compute_optimization_history(f_cost, p_start, algo, lr, max_it, gamma=0.9, beta1=0.9, beta2=0.999, tol=1e-5):
    p = np.array(p_start, dtype=float)
    hist_p = [p.copy()]
    eval_f = lambda pt: f_cost(*pt)
    hist_c = [eval_f(p)]
    step, m, v = np.zeros_like(p), np.zeros_like(p), np.zeros_like(p)
    converged = False
    
    for i in range(max_it):
        p_old = p.copy()
        grad = compute_gradient(f_cost, p)
        
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)): grad = np.zeros_like(grad)
        else: grad = np.clip(grad, -100, 100) 
            
        if algo == "simple_descent": p = p - lr * grad
        elif algo == "momentum":
            step = gamma * step + lr * grad
            p = p - step
        elif algo == "nesterov":
            grad_anticip = compute_gradient(f_cost, p - gamma * step)
            if np.any(np.isnan(grad_anticip)) or np.any(np.isinf(grad_anticip)): grad_anticip = np.zeros_like(grad_anticip)
            else: grad_anticip = np.clip(grad_anticip, -100, 100)
            step = gamma * step + lr * grad_anticip
            p = p - step
        elif algo == "adam":
            epsilon = 1e-8
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** (i + 1))
            v_hat = v / (1 - beta2 ** (i + 1))
            nu = lr / (np.sqrt(v_hat) + epsilon)
            p = p - nu * m_hat
            
        p = np.clip(p, -50, 50)
        
        if np.max(np.abs(p - p_old)) < tol:
            converged = True
            hist_p.append(p.copy())
            hist_c.append(eval_f(p))
            break
            
        hist_p.append(p.copy())
        c = eval_f(p)
        if np.isnan(c) or np.isinf(c): c = hist_c[-1]
        hist_c.append(c)
        
    return np.array(hist_p), np.array(hist_c), converged

# ==========================================
# 4. CONFIGURATION UI
# ==========================================
st.set_page_config(page_title="Labo Mathématiques", layout="wide")

DEFAULTS = {
    "Sphère (2D)":      {"lr": {"simple_descent": 0.1, "momentum": 0.1, "nesterov": 0.1, "adam": 0.5}, "range": (-10, 10), "fn": fn_sphere},
    "Rosenbrock (2D)":  {"lr": {"simple_descent": 0.001, "momentum": 0.001, "nesterov": 0.001, "adam": 0.1}, "range": (-3, 3), "fn": fn_rosenbrock},
    "Beale (2D)":       {"lr": {"simple_descent": 0.01, "momentum": 0.01, "nesterov": 0.01, "adam": 0.2}, "range": (-5, 5), "fn": fn_beale},
    "Booth (2D)":       {"lr": {"simple_descent": 0.05, "momentum": 0.05, "nesterov": 0.05, "adam": 1.0}, "range": (-10, 10), "fn": fn_booth},
    "Quadratique (1D)": {"lr": {"simple_descent": 0.1, "momentum": 0.1, "nesterov": 0.1, "adam": 0.5}, "range": (-10, 10), "fn": fn_quad},
    "Double Puits (1D)":{"lr": {"simple_descent": 0.01, "momentum": 0.01, "nesterov": 0.01, "adam": 0.2}, "range": (-3, 3), "fn": fn_puits},
    "Asymétrique (1D)": {"lr": {"simple_descent": 0.05, "momentum": 0.05, "nesterov": 0.05, "adam": 0.5}, "range": (-3, 4), "fn": fn_asym},
    "Fond Plat (1D)":   {"lr": {"simple_descent": 0.01, "momentum": 0.01, "nesterov": 0.01, "adam": 0.1}, "range": (-3, 3), "fn": fn_plat},
    "Ondulation (1D)":  {"lr": {"simple_descent": 0.1, "momentum": 0.1, "nesterov": 0.1, "adam": 0.5}, "range": (-4, 4), "fn": fn_ondul}
}

st.sidebar.title("🛠️ Configuration")
vue = st.sidebar.radio("Mode :", ["Régression Linéaire", "Régression Logistique", "Descente de Gradient"])

# Configuration des données
if vue in ["Régression Linéaire", "Régression Logistique"]:
    st.sidebar.markdown("### Données")
    n_samples = st.sidebar.slider("Nombre de données (N)", 10, 1000, 50, step=10)
else:
    n_samples = 0

fn_name, fn_test = None, None
if vue == "Descente de Gradient":
    st.sidebar.markdown("### Fonction à optimiser")
    dim = st.sidebar.radio("Dimension", ["2D (Surfaces)", "1D (Courbes)"], horizontal=True)
    if "2D" in dim: fn_name = st.sidebar.selectbox("Fonction :", ["Sphère (2D)", "Rosenbrock (2D)", "Beale (2D)", "Booth (2D)"])
    else: fn_name = st.sidebar.selectbox("Fonction :", ["Quadratique (1D)", "Double Puits (1D)", "Asymétrique (1D)", "Fond Plat (1D)", "Ondulation (1D)"])
    fn_test = DEFAULTS[fn_name]["fn"]
    grid_range = DEFAULTS[fn_name]["range"]
else: grid_range = (-5, 5)

st.sidebar.markdown("### Algorithme")
algo = st.sidebar.selectbox("Méthode", ["simple_descent", "momentum", "nesterov", "adam"])
def_lr = DEFAULTS[fn_name]["lr"][algo] if vue == "Descente de Gradient" else 0.01
lr = st.sidebar.number_input("Learning Rate (α)", value=def_lr, step=0.005, format="%.4f")
max_iter = st.sidebar.slider("Itérations Max", 10, 10000, 1000, step=10)

gamma, beta1, beta2 = 0.9, 0.9, 0.999
if algo in ["momentum", "nesterov"]: gamma = st.sidebar.slider("Inertie (γ)", 0.0, 0.99, 0.9)
elif algo == "adam":
    c1, c2 = st.sidebar.columns(2)
    beta1 = c1.slider("Beta 1", 0.0, 0.999, 0.9)
    beta2 = c2.slider("Beta 2", 0.0, 0.999, 0.999)

if "start_x" not in st.session_state: st.session_state.start_x, st.session_state.start_y = 0.0, 0.0
if "current_fn" not in st.session_state or st.session_state.current_fn != fn_name:
    st.session_state.current_fn, st.session_state.randomize_start = fn_name, True

if st.sidebar.button("🎲 Point de départ Aléatoire", use_container_width=True): st.session_state.randomize_start = True
if st.session_state.get("randomize_start", False):
    st.session_state.start_x = float(np.random.uniform(grid_range[0], grid_range[1]))
    st.session_state.start_y = float(np.random.uniform(grid_range[0], grid_range[1]))
    st.session_state.randomize_start = False

if vue == "Descente de Gradient":
    if "2D" in dim:
        cx, cy = st.sidebar.columns(2)
        start_x = cx.number_input("X0", value=st.session_state.start_x, format="%.2f")
        start_y = cy.number_input("Y0", value=st.session_state.start_y, format="%.2f")
        p_start = [start_x, start_y]
    else:
        start_x = st.sidebar.number_input("X0", value=st.session_state.start_x, format="%.2f")
        p_start = [start_x]
else:
    ca, cb = st.sidebar.columns(2)
    p_start = [ca.number_input("a0", value=st.session_state.start_x), cb.number_input("b0", value=st.session_state.start_y)]

# ==========================================
# 5. LOGIQUE D'ANIMATION NATIVE PLOTLY
# ==========================================
def setup_native_animation(fig, frames_data, speed=50, redraw=False):
    frames = [go.Frame(data=[d], traces=[1], name=str(k)) for k, d in enumerate(frames_data)]
    fig.frames = frames
    
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=-0.15,
            xanchor="left", yanchor="top", direction="right",
            buttons=[
                dict(label="▶ Play", method="animate", 
                     args=[None, dict(frame=dict(duration=speed, redraw=redraw), 
                                      transition=dict(duration=0), 
                                      fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate", 
                     args=[[None], dict(frame=dict(duration=0, redraw=False), 
                                        transition=dict(duration=0), 
                                        mode="immediate")])
            ]
        )],
        sliders=[dict(
            active=0, yanchor="top", xanchor="left",
            currentvalue=dict(font=dict(size=12), prefix="Étape: ", visible=True, xanchor="right"),
            transition=dict(duration=0), pad=dict(b=10, t=50), len=0.9, x=0.1, y=-0.15,
            steps=[dict(method="animate",
                        args=[[str(k)], dict(mode="immediate", transition=dict(duration=0), frame=dict(duration=speed, redraw=redraw))],
                        label=str(k)) for k in range(len(frames))]
        )]
    )

# ==========================================
# BOUTON D'ENTRAÎNEMENT
# ==========================================
def generate_random_data(vue_type, n):
    if vue_type == "Régression Linéaire":
        X = np.linspace(-5, 5, n)
        Y = np.random.uniform(-2,2)*X + np.random.uniform(-1,1) + np.random.normal(0, 1.5, size=n)
        return X, Y, (-5, 5)
    elif vue_type == "Régression Logistique":
        n_half = n // 2
        c1, c2, spread = np.random.uniform(-4, -1), np.random.uniform(1, 4), np.random.uniform(0.5, 2.0)
        X = np.concatenate([np.random.normal(c1, spread, n_half), np.random.normal(c2, spread, n - n_half)])
        Y = np.concatenate([np.zeros(n_half), np.ones(n - n_half)])
        return X, Y, (-10, 10)
    return None, None, None

if st.sidebar.button("🚀 Lancer l'entraînement", type="primary"):
    with st.spinner("Calcul en cours..."):
        if vue in ["Régression Linéaire", "Régression Logistique"]:
            X, Y, _ = generate_random_data(vue, n_samples)
            st.session_state.X_data, st.session_state.Y_data = X, Y
            cost_fn = lambda a, b: mse_cost(a, b, X, Y) if vue == "Régression Linéaire" else bce_cost(a, b, X, Y)
        else: cost_fn = fn_test

        hp, hc, converged = compute_optimization_history(cost_fn, p_start, algo, lr, max_iter, gamma, beta1, beta2)
        st.session_state.hist_p, st.session_state.hist_c, st.session_state.converged = hp, hc, converged
        
        A_vals = np.linspace(grid_range[0], grid_range[1], 50)
        B_vals = np.linspace(grid_range[0], grid_range[1], 50)
        A_grid, B_grid = np.meshgrid(A_vals, B_vals)
        if vue != "Descente de Gradient" or "2D" in dim:
            Z_grid = np.array([[cost_fn(a, b) for a in A_vals] for b in B_vals])
            st.session_state.A_grid, st.session_state.B_grid, st.session_state.Z_grid = A_grid, B_grid, Z_grid
        else:
            st.session_state.X_grid = np.linspace(grid_range[0], grid_range[1], 200)
            st.session_state.Y_grid = np.array([float(cost_fn(x)) for x in st.session_state.X_grid])
        
        st.session_state.step_idx = 0
        st.session_state.is_playing = False

# ==========================================
# AFFICHAGE DES RÉSULTATS
# ==========================================
if "hist_p" in st.session_state:
    st.title(f"📊 {vue} - Résultat Final")
    hp, hc, converged = st.session_state.hist_p, st.session_state.hist_c, st.session_state.converged
    final_p, final_c = hp[-1], hc[-1]

    # --- RAPPELS THÉORIQUES ENRICHIS ---
    with st.expander("📚 Rappels Théoriques et Mathématiques", expanded=False):
        tab_math, tab_algo = st.tabs(["🧮 Modèle & Mathématiques", "🤖 Analyse de l'Algorithme"])
        
        with tab_math:
            if vue == "Régression Linéaire":
                st.markdown("### 📈 Régression Linéaire (Moindres Carrés)")
                st.markdown("**1. Le Modèle d'Hypothèse**\nNotre objectif est de trouver la droite qui passe au plus près de tous les points :")
                st.latex(r"\hat{y}_i = f(x_i) = ax_i + b")
                st.markdown("**2. Les Résidus (Erreurs)**\nPour chaque point, on calcule l'écart vertical entre la valeur réelle $y_i$ et la prédiction $\hat{y}_i$ :")
                st.latex(r"\epsilon_i = \hat{y}_i - y_i")
                st.markdown("**3. La Fonction de Coût (MSE - Mean Squared Error)**\nOn minimise la moyenne des erreurs au carré :")
                st.latex(r"J(a, b) = \frac{1}{n} \sum_{i=1}^{n} \epsilon_i^2")
                
            elif vue == "Régression Logistique":
                st.markdown("### 🔮 Régression Logistique & Maximum de Vraisemblance")
                st.markdown("**1. Le Modèle (Fonction Sigmoïde)**\nPour classifier entre 0 et 1, on écrase notre droite $z = ax+b$ en une probabilité avec la sigmoïde $\sigma$ :")
                st.latex(r"P(y=1|x) = \hat{p}_i = \sigma(z) = \frac{1}{1 + e^{-(ax_i + b)}}")
                st.markdown("**2. La Vraisemblance (Likelihood)**\nOn cherche les paramètres qui maximisent la probabilité d'avoir observé notre jeu de données :")
                st.latex(r"\mathcal{L}(a, b) = \prod_{i=1}^{n} \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}")
                st.markdown("**3. Log-Vraisemblance et Entropie Croisée (BCE)**\nMaximiser la Log-Vraisemblance revient exactement à minimiser son opposé, la BCE :")
                st.latex(r"Log\text{-}Vraisemblance = \sum_{i=1}^{n} \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]")
                st.latex(r"BCE\_Cost = - \frac{1}{n} \times Log\text{-}Vraisemblance")

            elif vue == "Descente de Gradient":
                st.markdown("### 🔬 Formules des Fonctions de Test")
                if fn_name == "Sphère (2D)": 
                    st.latex(r"f(x, y) = x^2 + y^2")
                    st.markdown("*Fonction convexe parfaite. Tous les algorithmes convergent facilement au centre.*")
                elif fn_name == "Rosenbrock (2D)": 
                    st.latex(r"f(x, y) = (1-x)^2 + 100(y-x^2)^2")
                    st.markdown("*La fameuse 'Vallée de la mort'. Le minimum global est caché au bout d'une vallée plate.*")
                elif fn_name == "Beale (2D)": 
                    st.latex(r"f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2")
                elif fn_name == "Booth (2D)": 
                    st.latex(r"f(x,y) = (x+2y-7)^2 + (2x+y-5)^2")
                elif fn_name == "Quadratique (1D)": st.latex(r"f(x) = x^2")
                elif fn_name == "Double Puits (1D)": 
                    st.latex(r"f(x) = x^4 - 4x^2 + x")
                elif fn_name == "Asymétrique (1D)": st.latex(r"f(x) = \frac{1}{4}x^4 - \frac{1}{3}x^3 - x^2 + 1")
                elif fn_name == "Fond Plat (1D)": st.latex(r"f(x) = 0.05 x^6")
                elif fn_name == "Ondulation (1D)": st.latex(r"f(x) = 0.01 x^6 - 0.1 x^4 + 0.2 x^2 - 0.05 x")

        with tab_algo:
            algo_title = algo.replace("_", " ").title()
            st.markdown(f"### {algo_title}")
            col_pseudo, col_explic = st.columns(2)
            with col_pseudo:
                st.markdown("**💻 Pseudo-code :**")
                if algo == "simple_descent": st.code("x = x - lr * grad", language="python")
                elif algo == "momentum": st.code("v = gamma * v + lr * grad\nx = x - v", language="python")
                elif algo == "nesterov": st.code("grad_anticipé = gradient(x - gamma * v)\nv = gamma * v + lr * grad_anticipé\nx = x - v", language="python")
                elif algo == "adam": st.code("m = beta1 * m + (1 - beta1) * grad\nv = beta2 * v + (1 - beta2) * (grad^2)\nm_hat = m / (1 - beta1^t)\nv_hat = v / (1 - beta2^t)\nx = x - (lr / (sqrt(v_hat) + epsilon)) * m_hat", language="python")
            with col_explic:
                st.markdown("**💡 Points forts & faibles :**")
                if algo == "simple_descent": st.markdown("- **Forces :** Intuitif.\n- **Faiblesses :** Lent, bloque dans les minimums locaux.")
                elif algo == "momentum": st.markdown("- **Forces :** Inertie. Permet de traverser les plateaux.\n- **Faiblesses :** Overshooting (Dépassement de cible).")
                elif algo == "nesterov": st.markdown("- **Forces :** Améliore le Momentum en anticipant la pente.\n- **Faiblesses :** Calcul complexe.")
                elif algo == "adam": st.markdown("- **Forces :** L'algorithme roi. Adapte le LR dynamiquement.\n- **Faiblesses :** Précision fine parfois moyenne.")
                    
    st.markdown("---")
    
    # --- INDICATION DE CONVERGENCE ET METRIQUES ---
    if converged:
        st.success(f"✅ **L'algorithme a convergé !** Le minimum a été trouvé en seulement **{len(hp)-1} itérations** (arrêt prématuré).")
    else:
        st.error(f"❌ **L'algorithme n'a pas convergé.** Il a atteint la limite de **{max_iter} itérations** sans se stabiliser totalement.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Itérations effectuées", len(hp)-1)
    m4.metric("Coût Final (Loss)", f"{final_c:.4f}")
    if len(final_p) == 2:
        m2.metric("Paramètre A / X (Final)", f"{final_p[0]:.3f}")
        m3.metric("Paramètre B / Y (Final)", f"{final_p[1]:.3f}")
    else:
        m2.metric("Paramètre X (Final)", f"{final_p[0]:.3f}")

    # Préparation des frames d'animation (max 60 frames pour la performance web)
    step = max(1, len(hp) // 60) 
    indices = list(range(0, len(hp), step))
    if indices[-1] != len(hp)-1: indices.append(len(hp)-1)

    col_g1, col_g2 = st.columns(2)

    # --- GRAPHIQUE 1 (Modèle ou Contour animé) ---
    with col_g1:
        fig1 = go.Figure()
        if vue in ["Régression Linéaire", "Régression Logistique"]:
            X_d, Y_d = st.session_state.X_data, st.session_state.Y_data
            if vue == "Régression Linéaire":
                fig1.add_trace(go.Scatter(x=X_d, y=Y_d, mode='markers', name='Data', marker=dict(color='#3498db')))
                x_line = np.array([min(X_d), max(X_d)])
                fig1.add_trace(go.Scatter(x=x_line, y=model_lin(x_line, hp[0,0], hp[0,1]), mode='lines', name='Modèle', line=dict(color='red', width=3)))
                frames_data = [go.Scatter(x=x_line, y=model_lin(x_line, hp[i,0], hp[i,1])) for i in indices]
            else:
                fig1.add_trace(go.Scatter(x=X_d, y=Y_d, mode='markers', name='Data', marker=dict(color=Y_d, colorscale='Viridis')))
                x_sig = np.linspace(min(X_d), max(X_d), 100)
                fig1.add_trace(go.Scatter(x=x_sig, y=[model_log(x, hp[0,0], hp[0,1]) for x in x_sig], mode='lines', name='Sigmoïde', line=dict(color='red')))
                frames_data = [go.Scatter(y=[model_log(x, hp[i,0], hp[i,1]) for x in x_sig]) for i in indices]
            
            # uirevision="constant" verrouille le zoom/pan de l'utilisateur
            fig1.update_layout(title="Évolution du Modèle", height=500, margin=dict(b=80), uirevision="constant")
            setup_native_animation(fig1, frames_data, speed=50, redraw=False)
            
        elif "1D" in dim:
            fig1.add_trace(go.Scatter(x=st.session_state.X_grid, y=st.session_state.Y_grid, mode='lines', name='f(x)'))
            fig1.add_trace(go.Scatter(x=[hp[0,0]], y=[hc[0]], mode='markers+lines', name='Bille', marker=dict(size=12, color='red')))
            frames_data = [go.Scatter(x=hp[:i+1, 0], y=hc[:i+1]) for i in indices]
            fig1.update_layout(title="Descente 1D (Animée)", height=500, margin=dict(b=80), uirevision="constant")
            setup_native_animation(fig1, frames_data, speed=50, redraw=False)
            
        else: # Contour 2D
            fig1.add_trace(go.Contour(z=st.session_state.Z_grid, x=st.session_state.A_grid[0], y=st.session_state.B_grid[:,0], colorscale='Viridis', opacity=0.6))
            fig1.add_trace(go.Scatter(x=[hp[0,0]], y=[hp[0,1]], mode='lines+markers', name='Chemin', line=dict(color='white')))
            frames_data = [go.Scatter(x=hp[:i+1, 0], y=hp[:i+1, 1]) for i in indices]
            fig1.update_layout(title="Vue Topographique", height=500, margin=dict(b=80), uirevision="constant")
            
            # --- LE FIX TOPOGRAPHIQUE EST ICI ---
            # redraw=True force Plotly à redessiner la frame de A à Z, ce qui empêche le fond de disparaitre.
            setup_native_animation(fig1, frames_data, speed=50, redraw=True) 
        
        st.plotly_chart(fig1, use_container_width=True)

    # --- GRAPHIQUE 2 (Espace 3D animé) ---
    with col_g2:
        if vue != "Descente de Gradient" or "2D" in dim:
            fig2 = go.Figure()
            fig2.add_trace(go.Surface(z=st.session_state.Z_grid, x=st.session_state.A_grid, y=st.session_state.B_grid, colorscale='Viridis', opacity=0.7, showscale=False))
            fig2.add_trace(go.Scatter3d(x=[hp[0,0]], y=[hp[0,1]], z=[hc[0]], mode='lines+markers', line=dict(color='red', width=4), marker=dict(size=4)))
            frames_data = [go.Scatter3d(x=hp[:i+1, 0], y=hp[:i+1, 1], z=hc[:i+1]) for i in indices]
            
            # --- LE FIX CAMÉRA 3D EST ICI ---
            # uirevision="constant" dit à Plotly de garder en mémoire la caméra que tu as choisie
            fig2.update_layout(title="Surface de Coût 3D", scene=dict(xaxis_title='A/X', yaxis_title='B/Y', zaxis_title='Coût'), height=500, margin=dict(b=80), uirevision="constant")
            # redraw=False empêche l'écran de clignoter, le rendu WebGL s'occupe de mettre à jour la ligne
            setup_native_animation(fig2, frames_data, speed=30, redraw=True) 
            
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("La vue 3D est réservée aux fonctions de dimension 2. Utilise le graphique 1D à gauche pour visualiser la descente.")

    # ==========================================
    # STATISTIQUES AVANCÉES (Statiques, sur modèle final)
    # ==========================================
    if vue in ["Régression Linéaire", "Régression Logistique"]:
        st.markdown("---")
        st.markdown("### 🔬 Analyses Statistiques du Modèle Final")
        col_s1, col_s2 = st.columns(2)
        X_data, Y_data = st.session_state.X_data, st.session_state.Y_data

        if vue == "Régression Linéaire":
            Y_pred = model_lin(X_data, final_p[0], final_p[1])
            residuals = Y_data - Y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((Y_data - np.mean(Y_data))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            mae = np.mean(np.abs(residuals))
            
            met1, met2, met3, met4 = st.columns(4)
            met1.metric("Score R²", f"{r2:.4f}")
            met2.metric("Erreur Absolue Moyenne (MAE)", f"{mae:.4f}")
            
            with col_s1:
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=X_data, y=residuals, mode='markers', marker=dict(color='#e67e22', size=8)))
                fig_res.add_hline(y=0, line_dash="dash", line_color="black")
                fig_res.update_layout(height=350, template='plotly_white', title="Distribution des Résidus (Erreurs)", xaxis_title="X", yaxis_title="Erreur (Réel - Prédit)")
                st.plotly_chart(fig_res, use_container_width=True)
                
            with col_s2:
                fig_yp = go.Figure()
                fig_yp.add_trace(go.Scatter(x=Y_data, y=Y_pred, mode='markers', marker=dict(color='#9b59b6', size=8)))
                min_val, max_val = min(np.min(Y_data), np.min(Y_pred)), max(np.max(Y_data), np.max(Y_pred))
                fig_yp.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(dash='dash', color='grey'), name='Modèle Parfait'))
                fig_yp.update_layout(height=350, template='plotly_white', title="Valeurs Réelles vs Prédictions", xaxis_title="Valeur Réelle (Y)", yaxis_title="Valeur Prédite (Y_pred)", showlegend=False)
                st.plotly_chart(fig_yp, use_container_width=True)

        else:
            Y_pred_prob = np.array([float(model_log(x, final_p[0], final_p[1])) for x in X_data])
            Y_pred_class = (Y_pred_prob >= 0.5).astype(int)
            
            tp, tn = np.sum((Y_pred_class == 1) & (Y_data == 1)), np.sum((Y_pred_class == 0) & (Y_data == 0))
            fp, fn = np.sum((Y_pred_class == 1) & (Y_data == 0)), np.sum((Y_pred_class == 0) & (Y_data == 1))
            
            accuracy = (tp + tn) / len(Y_data)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            n_tot = len(X_data)
            log_likelihood_history = [-c * n_tot for c in hc]
            final_ll = log_likelihood_history[-1]
            
            met1, met2, met3, met4 = st.columns(4)
            met1.metric("Accuracy (Précision globale)", f"{accuracy*100:.1f}%")
            met2.metric("Score F1", f"{f1:.3f}")
            met3.metric("Log-Vraisemblance finale", f"{final_ll:.2f}")
            met4.metric("Rappel", f"{recall:.3f}")
            
            with col_s1:
                z = [[tn, fp], [fn, tp]]
                fig_conf = go.Figure(data=go.Heatmap(z=z, x=['Prédit 0', 'Prédit 1'], y=['Vrai 0', 'Vrai 1'], colorscale='Blues', texttemplate="%{z}", textfont={"size":20}, showscale=False))
                fig_conf.update_layout(height=350, template='plotly_white', title="Matrice de Confusion")
                st.plotly_chart(fig_conf, use_container_width=True)
                
            with col_s2:
                fig_ll = go.Figure()
                fig_ll.add_trace(go.Scatter(x=list(range(len(log_likelihood_history))), y=log_likelihood_history, mode='lines', name='Log-Vraisemblance', line=dict(color='#2ecc71', width=3)))
                fig_ll.update_layout(height=350, template='plotly_white', title="Évolution de la Log-Vraisemblance", xaxis_title="Itérations", yaxis_title="Log-Vraisemblance (Max=0)")
                st.plotly_chart(fig_ll, use_container_width=True)

else:
    st.info("👈 Configure et lance l'entraînement pour commencer !")
