"""Gera as figuras das fichas de modelo (Parte II).

Uso:
    python src/gerar_figuras_modelos.py

Saída:
    docs/img/ridge_lasso_caminho.png
    docs/img/logistica_sigmoide.png
    docs/img/arvore_decisao.png
    docs/img/random_forest_oob.png
    docs/img/gradient_boosting_curvas.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
IMG = RAIZ / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

AZUL = "#3f5d7a"
TERRA = "#b96a4c"
VERDE = "#4a5937"
CINZA = "#6e6a59"
AREIA = "#d8cdb6"


def estilo(ax):
    ax.grid(True, color=AREIA, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor("#fbf8f2")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(AREIA)
    ax.spines["bottom"].set_color(AREIA)


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(IMG / nome, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Ridge/Lasso
dados = load_diabetes()
X = StandardScaler().fit_transform(dados.data)
y = dados.target
alphas = np.logspace(-4, 1, 120)
_, coefs, _ = lasso_path(X, y, alphas=alphas)
lasso = LassoCV(cv=5, random_state=42).fit(X, y)

fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=160)
cores = [AZUL, TERRA, VERDE, "#a8761f", "#6e6a59", "#8a5a83",
         "#3f7a7a", "#7a3f5d", "#5d7a3f", "#5d5d7a"]
for i, nome in enumerate(dados.feature_names):
    ax.plot(alphas, coefs[i], color=cores[i % len(cores)], linewidth=1.5, label=nome)
ax.axvline(lasso.alpha_, color=CINZA, linestyle="--", linewidth=1.3)
ax.set_xscale("log")
ax.set_xlabel("alpha (escala log)")
ax.set_ylabel("coeficiente")
ax.set_title("Caminho do Lasso: os coeficientes encolhem e viram zero", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=7.5, ncol=2)
estilo(ax)
salvar(fig, "ridge_lasso_caminho.png")

# ---------------------------------------------------------------- sigmoide
z = np.linspace(-6, 6, 240)
p = 1 / (1 + np.exp(-z))

fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=160)
ax.plot(z, p, color=AZUL, linewidth=2.0)
ax.axhline(0.5, color=TERRA, linestyle="--", linewidth=1.3)
ax.axvline(0, color=CINZA, linestyle=":", linewidth=1.2)
ax.fill_between(z, 0, p, where=z >= 0, color=TERRA, alpha=0.10)
ax.fill_between(z, 0, p, where=z < 0, color=AZUL, alpha=0.10)
ax.text(-5.5, 0.12, "classe 0", fontsize=10, color=AZUL)
ax.text(3.2, 0.82, "classe 1", fontsize=10, color=TERRA)
ax.annotate("limiar 0,5", xy=(0, 0.5), xytext=(1.2, 0.35),
            fontsize=9, color=TERRA,
            arrowprops=dict(arrowstyle="->", color=TERRA, lw=1))
ax.set_xlabel("z = b0 + b1·x1 + ... + bp·xp")
ax.set_ylabel("probabilidade prevista")
ax.set_ylim(-0.02, 1.02)
ax.set_title("Regressão logística: probabilidade e limiar", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "logistica_sigmoide.png")

# ---------------------------------------------------------------- árvore
df = pd.read_csv(DATA / "exemplo_ch4.csv")
X = df[["temperatura", "umidade", "vento", "pressao"]]
y = df["ch4"]
arvore = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)

fig, ax = plt.subplots(figsize=(13.5, 7.0), dpi=150)
plot_tree(
    arvore,
    feature_names=list(X.columns),
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    impurity=False,
    proportion=False,
)
ax.set_title("Árvore de decisão (max_depth = 3): cada nó é uma pergunta", loc="left", fontsize=12)
salvar(fig, "arvore_decisao.png")

# ---------------------------------------------------------------- Random Forest OOB
rf = RandomForestRegressor(
    n_estimators=10, warm_start=True, oob_score=True,
    random_state=42, n_jobs=-1,
)
ns = list(range(10, 301, 10))
oob = []
for n in ns:
    rf.set_params(n_estimators=n)
    rf.fit(X, y)
    oob.append(np.sqrt(mean_squared_error(y, rf.oob_prediction_)))

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
ax.plot(ns, oob, color=AZUL, linewidth=1.8, marker="o", markersize=3)
platô = oob[-1]
ax.axhline(platô, color=TERRA, linestyle="--", linewidth=1.2,
           label=f"platô ≈ {platô:.2f}")
ax.set_xlabel("n_estimators (número de árvores)")
ax.set_ylabel("RMSE OOB")
ax.set_title("Random Forest: o erro OOB estabiliza com o número de árvores", loc="left", fontsize=12)
ax.legend(frameon=False)
estilo(ax)
salvar(fig, "random_forest_oob.png")

# ---------------------------------------------------------------- Gradient Boosting
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=160)
for lr, cor in [(0.01, AZUL), (0.1, VERDE), (0.3, TERRA)]:
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=lr, max_depth=3, random_state=42
    ).fit(X_tr, y_tr)
    tr = [mean_squared_error(y_tr, p) ** 0.5 for p in gb.staged_predict(X_tr)]
    te = [mean_squared_error(y_te, p) ** 0.5 for p in gb.staged_predict(X_te)]
    eixo = np.arange(1, 301)
    ax.plot(eixo, tr, color=cor, linewidth=1.2, alpha=0.45, linestyle=":")
    ax.plot(eixo, te, color=cor, linewidth=1.8, label=f"learning_rate = {lr}")
ax.set_xlabel("n_estimators (árvores acumuladas)")
ax.set_ylabel("RMSE")
ax.set_title("Gradient boosting: learning rate e número de árvores", loc="left", fontsize=12)
ax.legend(frameon=False)
estilo(ax)
salvar(fig, "gradient_boosting_curvas.png")

print("Figuras de modelos geradas em:", IMG)

