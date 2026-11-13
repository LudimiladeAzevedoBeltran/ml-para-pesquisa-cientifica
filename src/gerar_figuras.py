"""Gera as figuras do laboratorio de interpretacao.

Uso:
    python src/gerar_figuras.py

Saida:
    docs/img/previsto_observado_bom.png
    docs/img/previsto_observado_curvatura.png
    docs/img/previsto_observado_heterocedastico.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
IMG = RAIZ / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

AZUL = "#3f5d7a"
TERRA = "#b96a4c"
VERDE = "#4a5937"
CINZA = "#6e6a59"


def estilo(ax):
    ax.grid(True, color="#d8cdb6", linewidth=0.7, alpha=0.8)
    ax.set_facecolor("#fbf8f2")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color("#d8cdb6")
    ax.spines["bottom"].set_color("#d8cdb6")


def figura_previsto_observado(y_obs, y_pred, titulo, caminho):
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160)
    ax.scatter(y_obs, y_pred, s=38, color=AZUL, alpha=0.75, edgecolor="white", linewidth=0.6)
    lim = [min(y_obs.min(), y_pred.min()) - 2, max(y_obs.max(), y_pred.max()) + 2]
    ax.plot(lim, lim, color=TERRA, linewidth=1.6, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Valor observado", fontsize=11)
    ax.set_ylabel("Valor previsto", fontsize=11)
    ax.set_title(titulo, fontsize=13, color=VERDE, loc="left", pad=10)
    r2 = r2_score(y_obs, y_pred)
    rmse = mean_squared_error(y_obs, y_pred) ** 0.5
    ax.text(
        0.98,
        0.05,
        f"R² = {r2:.3f}\nRMSE = {rmse:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=CINZA,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f3ecdf", edgecolor="#d8cdb6"),
    )
    estilo(ax)
    fig.tight_layout()
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- caso bom
x = rng.uniform(0, 10, 140)
y = 4 + 1.8 * x + rng.normal(0, 2.2, x.size)
X_tr, X_te, y_tr, y_te = train_test_split(
    x.reshape(-1, 1), y, test_size=0.3, random_state=42
)
modelo = LinearRegression().fit(X_tr, y_tr)
pred = modelo.predict(X_te)
figura_previsto_observado(
    y_te,
    pred,
    "Ajuste saudável: pontos em torno da linha 1:1",
    IMG / "previsto_observado_bom.png",
)

# --------------------------------------------------------- caso curvatura
x2 = rng.uniform(-3, 3, 160)
y2 = 0.9 * x2**2 + rng.normal(0, 1.2, x2.size)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(
    x2.reshape(-1, 1), y2, test_size=0.3, random_state=42
)
modelo2 = LinearRegression().fit(X2_tr, y2_tr)
pred2 = modelo2.predict(X2_te)
figura_previsto_observado(
    y2_te,
    pred2,
    "Underfitting: relação não linear ajustada com reta",
    IMG / "previsto_observado_curvatura.png",
)

# ------------------------------------------------- caso heterocedastico
x3 = rng.uniform(1, 10, 160)
y3 = 2 + 1.5 * x3 + rng.normal(0, 0.8 * x3, x3.size)
X3_tr, X3_te, y3_tr, y3_te = train_test_split(
    x3.reshape(-1, 1), y3, test_size=0.3, random_state=42
)
modelo3 = LinearRegression().fit(X3_tr, y3_tr)
pred3 = modelo3.predict(X3_te)
figura_previsto_observado(
    y3_te,
    pred3,
    "Heterocedasticidade: a dispersão cresce com o valor",
    IMG / "previsto_observado_heterocedastico.png",
)

# ------------------------------------------------- caso de extrapolacao (quiz)
x4 = rng.uniform(0, 10, 220)
y4 = 10 * (1 - np.exp(-0.5 * x4)) + rng.normal(0, 0.45, x4.size)
treino = x4 < 4
modelo4 = LinearRegression().fit(x4[treino].reshape(-1, 1), y4[treino])
pred4 = modelo4.predict(x4[~treino].reshape(-1, 1))
figura_previsto_observado(
    y4[~treino],
    pred4,
    "Caso para diagnosticar: qual é o problema?",
    IMG / "previsto_observado_quiz.png",
)

print("Figuras geradas em:", IMG)


