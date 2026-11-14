"""Gera diagramas conceituais para os fundamentos e a calibração."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

RAIZ = Path(__file__).resolve().parents[1]
IMG = RAIZ / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

AZUL = "#3f5d7a"
TERRA = "#b96a4c"
VERDE = "#4a5937"
CINZA = "#6e6a59"
AREIA = "#d8cdb6"
FUNDO = "#fbf8f2"


def estilo(ax):
    ax.grid(True, color=AREIA, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(FUNDO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(AREIA)
    ax.spines["bottom"].set_color(AREIA)


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(IMG / nome, bbox_inches="tight")
    plt.close(fig)


# ================================================================ viés-variância
x = np.linspace(1, 10, 240)
treino = 0.9 / (x + 0.4) + 0.12
validacao = 0.065 * (x - 4.6) ** 2 + 0.32

fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
ax.plot(x, treino, color=AZUL, linewidth=2.0, label="erro de treino")
ax.plot(x, validacao, color=TERRA, linewidth=2.0, label="erro de validação")
ax.axvline(4.6, color=VERDE, linestyle="--", linewidth=1.3)
ax.text(1.5, 1.25, "underfitting\nviés alto", ha="center", fontsize=9, color=AZUL)
ax.text(4.6, 1.35, "equilíbrio", ha="center", fontsize=9, color=VERDE)
ax.text(8.3, 1.25, "overfitting\nvariância alta", ha="center", fontsize=9, color=TERRA)
ax.set_xlabel("complexidade do modelo")
ax.set_ylabel("erro")
ax.set_ylim(0, 1.5)
ax.set_title("Viés–variância: o erro de validação tem um mínimo", loc="left", fontsize=12)
ax.legend(frameon=False, loc="lower center")
estilo(ax)
salvar(fig, "diag_bias_variancia.png")

# ================================================================ overfitting
rng = np.random.default_rng(42)
x = np.sort(rng.uniform(0, 1, 30))
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, x.size)
xs = np.linspace(0, 1, 300)

fig, eixos = plt.subplots(1, 3, figsize=(12.0, 4.4), dpi=160, sharey=True)
for ax, grau, titulo in zip(
    eixos, [1, 3, 15],
    ["Underfitting (grau 1)", "Ajuste adequado (grau 3)", "Overfitting (grau 15)"],
):
    coef = np.polyfit(x, y, grau)
    ax.scatter(x, y, s=24, color=AZUL, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.plot(xs, np.polyval(coef, xs), color=TERRA, linewidth=2.0)
    ax.set_title(titulo, loc="left", fontsize=10)
    ax.set_ylim(-1.8, 1.8)
    ax.set_xticks([])
    estilo(ax)
fig.suptitle("O mesmo dado, três complexidades", fontsize=12, x=0.02, ha="left")
salvar(fig, "diag_overfitting.png")

# ================================================================ vazamento
fig, ax = plt.subplots(figsize=(9.0, 4.4), dpi=160)
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
ax.axis("off")
ax.add_patch(Rectangle((5, 5.4), 55, 2.0, color=VERDE, alpha=0.75))
ax.add_patch(Rectangle((64, 5.4), 26, 2.0, color=TERRA, alpha=0.8))
ax.text(32.5, 6.4, "treino", ha="center", va="center", color="white", fontsize=11)
ax.text(77, 6.4, "teste", ha="center", va="center", color="white", fontsize=11)
ax.add_patch(Rectangle((18, 1.2), 34, 1.5, color=AZUL, alpha=0.8))
ax.text(35, 1.95, "pré-processamento / seleção", ha="center", va="center",
        color="white", fontsize=9)
ax.add_patch(FancyArrowPatch((77, 5.2), (52, 2.9), arrowstyle="-|>",
                             mutation_scale=18, color=TERRA, linewidth=2.0))
ax.text(64, 3.5, "vazamento", color=TERRA, fontsize=11, rotation=-20)
ax.text(50, 8.8, "O teste não pode influenciar o treino.", ha="center", fontsize=11, color=CINZA)
salvar(fig, "diag_leakage.png")

# ================================================================ CV aleatória
fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=160)
n_folds = 5
for i in range(n_folds):
    for j in range(n_folds):
        cor = TERRA if j == i else VERDE
        ax.add_patch(Rectangle((j, n_folds - i - 1), 0.95, 0.8, color=cor, alpha=0.8))
ax.set_xlim(-0.2, n_folds + 0.2)
ax.set_ylim(-0.2, n_folds + 0.2)
ax.set_xticks(np.arange(n_folds) + 0.5, [f"bloco {i+1}" for i in range(n_folds)], fontsize=8)
ax.set_yticks(np.arange(n_folds) + 0.5, [f"fold {i+1}" for i in range(n_folds)][::-1], fontsize=8)
ax.set_title("Validação cruzada: cada bloco é teste uma vez", loc="left", fontsize=12)
ax.grid(False)
salvar(fig, "diag_cv.png")

# ================================================================ CV temporal
fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=160)
blocos = 6
for i in range(4):
    for j in range(blocos):
        if j <= i + 1:
            cor = VERDE
        elif j == i + 2:
            cor = TERRA
        else:
            cor = AREIA
        ax.add_patch(Rectangle((j, 3 - i), 0.95, 0.8, color=cor, alpha=0.85))
ax.set_xlim(-0.2, blocos + 0.2)
ax.set_ylim(-0.2, 4.2)
ax.set_xticks(np.arange(blocos) + 0.5, [f"t{i+1}" for i in range(blocos)], fontsize=8)
ax.set_yticks(np.arange(4) + 0.5, ["split 4", "split 3", "split 2", "split 1"], fontsize=8)
ax.set_title("Split temporal: o passado treina; o futuro testa", loc="left", fontsize=12)
ax.grid(False)
salvar(fig, "diag_cv_temporal.png")

# ================================================================ CV espacial
fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=160)
n = 6
for i in range(n):
    for j in range(n):
        if j in (3,):
            cor = TERRA
        elif j in (2, 4):
            cor = "#e6ddca"
        else:
            cor = VERDE
        ax.add_patch(Rectangle((j, i), 0.95, 0.95, color=cor, alpha=0.85))
ax.set_xlim(-0.2, n + 0.2)
ax.set_ylim(-0.2, n + 0.2)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("CV espacial: blocos de teste e buffer", loc="left", fontsize=12)
ax.text(3.5, -0.7, "teste", ha="center", fontsize=9, color=TERRA)
ax.text(1.5, -0.7, "treino", ha="center", fontsize=9, color=VERDE)
ax.text(5.5, -0.7, "buffer", ha="center", fontsize=9, color=CINZA)
salvar(fig, "diag_cv_espacial.png")

# ================================================================ calibração vs tuning
fig, eixos = plt.subplots(1, 2, figsize=(10.0, 4.4), dpi=160)
# calibração de probabilidade
ax = eixos[0]
p = np.linspace(0.05, 0.95, 7)
obs = p + np.array([-0.05, 0.06, -0.03, 0.08, -0.02, 0.05, -0.04])
ax.plot(p, obs, marker="o", color=VERDE, label="modelo")
ax.plot([0, 1], [0, 1], color=TERRA, linestyle="--", label="perfeito")
ax.set_xlabel("probabilidade prevista")
ax.set_ylabel("fração observada")
ax.set_title("Calibração de probabilidade", loc="left", fontsize=11)
ax.legend(frameon=False)
estilo(ax)
# tuning
ax = eixos[1]
alpha = np.logspace(-4, 1, 80)
rmse = 0.065 * (np.log10(alpha) + 0.5) ** 2 + 0.32
ax.plot(alpha, rmse, color=AZUL, linewidth=2.0)
ax.set_xscale("log")
ax.axvline(alpha[np.argmin(rmse)], color=VERDE, linestyle="--")
ax.set_xlabel("alpha (log)")
ax.set_ylabel("RMSE em CV")
ax.set_title("Calibração de hiperparâmetros", loc="left", fontsize=11)
estilo(ax)
fig.suptitle("Duas coisas chamadas “calibração”", fontsize=12, x=0.02, ha="left")
salvar(fig, "diag_calibracao.png")

print("Diagramas gerados em:", IMG)

