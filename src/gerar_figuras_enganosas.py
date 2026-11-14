"""Gera as figuras da página "Gráficos enganosos"."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
IMG = RAIZ / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

AZUL = "#3f5d7a"
TERRA = "#b96a4c"
VERDE = "#4a5937"
CINZA = "#6e6a59"
AREIA = "#d8cdb6"
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]


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


# pizza vs barras
categorias = ["A", "B", "C", "D", "E", "F"]
valores = np.array([30, 25, 20, 15, 7, 3])
fig, eixos = plt.subplots(1, 2, figsize=(10.0, 4.4), dpi=160)
eixos[0].pie(valores, labels=categorias, autopct="%1.0f%%",
             colors=OKABE, textprops={"fontsize": 9})
eixos[0].set_title("Pizza: difícil comparar e ordenar", loc="left", fontsize=11)
ordem = np.argsort(valores)
eixos[1].barh(np.array(categorias)[ordem], valores[ordem], color=AZUL, alpha=0.85)
eixos[1].set_xlabel("Valor")
eixos[1].set_title("Barras: comparação direta", loc="left", fontsize=11)
estilo(eixos[1])
salvar(fig, "graf_eng_pizza.png")

# 3D bars vs 2D
fig = plt.figure(figsize=(10.0, 4.6), dpi=160)
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
x = np.arange(len(categorias))
y = np.zeros_like(x)
ax3.bar3d(x, y, 0, 0.8, 0.8, valores, color=OKABE, alpha=0.85)
ax3.set_xticks(x + 0.4, categorias, fontsize=8)
ax3.set_yticks([])
ax3.set_title("3D: perspectiva distorce a comparação", loc="left", fontsize=10)
ax2 = fig.add_subplot(1, 2, 2)
ax2.bar(categorias, valores, color=AZUL, alpha=0.85)
ax2.set_ylabel("Valor")
ax2.set_title("2D: alturas comparáveis", loc="left", fontsize=10)
estilo(ax2)
salvar(fig, "graf_eng_3d_barras.png")

# duplo eixo vs indexado
anos = np.arange(2019, 2025)
a = np.array([10, 12, 14, 16, 18, 20])
b = np.array([100, 95, 90, 85, 80, 75])
fig, eixos = plt.subplots(1, 2, figsize=(10.0, 4.4), dpi=160)
eixos[0].plot(anos, a, color=AZUL, marker="o")
eixos[0].set_ylabel("Série A", color=AZUL)
axb = eixos[0].twinx()
axb.plot(anos, b, color=TERRA, marker="o")
axb.set_ylabel("Série B", color=TERRA)
eixos[0].set_title("Duplo eixo: escalas arbitrárias", loc="left", fontsize=11)
eixos[0].tick_params(axis="y", colors=AZUL)
axb.tick_params(axis="y", colors=TERRA)
estilo(eixos[0])
eixos[1].plot(anos, a / a[0] * 100, color=AZUL, marker="o", label="A (base 100)")
eixos[1].plot(anos, b / b[0] * 100, color=TERRA, marker="o", label="B (base 100)")
eixos[1].set_ylabel("Índice (2019 = 100)")
eixos[1].set_title("Eixo único: comparação honesta", loc="left", fontsize=11)
eixos[1].legend(frameon=False)
estilo(eixos[1])
salvar(fig, "graf_eng_duplo_eixo.png")

# eixo truncado vs zero
valores2 = np.array([100, 102, 105, 103, 108])
fig, eixos = plt.subplots(1, 2, figsize=(10.0, 4.4), dpi=160)
eixos[0].bar(categorias[:5], valores2, color=AZUL, alpha=0.85)
eixos[0].set_ylim(95, 110)
eixos[0].set_ylabel("Valor")
eixos[0].set_title("Eixo truncado: diferenças parecem enormes", loc="left", fontsize=11)
estilo(eixos[0])
eixos[1].bar(categorias[:5], valores2, color=AZUL, alpha=0.85)
eixos[1].set_ylim(0, 120)
eixos[1].set_ylabel("Valor")
eixos[1].set_title("Zero no eixo: diferenças reais", loc="left", fontsize=11)
estilo(eixos[1])
salvar(fig, "graf_eng_eixo_truncado.png")

# rainbow vs viridis
grad = np.linspace(0, 1, 200).reshape(1, -1)
fig, eixos = plt.subplots(2, 1, figsize=(9.0, 4.0), dpi=160)
for ax, cmap, titulo in [
    (eixos[0], "jet", "jet/rainbow: ordem perceptual irregular"),
    (eixos[1], "viridis", "viridis: ordem perceptual uniforme"),
]:
    im = ax.imshow(grad, aspect="auto", cmap=cmap)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo, loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, fraction=0.08)
salvar(fig, "graf_eng_arcoiris.png")

print("Figuras de gráficos enganosos geradas em:", IMG)

