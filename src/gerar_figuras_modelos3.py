"""Gera figuras das fichas Baseline, MLP e DBSCAN/hierárquico."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
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


# ================================================================ baseline
df = pd.read_csv(DATA / "exemplo_ch4.csv")
X = df[["temperatura"]].to_numpy()
y = df["ch4"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
media = y_tr.mean()

fig, eixos = plt.subplots(1, 2, figsize=(10.0, 4.4), dpi=160)
ax = eixos[0]
ax.scatter(X_te, y_te, s=24, color=AZUL, alpha=0.7)
ax.axhline(media, color=TERRA, linewidth=2, label=f"média do treino = {media:.0f}")
ax.set_xlabel("Temperatura (°C)")
ax.set_ylabel("CH4 (ppb)")
ax.set_title("Baseline de regressão: prever a média", loc="left", fontsize=11)
ax.legend(frameon=False)
estilo(ax)

dfc = pd.read_csv(DATA / "exemplo_classificacao.csv")
contagem = dfc["risco"].value_counts(normalize=True).sort_index()
ax = eixos[1]
ax.bar(["classe 0", "classe 1"], contagem, color=[AZUL, TERRA], alpha=0.85)
for i, v in enumerate(contagem):
    ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9, color=CINZA)
ax.set_ylim(0, 1)
ax.set_ylabel("proporção")
ax.set_title("Baseline de classificação: prever a majoritária", loc="left", fontsize=11)
estilo(ax)
salvar(fig, "baseline_regressao_classificacao.png")

# ================================================================ MLP diagrama
fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
camadas = [4, 6, 3, 1]
xs = [0.15, 0.42, 0.69, 0.92]
for x, n in zip(xs, camadas):
    ys = np.linspace(0.2, 0.8, n)
    for yy in ys:
        ax.add_patch(Circle((x, yy), 0.035, color=VERDE, alpha=0.85, zorder=3))
for i in range(len(camadas) - 1):
    for y1 in np.linspace(0.2, 0.8, camadas[i]):
        for y2 in np.linspace(0.2, 0.8, camadas[i + 1]):
            ax.plot([xs[i], xs[i + 1]], [y1, y2], color=AREIA, linewidth=0.8, zorder=1)
ax.set_xlim(0.05, 1.0)
ax.set_ylim(0.1, 0.9)
ax.axis("off")
ax.set_title("MLP: camadas de transformações lineares + ativação", loc="left", fontsize=12)
salvar(fig, "mlp_diagrama.png")

# MLP treino
Xm = StandardScaler().fit_transform(df[["temperatura", "umidade", "vento", "pressao"]])
ym = df["ch4"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(Xm, ym, test_size=0.25, random_state=42)
scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
y_tr_s = scaler_y.transform(y_tr.reshape(-1, 1)).ravel()
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
    learning_rate_init=1e-3, max_iter=500, early_stopping=True,
    n_iter_no_change=20, random_state=42,
).fit(X_tr, y_tr_s)

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
ax.plot(mlp.loss_curve_, color=AZUL, linewidth=2)
ax.set_xlabel("iteração")
ax.set_ylabel("perda de treino")
ax.set_title("MLP: a perda deve cair e estabilizar", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "mlp_treino.png")

# ================================================================ DBSCAN
Xmoon, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(Xmoon)
fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=160)
for lab in sorted(set(labels)):
    mask = labels == lab
    if lab == -1:
        ax.scatter(Xmoon[mask, 0], Xmoon[mask, 1], s=24, color="#b0a99a",
                   label="ruído", edgecolor="white", linewidth=0.3)
    else:
        ax.scatter(Xmoon[mask, 0], Xmoon[mask, 1], s=24, color=OKABE[lab % len(OKABE)],
                   label=f"cluster {lab}", edgecolor="white", linewidth=0.3)
ax.set_title("DBSCAN: formas arbitrárias e ruído", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=8)
estilo(ax)
salvar(fig, "dbscan_clusters.png")

# Dendrograma
dados = pd.read_csv(DATA / "exemplo_cluster.csv").sample(30, random_state=42)
Z = linkage(StandardScaler().fit_transform(dados), method="ward")
fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=160)
dendrogram(Z, ax=ax, color_threshold=8, above_threshold_color=CINZA)
ax.axhline(8, color=TERRA, linestyle="--", linewidth=1.3)
ax.set_ylabel("distância")
ax.set_title("Hierárquico: dendrograma e linha de corte", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "hierarquico_dendrograma.png")

print("Figuras de modelos 3 geradas em:", IMG)

