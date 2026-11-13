"""Gera as figuras da Parte IX (visualizacao de dados).

Uso:
    python src/gerar_figuras_graficos.py

Saída:
    docs/img/graf_*.png
    docs/img/plot3d_*.html
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.spatial import Delaunay
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
IMG = RAIZ / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

AZUL = "#3f5d7a"
TERRA = "#b96a4c"
VERDE = "#4a5937"
CINZA = "#6e6a59"
AREIA = "#d8cdb6"
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


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


dados = pd.read_csv(DATA / "exemplo_ch4.csv")
dados["estacao"] = pd.cut(
    dados["temperatura"],
    bins=[-np.inf, 23, 27, np.inf],
    labels=["frio", "ameno", "quente"],
)
estacoes = ["frio", "ameno", "quente"]
cores_estacao = {"frio": "#3f5d7a", "ameno": "#4a5937", "quente": "#b96a4c"}

# ================================================================ distribuição
# histograma
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
ax.hist(dados["ch4"], bins=18, color=AZUL, alpha=0.8, edgecolor="white")
ax.set_xlabel("CH4 (ppb)")
ax.set_ylabel("Frequência")
ax.set_title("Histograma: a forma da distribuição", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_histograma.png")

# KDE
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
sns.kdeplot(dados["ch4"], ax=ax, color=AZUL, linewidth=2.0, fill=True, alpha=0.18)
ax.set_xlabel("CH4 (ppb)")
ax.set_ylabel("Densidade")
ax.set_title("Densidade (KDE): uma curva suave para a forma", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_kde.png")

# ECDF
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
x = np.sort(dados["ch4"])
y = np.arange(1, len(x) + 1) / len(x)
ax.step(x, y, where="post", color=AZUL, linewidth=2.0)
ax.set_xlabel("CH4 (ppb)")
ax.set_ylabel("Proporção acumulada")
ax.set_title("ECDF: cada ponto conta a proporção até ali", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_ecdf.png")

# Box
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
dados_box = [dados.loc[dados["estacao"] == e, "ch4"].dropna() for e in estacoes]
bp = ax.boxplot(dados_box, tick_labels=estacoes, patch_artist=True, widths=0.55)
for patch, e in zip(bp["boxes"], estacoes):
    patch.set_facecolor(cores_estacao[e])
    patch.set_alpha(0.75)
    patch.set_edgecolor("white")
for mediana in bp["medians"]:
    mediana.set_color("#2b2e23")
    mediana.set_linewidth(1.5)
ax.set_xlabel("Estação (corte por temperatura)")
ax.set_ylabel("CH4 (ppb)")
ax.set_title("Boxplot: mediana, quartis e outliers", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_box.png")

# Violino
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
vp = ax.violinplot(dados_box, showmedians=True, widths=0.75)
for corpo, e in zip(vp["bodies"], estacoes):
    corpo.set_facecolor(cores_estacao[e])
    corpo.set_alpha(0.55)
    corpo.set_edgecolor("white")
for parte in ("cmedians", "cmaxes", "cmins", "cbars"):
    if parte in vp:
        vp[parte].set_color("#2b2e23")
ax.set_xticks([1, 2, 3], estacoes)
ax.set_xlabel("Estação")
ax.set_ylabel("CH4 (ppb)")
ax.set_title("Violino: a forma completa de cada grupo", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_violino.png")

# Raincloud
fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
for i, e in enumerate(estacoes, start=1):
    vals = dados.loc[dados["estacao"] == e, "ch4"].dropna().to_numpy()
    kde = gaussian_kde(vals)
    ys = np.linspace(vals.min(), vals.max(), 200)
    dens = kde(ys)
    dens = dens / dens.max() * 0.35
    ax.fill_betweenx(ys, i - dens, i, color=cores_estacao[e], alpha=0.55, linewidth=0)
    ax.boxplot([vals], positions=[i + 0.18], widths=0.12, vert=True,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=cores_estacao[e], edgecolor="white"),
               medianprops=dict(color="#2b2e23"))
    jitter = np.random.default_rng(42).normal(0, 0.045, len(vals))
    ax.scatter(np.full(len(vals), i + 0.38) + jitter, vals, s=10,
               color=cores_estacao[e], alpha=0.55, edgecolor="white", linewidth=0.2)
ax.set_xticks([1, 2, 3], estacoes)
ax.set_xlabel("Estação")
ax.set_ylabel("CH4 (ppb)")
ax.set_title("Raincloud: densidade + caixa + pontos", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_raincloud.png")

# Ridgeline
fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
for i, e in enumerate(estacoes):
    vals = dados.loc[dados["estacao"] == e, "ch4"].dropna().to_numpy()
    kde = gaussian_kde(vals)
    xs = np.linspace(dados["ch4"].min() - 20, dados["ch4"].max() + 20, 300)
    dens = kde(xs)
    dens = dens / dens.max() * 0.9
    ax.fill_between(xs, i, i + dens, color=cores_estacao[e], alpha=0.65, linewidth=0)
    ax.plot(xs, i + dens, color="white", linewidth=1.0)
ax.set_yticks(range(len(estacoes)), estacoes)
ax.set_xlabel("CH4 (ppb)")
ax.set_ylabel("Estação")
ax.set_title("Ridgeline: várias distribuições empilhadas", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_ridgeline.png")

# Strip
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
rng = np.random.default_rng(42)
for i, e in enumerate(estacoes, start=1):
    vals = dados.loc[dados["estacao"] == e, "ch4"].dropna().to_numpy()
    ax.scatter(i + rng.normal(0, 0.06, len(vals)), vals, s=22,
               color=cores_estacao[e], alpha=0.65, edgecolor="white", linewidth=0.3)
ax.set_xticks([1, 2, 3], estacoes)
ax.set_xlabel("Estação")
ax.set_ylabel("CH4 (ppb)")
ax.set_title("Strip: todos os pontos, com jitter", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_strip.png")

# ================================================================ relação
# Dispersão
fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
for e in estacoes:
    sub = dados[dados["estacao"] == e]
    ax.scatter(sub["temperatura"], sub["ch4"], s=28, color=cores_estacao[e],
               alpha=0.7, edgecolor="white", linewidth=0.3, label=e)
ax.set_xlabel("Temperatura (°C)")
ax.set_ylabel("CH4 (ppb)")
ax.set_title("Dispersão: relação entre duas variáveis", loc="left", fontsize=12)
ax.legend(frameon=False, title="estação")
estilo(ax)
salvar(fig, "graf_dispersao.png")

# Hexbin
fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
hb = ax.hexbin(dados["temperatura"], dados["umidade"], gridsize=18,
               cmap="viridis", mincnt=1, linewidths=0.2)
fig.colorbar(hb, ax=ax, label="contagem")
ax.set_xlabel("Temperatura (°C)")
ax.set_ylabel("Umidade (%)")
ax.set_title("Hexbin: densidade quando há muitos pontos", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_hexbin.png")

# Densidade 2D
fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
xy = np.vstack([dados["temperatura"], dados["umidade"]])
kde = gaussian_kde(xy)
gx, gy = np.mgrid[dados["temperatura"].min():dados["temperatura"].max():100j,
                  dados["umidade"].min():dados["umidade"].max():100j]
gz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
cf = ax.contourf(gx, gy, gz, levels=14, cmap="viridis", alpha=0.9)
fig.colorbar(cf, ax=ax, label="densidade")
ax.set_xlabel("Temperatura (°C)")
ax.set_ylabel("Umidade (%)")
ax.set_title("Densidade 2D: onde a massa de dados se concentra", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_densidade2d.png")

# Bolha
fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
sc = ax.scatter(dados["temperatura"], dados["ch4"],
                s=20 + 18 * (dados["vento"] - dados["vento"].min()),
                c=dados["umidade"], cmap="viridis", alpha=0.75,
                edgecolor="white", linewidth=0.4)
fig.colorbar(sc, ax=ax, label="umidade (%)")
ax.set_xlabel("Temperatura (°C)")
ax.set_ylabel("CH4 (ppb)")
ax.set_title("Bolha: tamanho e cor adicionam duas variáveis", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_bolha.png")

# Pairplot
sns.set_theme(style="whitegrid")
g = sns.pairplot(
    dados[["temperatura", "umidade", "vento", "ch4"]],
    diag_kind="kde",
    plot_kws={"s": 18, "alpha": 0.6, "color": AZUL, "edgecolor": "white", "linewidth": 0.2},
    diag_kws={"color": AZUL, "fill": True, "alpha": 0.2},
)
g.figure.set_size_inches(8.0, 7.2)
g.figure.suptitle("Scatter matrix: visão geral de várias relações", y=1.02, fontsize=12)
g.savefig(IMG / "graf_pairplot.png", bbox_inches="tight", dpi=160)
plt.close(g.figure)

# Correlação
fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=160)
corr = dados[["temperatura", "umidade", "vento", "pressao", "ch4"]].corr()
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=9, color="white" if abs(corr.iloc[i, j]) > 0.55 else "#2b2e23")
fig.colorbar(im, ax=ax, label="correlação")
ax.set_title("Heatmap de correlação", loc="left", fontsize=12)
ax.grid(False)
salvar(fig, "graf_correlacao.png")

# Contorno
fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
cs = ax.contour(gx, gy, gz, levels=10, cmap="viridis", linewidths=1.4)
ax.clabel(cs, inline=True, fontsize=7)
ax.set_xlabel("Temperatura (°C)")
ax.set_ylabel("Umidade (%)")
ax.set_title("Contorno: linhas de mesma densidade", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_contorno.png")

# ================================================================ cores
fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=160)
paletas = [
    ("viridis (sequencial)", plt.cm.viridis(np.linspace(0, 1, 8))),
    ("cividis (sequencial, CVD)", plt.cm.cividis(np.linspace(0, 1, 8))),
    ("Okabe-Ito (categórica)", [matplotlib.colors.to_rgba(c) for c in OKABE]),
    ("ColorBrewer Set2 (categórica)", plt.cm.Set2(np.linspace(0, 1, 8))),
]
for i, (nome, cores) in enumerate(paletas):
    for j, cor in enumerate(cores):
        ax.add_patch(Rectangle((j, -i), 0.92, 0.8, color=cor, linewidth=0))
    ax.text(-0.2, -i + 0.4, nome, ha="right", va="center", fontsize=9)
ax.set_xlim(-5.5, 8.2)
ax.set_ylim(-3.4, 1.2)
ax.axis("off")
ax.set_title("Paletas: sequencial, categórica e para daltonismo", loc="left", fontsize=12)
salvar(fig, "graf_paletas.png")

# Daltonismo
def simular_cvd(hexcor, tipo="deuteranopia"):
    rgb = np.array(matplotlib.colors.to_rgb(hexcor))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrizes = {
        "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                                  [0.280085, 0.672501, 0.047413],
                                  [-0.011820, 0.042940, 0.968881]]),
        "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                                [0.114503, 0.786281, 0.099216],
                                [-0.003882, -0.048116, 1.051998]]),
        "tritanopia": np.array([[1.255528, -0.076749, -0.178779],
                                [-0.078411, 0.930809, 0.147602],
                                [0.004733, 0.691367, 0.303900]]),
    }
    sim = np.clip(matrizes[tipo] @ linear, 0, 1)
    srgb = np.where(sim <= 0.0031308, sim * 12.92, 1.055 * sim ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0, 1)

ruim = ["#D00000", "#008000"]
boa = ["#D55E00", "#0072B2"]
fig, eixos = plt.subplots(2, 3, figsize=(9.0, 5.4), dpi=160)
for linha, (nome, paleta) in enumerate([("vermelho/verde", ruim), ("Okabe-Ito", boa)]):
    for coluna, tipo in enumerate(["normal", "deuteranopia", "protanopia"]):
        ax = eixos[linha, coluna]
        for k, cor in enumerate(paleta):
            c = cor if tipo == "normal" else simular_cvd(cor, tipo)
            ax.scatter(np.full(12, k), np.linspace(0, 1, 12), s=70, color=c, edgecolor="white")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([0, 1], ["A", "B"])
        ax.set_yticks([])
        ax.set_title(f"{nome} · {tipo}", fontsize=9)
        estilo(ax)
fig.suptitle("Simulação de daltonismo: cor não pode ser o único canal", fontsize=12,
             x=0.02, ha="left")
salvar(fig, "graf_daltonismo.png")

# ================================================================ 3D interativo
x = dados["temperatura"].to_numpy()
y = dados["umidade"].to_numpy()
z = dados["vento"].to_numpy()
w = dados["ch4"].to_numpy()

fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z, mode="markers",
    marker=dict(size=4, color=w, colorscale="Viridis", colorbar=dict(title="CH4 (ppb)"),
                opacity=0.75, line=dict(width=0)),
    hovertemplate="Temp %{x:.1f} °C<br>Umid %{y:.1f}%<br>Vento %{z:.2f} m/s<br>CH4 %{marker.color:.0f} ppb<extra></extra>",
)])
fig.update_layout(
    title="3D: três variáveis + cor",
    scene=dict(xaxis_title="Temperatura (°C)", yaxis_title="Umidade (%)", zaxis_title="Vento (m/s)"),
    margin=dict(l=0, r=0, t=40, b=0), height=560,
)
fig.write_html(IMG / "plot3d_scatter.html", include_plotlyjs="cdn", full_html=True)

# Superfície de resposta
X = dados[["temperatura", "umidade"]].to_numpy()
poly = PolynomialFeatures(degree=2, include_bias=False)
Xp = poly.fit_transform(X)
modelo = LinearRegression().fit(Xp, w)
gx2, gy2 = np.meshgrid(
    np.linspace(x.min(), x.max(), 35),
    np.linspace(y.min(), y.max(), 35),
)
Xg = poly.transform(np.c_[gx2.ravel(), gy2.ravel()])
gz2 = modelo.predict(Xg).reshape(gx2.shape)

fig = go.Figure(data=[go.Surface(
    x=gx2, y=gy2, z=gz2, colorscale="Viridis",
    colorbar=dict(title="CH4 previsto"),
    contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))),
)])
fig.update_layout(
    title="3D: superfície de resposta (CH4 ~ temperatura + umidade)",
    scene=dict(xaxis_title="Temperatura (°C)", yaxis_title="Umidade (%)", zaxis_title="CH4 (ppb)"),
    margin=dict(l=0, r=0, t=40, b=0), height=560,
)
fig.write_html(IMG / "plot3d_surface.html", include_plotlyjs="cdn", full_html=True)

# Trisurf
tri = Delaunay(np.c_[x, y])
fig = go.Figure(data=[go.Mesh3d(
    x=x, y=y, z=w, i=tri.simplices[:, 0], j=tri.simplices[:, 1], k=tri.simplices[:, 2],
    intensity=w, colorscale="Viridis", opacity=0.85,
    colorbar=dict(title="CH4 (ppb)"),
)])
fig.update_layout(
    title="3D: superfície triangulada dos dados",
    scene=dict(xaxis_title="Temperatura (°C)", yaxis_title="Umidade (%)", zaxis_title="CH4 (ppb)"),
    margin=dict(l=0, r=0, t=40, b=0), height=560,
)
fig.write_html(IMG / "plot3d_trisurf.html", include_plotlyjs="cdn", full_html=True)

# Volume (isosuperfície de uma mistura gaussiana)
gx3, gy3, gz3 = np.mgrid[-3:3:35j, -3:3:35j, -3:3:35j]
densidade = np.zeros_like(gx3)
for centro in [(-1.0, -0.5, 0.2), (1.0, 0.8, -0.4), (0.0, 0.0, 1.2)]:
    densidade += np.exp(-((gx3 - centro[0])**2 + (gy3 - centro[1])**2 + (gz3 - centro[2])**2) / 0.6)
fig = go.Figure(data=[go.Isosurface(
    x=gx3.flatten(), y=gy3.flatten(), z=gz3.flatten(), value=densidade.flatten(),
    isomin=0.25, isomax=densidade.max(), surface_count=4,
    colorscale="Viridis", opacity=0.45, caps=dict(x_show=False, y_show=False, z_show=False),
)])
fig.update_layout(
    title="3D: isosuperfície de uma densidade volumétrica",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    margin=dict(l=0, r=0, t=40, b=0), height=560,
)
fig.write_html(IMG / "plot3d_volume.html", include_plotlyjs="cdn", full_html=True)

# Linha 3D (trajetória)
t = np.linspace(0, 6 * np.pi, 240)
lx, ly, lz = np.cos(t), np.sin(t), t / (2 * np.pi)
fig = go.Figure(data=[go.Scatter3d(
    x=lx, y=ly, z=lz, mode="lines",
    line=dict(color=lz, colorscale="Viridis", width=6),
)])
fig.update_layout(
    title="3D: trajetória no espaço",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    margin=dict(l=0, r=0, t=40, b=0), height=560,
)
fig.write_html(IMG / "plot3d_linha.html", include_plotlyjs="cdn", full_html=True)

print("Figuras da galeria geradas em:", IMG)
