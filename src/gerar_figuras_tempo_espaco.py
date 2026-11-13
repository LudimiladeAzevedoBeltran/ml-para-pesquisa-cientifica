"""Gera dados e figuras de tempo e espaço (Parte IX).

Uso:
    python src/gerar_figuras_tempo_espaco.py

Saída:
    data/exemplo_serie.csv
    data/exemplo_espacial.csv
    docs/img/graf_serie_*.png
    docs/img/graf_mapa_*.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
from statsmodels.tsa.stattools import acf, pacf

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
IMG = RAIZ / "docs" / "img"
DATA.mkdir(parents=True, exist_ok=True)
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


# ================================================================ dados de tempo
rng = np.random.default_rng(42)
datas = pd.date_range("2015-01-01", "2024-12-01", freq="MS")
n = len(datas)
t = np.arange(n)
mes = datas.month.to_numpy()
serie = (
    1800
    + 0.9 * t
    + 12 * np.sin(2 * np.pi * (mes - 3) / 12)
    + rng.normal(0, 6, n)
)
temperatura = 25 + 4 * np.sin(2 * np.pi * (mes - 6) / 12) + rng.normal(0, 1.5, n)
vento = 3 + 0.6 * np.sin(2 * np.pi * (mes - 1) / 12) + rng.normal(0, 0.5, n)
df = pd.DataFrame({"data": datas, "ch4": serie, "temperatura": temperatura, "vento": vento})
df.to_csv(DATA / "exemplo_serie.csv", index=False)

# linha com média móvel
fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
ax.plot(df["data"], df["ch4"], color=AZUL, linewidth=1.2, alpha=0.75, label="mensal")
rol = df["ch4"].rolling(12, center=True).mean()
ax.plot(df["data"], rol, color=TERRA, linewidth=2.2, label="média móvel 12 meses")
ax.set_xlabel("Data")
ax.set_ylabel("CH4 (ppb)")
ax.set_title("Linha: tendência e sazonalidade ao longo do tempo", loc="left", fontsize=12)
ax.legend(frameon=False)
estilo(ax)
salvar(fig, "graf_serie_linha.png")

# área
fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
ax.fill_between(df["data"], df["ch4"], df["ch4"].min() - 5, color=AZUL, alpha=0.35)
ax.plot(df["data"], df["ch4"], color=AZUL, linewidth=1.3)
ax.set_xlabel("Data")
ax.set_ylabel("CH4 (ppb)")
ax.set_title("Área: magnitude acumulada no tempo", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_serie_area.png")

# sazonal: uma linha por ano
fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
df["ano"] = df["data"].dt.year
df["mes"] = df["data"].dt.month
cores = plt.cm.viridis(np.linspace(0.1, 0.9, df["ano"].nunique()))
for cor, (ano, sub) in zip(cores, df.groupby("ano")):
    ax.plot(sub["mes"], sub["ch4"], color=cor, linewidth=1.3, alpha=0.85, label=str(ano))
ax.set_xticks(range(1, 13))
ax.set_xlabel("Mês")
ax.set_ylabel("CH4 (ppb)")
ax.set_title("Sazonal: um ano por linha", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=7, ncol=2, title="ano")
estilo(ax)
salvar(fig, "graf_serie_sazonal.png")

# lag plot
fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=160)
ax.scatter(df["ch4"].shift(1), df["ch4"], s=26, color=AZUL, alpha=0.65,
           edgecolor="white", linewidth=0.3)
lim = [df["ch4"].min() - 5, df["ch4"].max() + 5]
ax.plot(lim, lim, color=TERRA, linestyle="--", linewidth=1.3)
ax.set_xlabel("CH4 no mês anterior (ppb)")
ax.set_ylabel("CH4 no mês atual (ppb)")
ax.set_title("Lag plot: o valor de hoje contra o de ontem", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_serie_lag.png")

# ACF / PACF
acf_vals = acf(df["ch4"], nlags=36, fft=True)
pacf_vals = pacf(df["ch4"], nlags=36, method="ywm")
conf = 1.96 / np.sqrt(len(df))

fig, eixos = plt.subplots(1, 2, figsize=(10.2, 4.4), dpi=160)
for ax, vals, titulo in zip(eixos, [acf_vals, pacf_vals], ["ACF", "PACF"]):
    ax.vlines(range(len(vals)), 0, vals, color=AZUL, linewidth=1.2)
    ax.scatter(range(len(vals)), vals, color=AZUL, s=18, zorder=3)
    ax.axhline(0, color=AREIA, linewidth=0.8)
    ax.axhline(conf, color=TERRA, linestyle="--", linewidth=0.9)
    ax.axhline(-conf, color=TERRA, linestyle="--", linewidth=0.9)
    ax.set_xlabel("Lag (meses)")
    ax.set_ylabel("Correlação")
    ax.set_title(titulo, loc="left", fontsize=12)
    estilo(ax)
fig.suptitle("Autocorrelação: memória e sazonalidade", fontsize=12, x=0.02, ha="left")
salvar(fig, "graf_serie_acf.png")

# calendário
cal = df.pivot_table(index="ano", columns="mes", values="ch4")
fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
im = ax.imshow(cal, aspect="auto", cmap="viridis")
ax.set_xticks(range(12), ["jan", "fev", "mar", "abr", "mai", "jun",
                          "jul", "ago", "set", "out", "nov", "dez"])
ax.set_yticks(range(len(cal.index)), cal.index)
ax.set_xlabel("Mês")
ax.set_ylabel("Ano")
ax.set_title("Calendário: o mesmo mês em anos diferentes", loc="left", fontsize=12)
fig.colorbar(im, ax=ax, label="CH4 (ppb)")
ax.grid(False)
salvar(fig, "graf_serie_calendario.png")

# small multiples por ano
anos = sorted(df["ano"].unique())
fig, eixos = plt.subplots(2, 5, figsize=(12.0, 5.2), dpi=160, sharey=True)
for ax, ano in zip(eixos.ravel(), anos):
    sub = df[df["ano"] == ano]
    ax.plot(sub["mes"], sub["ch4"], color=AZUL, linewidth=1.5, marker="o", markersize=2.5)
    ax.set_title(str(ano), fontsize=9, loc="left")
    ax.set_xticks([1, 4, 7, 10])
    ax.set_ylim(df["ch4"].min() - 5, df["ch4"].max() + 5)
    estilo(ax)
fig.suptitle("Small multiples: um painel por ano", fontsize=12, x=0.02, ha="left")
salvar(fig, "graf_serie_small_multiples.png")

# dumbbell antes/depois
primeiro = df[df["ano"] == anos[0]].groupby("mes")["ch4"].mean()
ultimo = df[df["ano"] == anos[-1]].groupby("mes")["ch4"].mean()
fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
for m in range(1, 13):
    ax.plot([primeiro[m], ultimo[m]], [m, m], color=CINZA, linewidth=1.5, alpha=0.6)
    ax.scatter(primeiro[m], m, color=AZUL, s=35, zorder=3)
    ax.scatter(ultimo[m], m, color=TERRA, s=35, zorder=3)
ax.set_yticks(range(1, 13), ["jan", "fev", "mar", "abr", "mai", "jun",
                              "jul", "ago", "set", "out", "nov", "dez"])
ax.set_xlabel("CH4 médio (ppb)")
ax.set_ylabel("Mês")
ax.set_title(f"Dumbbell: {anos[0]} vs {anos[-1]}", loc="left", fontsize=12)
ax.scatter([], [], color=AZUL, label=str(anos[0]))
ax.scatter([], [], color=TERRA, label=str(anos[-1]))
ax.legend(frameon=False)
estilo(ax)
salvar(fig, "graf_serie_dumbbell.png")

# ================================================================ dados espaciais
rng = np.random.default_rng(7)
n = 500
lon = rng.uniform(-60, -45, n)
lat = rng.uniform(-25, -5, n)
campo = (
    1800
    + 35 * np.exp(-((lon + 52) ** 2 + (lat + 15) ** 2) / 40)
    + 12 * np.sin(lon / 4)
    + 8 * np.cos(lat / 3)
    + rng.normal(0, 8, n)
)
esp = pd.DataFrame({"lon": lon, "lat": lat, "ch4": campo})
esp.to_csv(DATA / "exemplo_espacial.csv", index=False)

# bolha
fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=160)
sc = ax.scatter(esp["lon"], esp["lat"], c=esp["ch4"], s=18 + 0.5 * (esp["ch4"] - esp["ch4"].min()),
                cmap="viridis", alpha=0.75, edgecolor="white", linewidth=0.3)
fig.colorbar(sc, ax=ax, label="CH4 (ppb)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Mapa de bolhas: posição, cor e tamanho", loc="left", fontsize=12)
ax.set_aspect("equal")
estilo(ax)
salvar(fig, "graf_mapa_bolha.png")

# grade para raster/contorno
gx = np.linspace(esp["lon"].min(), esp["lon"].max(), 80)
gy = np.linspace(esp["lat"].min(), esp["lat"].max(), 80)
GX, GY = np.meshgrid(gx, gy)
GZ = griddata((esp["lon"], esp["lat"]), esp["ch4"], (GX, GY), method="cubic")

fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=160)
mesh = ax.pcolormesh(GX, GY, GZ, cmap="viridis", shading="auto")
fig.colorbar(mesh, ax=ax, label="CH4 (ppb)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Raster: cada célula é um valor interpolado", loc="left", fontsize=12)
ax.set_aspect("equal")
salvar(fig, "graf_mapa_raster.png")

fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=160)
cf = ax.contourf(GX, GY, GZ, levels=14, cmap="viridis")
cs = ax.contour(GX, GY, GZ, levels=8, colors="white", linewidths=0.8)
ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
fig.colorbar(cf, ax=ax, label="CH4 (ppb)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Contorno: linhas de mesmo valor", loc="left", fontsize=12)
ax.set_aspect("equal")
salvar(fig, "graf_mapa_contorno.png")

fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=160)
hb = ax.hexbin(esp["lon"], esp["lat"], C=esp["ch4"], reduce_C_function=np.mean,
               gridsize=16, cmap="viridis", mincnt=1, linewidths=0.2)
fig.colorbar(hb, ax=ax, label="CH4 médio (ppb)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Hexbin map: média por célula", loc="left", fontsize=12)
ax.set_aspect("equal")
salvar(fig, "graf_mapa_hexbin.png")

# small multiples espacial por estação
esp["estacao"] = pd.cut(esp["lat"], bins=[-np.inf, -18, -12, np.inf],
                        labels=["sul", "central", "norte"])
fig, eixos = plt.subplots(1, 3, figsize=(12.0, 4.6), dpi=160, sharex=True, sharey=True)
vmin, vmax = esp["ch4"].min(), esp["ch4"].max()
for ax, est in zip(eixos, ["sul", "central", "norte"]):
    sub = esp[esp["estacao"] == est]
    sc = ax.scatter(sub["lon"], sub["lat"], c=sub["ch4"], s=20, cmap="viridis",
                    vmin=vmin, vmax=vmax, alpha=0.8, edgecolor="white", linewidth=0.2)
    ax.set_title(est, fontsize=10, loc="left")
    ax.set_xlabel("Longitude")
    ax.set_aspect("equal")
    estilo(ax)
eixos[0].set_ylabel("Latitude")
fig.subplots_adjust(right=0.88, top=0.85)
cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.64])
fig.colorbar(sc, cax=cbar_ax, label="CH4 (ppb)")
fig.suptitle("Small multiples espaciais: Sul, Central e Norte", fontsize=12, x=0.02, ha="left")
fig.savefig(IMG / "graf_mapa_small_multiples.png", bbox_inches="tight")
plt.close(fig)

# coroplético em grade
nx, ny = 7, 6
xedges = np.linspace(esp["lon"].min(), esp["lon"].max(), nx + 1)
yedges = np.linspace(esp["lat"].min(), esp["lat"].max(), ny + 1)
ix = np.clip(np.digitize(esp["lon"], xedges) - 1, 0, nx - 1)
iy = np.clip(np.digitize(esp["lat"], yedges) - 1, 0, ny - 1)
grade = np.full((ny, nx), np.nan)
for i in range(nx):
    for j in range(ny):
        vals = esp.loc[(ix == i) & (iy == j), "ch4"]
        if len(vals):
            grade[j, i] = vals.mean()
fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=160)
mesh = ax.pcolormesh(xedges, yedges, grade, cmap="viridis", edgecolors="white", linewidth=0.8)
fig.colorbar(mesh, ax=ax, label="CH4 médio (ppb)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Coroplético em grade: média por célula", loc="left", fontsize=12)
ax.set_aspect("equal")
salvar(fig, "graf_mapa_coropletico.png")

print("Dados e figuras de tempo/espaço gerados em:", DATA, "e", IMG)

