"""Gera dados e figuras da leva 3 da Parte IX.

Uso:
    python src/gerar_figuras_avancadas.py

Saída:
    data/exemplo_categorias.csv
    docs/img/graf_cat_*.png
    docs/img/plot_cat_sankey.html
    docs/img/graf_multi_*.png
    docs/img/graf_inc_*.png
    docs/img/graf_diag_*.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.mosaicplot import mosaic

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
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


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


# ================================================================ categorias
rng = np.random.default_rng(42)
regioes = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
categorias = ["A", "B", "C"]
anos = [2019, 2020, 2021, 2022, 2023]
base = {"Norte": 50, "Nordeste": 45, "Centro-Oeste": 60, "Sudeste": 80, "Sul": 70}
efeito = {"A": 10, "B": 0, "C": -8}
probs = {
    "Norte": [0.50, 0.30, 0.20],
    "Nordeste": [0.40, 0.35, 0.25],
    "Centro-Oeste": [0.30, 0.40, 0.30],
    "Sudeste": [0.25, 0.35, 0.40],
    "Sul": [0.20, 0.30, 0.50],
}
linhas = []
for reg in regioes:
    for ano in anos:
        for _ in range(60):
            cat = rng.choice(categorias, p=probs[reg])
            valor = base[reg] + 2 * (ano - 2019) + efeito[cat] + rng.normal(0, 5)
            linhas.append({"regiao": reg, "categoria": cat, "ano": ano, "valor": valor})
df = pd.DataFrame(linhas)
df.to_csv(DATA / "exemplo_categorias.csv", index=False)

# barras com IC
medias = df.groupby("regiao")["valor"].mean().reindex(regioes)
erros = df.groupby("regiao")["valor"].sem().reindex(regioes) * 1.96
fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
ax.bar(regioes, medias, yerr=erros, capsize=5, color=AZUL, alpha=0.85, edgecolor="white")
ax.set_ylabel("Valor médio")
ax.set_title("Barras: média por região com IC 95%", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_cat_barras.png")

# dot/lollipop
ordem = medias.sort_values().index
fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
for i, reg in enumerate(ordem):
    ax.hlines(i, 0, medias[reg], color=AREIA, linewidth=2.5)
    ax.scatter(medias[reg], i, color=AZUL, s=60, zorder=3)
    ax.text(medias[reg] + 1, i, f"{medias[reg]:.1f}", va="center", fontsize=9, color=CINZA)
ax.set_yticks(range(len(ordem)), ordem)
ax.set_xlabel("Valor médio")
ax.set_title("Dot/lollipop: ranking sem barras pesadas", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_cat_dot.png")

# agrupadas
piv = df.groupby(["regiao", "categoria"])["valor"].mean().unstack().reindex(regioes)
x = np.arange(len(regioes))
largura = 0.26
fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=160)
for i, cat in enumerate(categorias):
    ax.bar(x + (i - 1) * largura, piv[cat], width=largura, label=cat,
           color=OKABE[i], edgecolor="white")
ax.set_xticks(x, regioes, rotation=15)
ax.set_ylabel("Valor médio")
ax.set_title("Barras agrupadas: região × categoria", loc="left", fontsize=12)
ax.legend(frameon=False, title="categoria")
estilo(ax)
salvar(fig, "graf_cat_agrupadas.png")

# empilhadas
fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=160)
base_acum = np.zeros(len(regioes))
for i, cat in enumerate(categorias):
    ax.bar(regioes, piv[cat], bottom=base_acum, label=cat,
           color=OKABE[i], edgecolor="white")
    base_acum += piv[cat].to_numpy()
ax.set_ylabel("Valor médio acumulado")
ax.set_title("Barras empilhadas: composição por região", loc="left", fontsize=12)
ax.legend(frameon=False, title="categoria")
estilo(ax)
salvar(fig, "graf_cat_empilhadas.png")

# mosaico
fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=160)
cores_cat = {c: OKABE[i] for i, c in enumerate(categorias)}


def props(key):
    if len(key) >= 2 and key[1] in cores_cat:
        return {"color": cores_cat[key[1]], "alpha": 0.8}
    return {}


mosaic(df, ["regiao", "categoria"], ax=ax, gap=0.02, properties=props)
ax.set_title("Mosaico: associação entre duas categóricas", loc="left", fontsize=12)
salvar(fig, "graf_cat_mosaico.png")

# waterfall
totais = df.groupby(["ano", "regiao"])["valor"].sum()
inicio = totais.loc[2019].sum()
fim = totais.loc[2023].sum()
deltas = (totais.loc[2023] - totais.loc[2019]).reindex(regioes)
fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
atual = inicio
ax.bar("2019", inicio, color=CINZA, alpha=0.8)
for reg, delta in deltas.items():
    cor = VERDE if delta >= 0 else TERRA
    ax.bar(reg, delta, bottom=atual, color=cor, alpha=0.85)
    ax.text(reg, atual + delta / 2, f"{delta:+.0f}", ha="center", va="center",
            fontsize=8, color="white")
    atual += delta
ax.bar("2023", fim, color=AZUL, alpha=0.85)
ax.set_ylabel("Valor total")
ax.set_title("Waterfall: como cada região contribuiu para a mudança", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_cat_waterfall.png")

# Sankey (Plotly)
fluxo = df.groupby(["regiao", "categoria"])["valor"].sum().reset_index()
labels = regioes + categorias
cores_nos = ["#4a5937", "#3f5d7a", "#a8761f", "#8a5a83", "#6e6a59"] + OKABE[:3]
src = [labels.index(r) for r in fluxo["regiao"]]
tgt = [labels.index(c) for c in fluxo["categoria"]]
fig = go.Figure(data=[go.Sankey(
    node=dict(label=labels, color=cores_nos, pad=18, thickness=18),
    link=dict(source=src, target=tgt, value=fluxo["valor"].round(1).tolist(),
              color="rgba(63,93,122,0.35)"),
)])
fig.update_layout(title="Sankey: fluxo de valor entre região e categoria",
                  margin=dict(l=0, r=0, t=40, b=0), height=520)
fig.write_html(IMG / "plot_cat_sankey.html", include_plotlyjs="cdn", full_html=True)

# ================================================================ multivariado
dados = pd.read_csv(DATA / "exemplo_ch4.csv")
dados["estacao"] = pd.cut(dados["temperatura"], bins=[-np.inf, 23, 27, np.inf],
                          labels=["frio", "ameno", "quente"])
features = ["temperatura", "umidade", "vento", "pressao", "ch4"]
X = StandardScaler().fit_transform(dados[features])
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42).fit(X)
Z = pca.transform(X)
cores_est = {"frio": AZUL, "ameno": VERDE, "quente": TERRA}

fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=160)
for est in ["frio", "ameno", "quente"]:
    mask = dados["estacao"] == est
    ax.scatter(Z[mask, 0], Z[mask, 1], s=26, color=cores_est[est], alpha=0.7,
               edgecolor="white", linewidth=0.3, label=est)
for i, nome in enumerate(features):
    ax.arrow(0, 0, pca.components_[0, i] * 2.6, pca.components_[1, i] * 2.6,
             color=CINZA, width=0.01, head_width=0.08, alpha=0.8)
    ax.text(pca.components_[0, i] * 2.9, pca.components_[1, i] * 2.9, nome,
            fontsize=8, color=CINZA)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA biplot: observações e cargas das variáveis", loc="left", fontsize=12)
ax.legend(frameon=False, title="estação")
estilo(ax)
salvar(fig, "graf_multi_pca_biplot.png")

# paralelas
fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=160)
par = dados[features + ["estacao"]].copy()
par[features] = StandardScaler().fit_transform(par[features])
pd.plotting.parallel_coordinates(
    par, "estacao",
    color=[cores_est["frio"], cores_est["ameno"], cores_est["quente"]],
    alpha=0.35, ax=ax,
)
ax.set_title("Coordenadas paralelas (variáveis padronizadas)", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_multi_parallel.png")

# radar
medias_est = dados.groupby("estacao")[features].mean()
medias_z = (medias_est - medias_est.mean()) / medias_est.std(ddof=0)
angulos = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angulos += angulos[:1]
fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=160, subplot_kw=dict(polar=True))
for est in ["frio", "ameno", "quente"]:
    valores = medias_z.loc[est].tolist()
    valores += valores[:1]
    ax.plot(angulos, valores, color=cores_est[est], linewidth=1.8, label=est)
    ax.fill(angulos, valores, color=cores_est[est], alpha=0.10)
ax.set_xticks(angulos[:-1], features, fontsize=8)
ax.set_title("Radar: compare perfis médios (use com cautela)", loc="left", fontsize=11, pad=20)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.2, 1.1))
salvar(fig, "graf_multi_radar.png")

# rede de correlação
corr = dados[features].corr()
G = nx.Graph()
for f in features:
    G.add_node(f)
for i, a in enumerate(features):
    for b in features[i + 1:]:
        r = corr.loc[a, b]
        if abs(r) >= 0.4:
            G.add_edge(a, b, weight=abs(r), sign=np.sign(r))
pos = nx.spring_layout(G, seed=42, k=1.2)
fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=160)
for u, v, d in G.edges(data=True):
    cor = AZUL if d["sign"] > 0 else TERRA
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=cor, linewidth=1 + 5 * d["weight"], alpha=0.75)
rotulos = {"temperatura": "temp", "umidade": "umid", "vento": "vento",
           "pressao": "press", "ch4": "ch4"}
for no, (x, y) in pos.items():
    ax.scatter(x, y, s=500, color=VERDE, edgecolor="white", zorder=3)
    ax.text(x, y, rotulos.get(no, no), ha="center", va="center", fontsize=8,
            color="white", zorder=4, clip_on=False)
ax.set_title("Rede de correlação: |r| ≥ 0,4", loc="left", fontsize=12)
ax.axis("off")
xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
salvar(fig, "graf_multi_rede.png")

# ================================================================ incerteza
# coeficientes com IC (forest)
Xr = dados[["temperatura", "umidade", "vento", "pressao"]]
Xr_z = StandardScaler().fit_transform(Xr)
Xr_z = sm.add_constant(Xr_z)
modelo = sm.OLS(dados["ch4"], Xr_z).fit()
nomes = list(Xr.columns)
coefs = modelo.params[1:]
ics = modelo.conf_int().iloc[1:]
fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
y = np.arange(len(nomes))
ax.errorbar(coefs, y, xerr=[coefs - ics[0], ics[1] - coefs],
            fmt="o", color=AZUL, ecolor=CINZA, capsize=4, markersize=6)
ax.axvline(0, color=TERRA, linestyle="--", linewidth=1.2)
ax.set_yticks(y, nomes)
ax.set_xlabel("Coeficiente padronizado (ppb por desvio-padrão)")
ax.set_title("Coeficientes com IC 95%: incerteza visível", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_inc_forest.png")

# Bland-Altman
X_tr, X_te, y_tr, y_te = train_test_split(Xr, dados["ch4"], test_size=0.25, random_state=42)
mod = LinearRegression().fit(X_tr, y_tr)
pred = mod.predict(X_te)
media = (y_te + pred) / 2
dif = y_te - pred
md = dif.mean()
sd = dif.std(ddof=1)
fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=160)
ax.scatter(media, dif, s=26, color=AZUL, alpha=0.65, edgecolor="white")
ax.axhline(md, color=VERDE, linewidth=1.5, label=f"média {md:.1f}")
ax.axhline(md + 1.96 * sd, color=TERRA, linestyle="--", linewidth=1.2,
           label=f"+1,96 DP ({md + 1.96 * sd:.1f})")
ax.axhline(md - 1.96 * sd, color=TERRA, linestyle="--", linewidth=1.2,
           label=f"-1,96 DP ({md - 1.96 * sd:.1f})")
ax.set_xlabel("Média dos dois métodos (ppb)")
ax.set_ylabel("Diferença (observado - previsto)")
ax.set_title("Bland–Altman: concordância entre métodos", loc="left", fontsize=12)
ax.legend(frameon=False)
estilo(ax)
salvar(fig, "graf_inc_bland_altman.png")

# QQ
fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=160)
stats.probplot(dif, dist="norm", plot=ax)
ax.get_lines()[0].set_markerfacecolor(AZUL)
ax.get_lines()[0].set_markeredgecolor("white")
ax.get_lines()[0].set_markersize(5)
ax.get_lines()[1].set_color(TERRA)
ax.set_title("")
ax.set_title("QQ dos resíduos: as caudas seguem a normal?", loc="left", fontsize=12)
ax.set_xlabel("Quantis teóricos")
ax.set_ylabel("Quantis observados")
estilo(ax)
salvar(fig, "graf_inc_qq.png")

# boxen
fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
sns.boxenplot(data=dados, x="estacao", y="ch4", hue="estacao",
              palette=[cores_est["frio"], cores_est["ameno"], cores_est["quente"]],
              legend=False, ax=ax)
ax.set_xlabel("Estação")
ax.set_ylabel("CH4 (ppb)")
ax.set_title("Boxen: caudas e quantis para n maior", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_inc_boxen.png")

# ================================================================ diagnóstico de modelos
# lift / gain
Xc, yc = make_classification(
    n_samples=1200, n_features=12, n_informative=6, n_redundant=3,
    class_sep=1.3, flip_y=0.05, weights=[0.75, 0.25], random_state=42,
)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.3, stratify=yc, random_state=42)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42)).fit(Xc_tr, yc_tr)
prob = clf.predict_proba(Xc_te)[:, 1]
ordem = np.argsort(-prob)
y_sorted = yc_te[ordem]
cum_pos = np.cumsum(y_sorted)
cum_n = np.arange(1, len(y_sorted) + 1)
ganho = cum_pos / cum_pos[-1]
frac = cum_n / len(y_sorted)
lift = ganho / frac

fig, eixos = plt.subplots(1, 2, figsize=(10.0, 4.4), dpi=160)
eixos[0].plot(frac, ganho, color=AZUL, linewidth=2)
eixos[0].plot([0, 1], [0, 1], color=TERRA, linestyle="--")
eixos[0].set_xlabel("Fração da amostra")
eixos[0].set_ylabel("Fração dos positivos capturados")
eixos[0].set_title("Curva de ganho acumulado", loc="left", fontsize=11)
estilo(eixos[0])
eixos[1].plot(frac, lift, color=VERDE, linewidth=2)
eixos[1].axhline(1, color=TERRA, linestyle="--")
eixos[1].set_xlabel("Fração da amostra")
eixos[1].set_ylabel("Lift")
eixos[1].set_title("Curva de lift", loc="left", fontsize=11)
estilo(eixos[1])
salvar(fig, "graf_diag_lift.png")

# permutation importance
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
imp = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=42)
ordem = np.argsort(imp.importances_mean)
fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=160)
ax.barh(np.array(Xr.columns)[ordem], imp.importances_mean[ordem],
        xerr=imp.importances_std[ordem], color=VERDE, alpha=0.85, capsize=4)
ax.set_xlabel("Queda no RMSE ao permutar")
ax.set_title("Importância por permutação", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "graf_diag_importancia.png")

# SHAP
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_te)
shap.summary_plot(shap_values, X_te, show=False)
fig = plt.gcf()
fig.set_size_inches(8.0, 4.8)
fig.savefig(IMG / "graf_diag_shap.png", bbox_inches="tight", dpi=160)
plt.close(fig)

print("Dados e figuras da leva 3 gerados em:", DATA, "e", IMG)



