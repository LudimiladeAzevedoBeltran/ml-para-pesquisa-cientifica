"""Gera as figuras dos projetos guiados e imprime os números usados nas páginas.

Uso:
    python src/gerar_figuras_projetos.py

Saída:
    docs/img/projeto_quickstart_previsto.png
    docs/img/residuos_4paineis.png
    docs/img/projeto_regressao_comparacao.png
    docs/img/projeto_classificacao_threshold.png
    docs/img/projeto_classificacao_matriz.png
    docs/img/projeto_classificacao_roc.png
    docs/img/projeto_classificacao_pr.png
    docs/img/projeto_cluster_pca.png
    docs/img/projeto_silhueta.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
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


# ================================================================ regressão
df = pd.read_csv(DATA / "exemplo_ch4.csv")

X = df[["temperatura", "vento"]].to_numpy()
y = df["ch4"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
modelo = LinearRegression().fit(X_tr, y_tr)
pred = modelo.predict(X_te)

print("== quickstart ==")
print("intercepto:", round(modelo.intercept_, 2))
print("coeficientes:", dict(zip(["temperatura", "vento"], modelo.coef_.round(2))))
print("R²:", round(1 - ((y_te - pred) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum(), 3))
print("RMSE:", round(np.sqrt(((y_te - pred) ** 2).mean()), 2))

fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=160)
ax.scatter(y_te, pred, s=34, color=AZUL, alpha=0.75, edgecolor="white", linewidth=0.5)
lim = [min(y_te.min(), pred.min()) - 10, max(y_te.max(), pred.max()) + 10]
ax.plot(lim, lim, color=TERRA, linestyle="--", linewidth=1.5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("CH4 observado (ppb)")
ax.set_ylabel("CH4 previsto (ppb)")
ax.set_title("Primeiro modelo: previsto vs observado", loc="left", fontsize=13)
estilo(ax)
salvar(fig, "projeto_quickstart_previsto.png")

# ---------------------------------------------------------------- resíduos 4 painéis
X = df[["temperatura", "umidade", "vento", "pressao"]].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
modelo = LinearRegression().fit(X_tr, y_tr)
pred = modelo.predict(X_te)
res = y_te - pred
std_res = res / res.std()

fig, eixos = plt.subplots(2, 2, figsize=(10.5, 8.5), dpi=160)

ax = eixos[0, 0]
ax.scatter(pred, res, s=26, color=AZUL, alpha=0.7, edgecolor="white", linewidth=0.4)
ax.axhline(0, color=TERRA, linestyle="--", linewidth=1.2)
ax.set_xlabel("Valores ajustados")
ax.set_ylabel("Resíduos")
ax.set_title("Resíduos vs ajustados", loc="left", fontsize=11)
estilo(ax)

ax = eixos[0, 1]
stats.probplot(res, dist="norm", plot=ax)
ax.get_lines()[0].set_markerfacecolor(AZUL)
ax.get_lines()[0].set_markeredgecolor("white")
ax.get_lines()[0].set_markersize(5)
ax.get_lines()[1].set_color(TERRA)
ax.set_title("")
ax.set_title("QQ dos resíduos", loc="left", fontsize=11)
ax.set_xlabel("Quantis teóricos")
ax.set_ylabel("Quantis observados")
estilo(ax)

ax = eixos[1, 0]
ax.hist(res, bins=16, color=AZUL, alpha=0.75, edgecolor="white")
ax.axvline(0, color=TERRA, linestyle="--", linewidth=1.2)
ax.set_xlabel("Resíduos")
ax.set_ylabel("Frequência")
ax.set_title("Distribuição dos resíduos", loc="left", fontsize=11)
estilo(ax)

ax = eixos[1, 1]
ax.scatter(pred, np.sqrt(np.abs(std_res)), s=26, color=AZUL, alpha=0.7, edgecolor="white", linewidth=0.4)
ax.axhline(1, color=TERRA, linestyle="--", linewidth=1.2)
ax.set_xlabel("Valores ajustados")
ax.set_ylabel("√|resíduo padronizado|")
ax.set_title("Escala-localização", loc="left", fontsize=11)
estilo(ax)

fig.suptitle("Diagnóstico dos resíduos: procure padrão, não beleza", fontsize=13,
             color=VERDE, x=0.02, ha="left")
salvar(fig, "residuos_4paineis.png")

# ---------------------------------------------------------------- comparação de modelos
df2 = df.copy()
df2["temp2"] = (df2["temperatura"] - 25) ** 2
X_lin = df2[["temperatura", "umidade", "vento", "pressao"]].to_numpy()
X_quad = df2[["temperatura", "umidade", "vento", "pressao", "temp2"]].to_numpy()
y = df2["ch4"].to_numpy()
cv = KFold(n_splits=5, shuffle=True, random_state=42)

nomes = ["baseline (média)", "linear", "linear + temp²"]
modelos = [
    DummyRegressor(strategy="mean"),
    LinearRegression(),
    LinearRegression(),
]
matrizes = [X_lin, X_lin, X_quad]
medias, desvios = [], []
for nome, modelo, Xm in zip(nomes, modelos, matrizes):
    scores = cross_val_score(modelo, Xm, y, cv=cv, scoring="neg_root_mean_squared_error")
    medias.append(-scores.mean())
    desvios.append(scores.std())
    print(f"{nome}: RMSE {(-scores.mean()):.2f} +/- {scores.std():.2f}")

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
ax.bar(nomes, medias, yerr=desvios, color=[CINZA, AZUL, VERDE], alpha=0.85,
       capsize=5, edgecolor="white")
for i, m in enumerate(medias):
    ax.text(i, m + desvios[i] + 2, f"{m:.1f}", ha="center", fontsize=10, color=CINZA)
ax.set_ylabel("RMSE (CV 5-fold)")
ax.set_title("Comparação de modelos: o ganho precisa passar da incerteza", loc="left", fontsize=12)
estilo(ax)
salvar(fig, "projeto_regressao_comparacao.png")

# ================================================================ classificação
dfc = pd.read_csv(DATA / "exemplo_classificacao.csv")
features = ["temperatura", "umidade", "vento", "ndvi", "dist_estrada"]
X = dfc[features].to_numpy()
y = dfc["risco"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
modelo = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42),
).fit(X_tr, y_tr)
prob = modelo.predict_proba(X_te)[:, 1]

print("\n== classificação ==")
print("prevalência no teste:", round(y_te.mean(), 3))
print("ROC-AUC:", round(roc_auc_score(y_te, prob), 3))
print("PR-AUC :", round(average_precision_score(y_te, prob), 3))
print("log loss:", round(log_loss(y_te, prob), 3))
print("Brier  :", round(brier_score_loss(y_te, prob), 3))

limiares = np.linspace(0.05, 0.95, 91)
prec, rec, f1 = [], [], []
custo = []
for t in limiares:
    p = (prob >= t).astype(int)
    prec.append(precision_score(y_te, p, zero_division=0))
    rec.append(recall_score(y_te, p, zero_division=0))
    f1.append(f1_score(y_te, p, zero_division=0))
    fn = ((p == 0) & (y_te == 1)).sum()
    fp = ((p == 1) & (y_te == 0)).sum()
    custo.append(10 * fn + fp)

prec, rec, f1, custo = map(np.array, (prec, rec, f1, custo))
i_f1 = int(np.argmax(f1))
i_custo = int(np.argmin(custo))
print("melhor F1:", round(f1[i_f1], 3), "no limiar", round(limiares[i_f1], 3))
print("menor custo:", int(custo[i_custo]), "no limiar", round(limiares[i_custo], 3))
for t in [0.5, limiares[i_custo]]:
    p = (prob >= t).astype(int)
    print(f"limiar {t:.2f}: acurácia {((p == y_te).mean()):.3f} | "
          f"precisão {precision_score(y_te, p, zero_division=0):.3f} | "
          f"recall {recall_score(y_te, p, zero_division=0):.3f} | "
          f"F1 {f1_score(y_te, p, zero_division=0):.3f} | "
          f"FN {((p == 0) & (y_te == 1)).sum()} | FP {((p == 1) & (y_te == 0)).sum()}")

fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
ax.plot(limiares, prec, color=AZUL, linewidth=1.8, label="precisão")
ax.plot(limiares, rec, color=TERRA, linewidth=1.8, label="recall")
ax.plot(limiares, f1, color=VERDE, linewidth=2.0, label="F1")
ax.axvline(0.5, color=CINZA, linestyle=":", linewidth=1.3, label="0,5 (padrão)")
ax.axvline(limiares[i_custo], color=TERRA, linestyle="--", linewidth=1.3,
           label=f"custo ótimo ({limiares[i_custo]:.2f})")
ax.set_xlabel("Limiar de probabilidade")
ax.set_ylabel("Valor da métrica")
ax.set_ylim(0, 1.05)
ax.set_title("Projeto: precisão, recall e F1 por limiar", loc="left", fontsize=13)
ax.legend(frameon=False, fontsize=8.5)
estilo(ax)
salvar(fig, "projeto_classificacao_threshold.png")

limiar_escolhido = limiares[i_custo]
pred = (prob >= limiar_escolhido).astype(int)
cm = confusion_matrix(y_te, pred)

fig, ax = plt.subplots(figsize=(5.6, 4.9), dpi=160)
im = ax.imshow(cm, cmap="YlGn", vmin=0, vmax=cm.max())
for i in range(2):
    for j in range(2):
        cor = "white" if cm[i, j] > cm.max() * 0.55 else "#2b2e23"
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=18,
                color=cor, fontweight="bold")
ax.set_xticks([0, 1], ["Previsto 0", "Previsto 1"])
ax.set_yticks([0, 1], ["Real 0", "Real 1"])
ax.set_title(f"Matriz de confusão (limiar {limiar_escolhido:.2f})", loc="left", fontsize=13)
ax.grid(False)
salvar(fig, "projeto_classificacao_matriz.png")

fpr, tpr, _ = roc_curve(y_te, prob)
auc = roc_auc_score(y_te, prob)
fig, ax = plt.subplots(figsize=(5.8, 5.1), dpi=160)
ax.plot(fpr, tpr, color=AZUL, linewidth=2.0, label=f"AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], color=TERRA, linestyle="--", linewidth=1.3, label="aleatório")
ax.set_xlabel("Taxa de falsos positivos")
ax.set_ylabel("Recall")
ax.set_title("Projeto: curva ROC", loc="left", fontsize=13)
ax.legend(frameon=False, loc="lower right")
estilo(ax)
salvar(fig, "projeto_classificacao_roc.png")

pr_p, pr_r, _ = precision_recall_curve(y_te, prob)
ap = average_precision_score(y_te, prob)
fig, ax = plt.subplots(figsize=(5.8, 5.1), dpi=160)
ax.plot(pr_r, pr_p, color=TERRA, linewidth=2.0, label=f"AP = {ap:.3f}")
ax.axhline(y_te.mean(), color=AZUL, linestyle="--", linewidth=1.3,
           label=f"baseline = {y_te.mean():.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precisão")
ax.set_title("Projeto: curva precisão–recall", loc="left", fontsize=13)
ax.legend(frameon=False, loc="lower left")
estilo(ax)
salvar(fig, "projeto_classificacao_pr.png")

# ================================================================ clusterização
dfk = pd.read_csv(DATA / "exemplo_cluster.csv")
X = dfk.to_numpy()
Xs = StandardScaler().fit_transform(X)

ks = list(range(2, 9))
silhuetas = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=25, random_state=42)
    labels = km.fit_predict(Xs)
    silhuetas.append(silhouette_score(Xs, labels))

melhor_k = ks[int(np.argmax(silhuetas))]
print("\n== clusterização ==")
for k, s in zip(ks, silhuetas):
    print(f"k={k}: silhueta {s:.3f}")
print("melhor k:", melhor_k)

fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
ax.plot(ks, silhuetas, marker="o", color=AZUL, linewidth=1.8)
ax.axvline(melhor_k, color=TERRA, linestyle="--", linewidth=1.3)
ax.scatter([melhor_k], [max(silhuetas)], color=TERRA, s=60, zorder=5)
ax.set_xlabel("Número de clusters (k)")
ax.set_ylabel("Silhueta média")
ax.set_title("Escolha de k: o pico da silhueta", loc="left", fontsize=13)
estilo(ax)
salvar(fig, "projeto_silhueta.png")

km = KMeans(n_clusters=melhor_k, n_init=25, random_state=42).fit(Xs)
labels = km.labels_
pca = PCA(n_components=2, random_state=42).fit(Xs)
Z = pca.transform(Xs)

fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=160)
cores = [AZUL, TERRA, VERDE, "#a8761f", "#6e6a59", "#8a5a83"]
for c in range(melhor_k):
    ax.scatter(Z[labels == c, 0], Z[labels == c, 1], s=30,
               color=cores[c % len(cores)], alpha=0.75,
               edgecolor="white", linewidth=0.4, label=f"cluster {c}")
centroides = pca.transform(km.cluster_centers_)
ax.scatter(centroides[:, 0], centroides[:, 1], marker="X", s=180,
           color="#2b2e23", edgecolor="white", linewidth=1.0, label="centroides")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% da variância)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% da variância)")
ax.set_title(f"Clusters no plano PCA (k = {melhor_k})", loc="left", fontsize=13)
ax.legend(frameon=False, fontsize=8.5, ncol=2)
estilo(ax)
salvar(fig, "projeto_cluster_pca.png")

print("\nperfil médio por cluster (unidades originais):")
perfil = dfk.copy()
perfil["cluster"] = labels
print(perfil.groupby("cluster").mean().round(2))

print("\nFiguras dos projetos geradas em:", IMG)
