"""Gera as figuras da Parte IV (classificacao).

Uso:
    python src/gerar_figuras_classificacao.py

Saida:
    docs/img/matriz_confusao.png
    docs/img/distribuicao_probabilidades.png
    docs/img/threshold_precisao_recall.png
    docs/img/curva_roc.png
    docs/img/curva_pr.png
    docs/img/curva_calibracao.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
IMG = RAIZ / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

AZUL = "#3f5d7a"
TERRA = "#b96a4c"
VERDE = "#4a5937"
CINZA = "#6e6a59"


def estilo(ax):
    ax.grid(True, color="#d8cdb6", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor("#fbf8f2")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color("#d8cdb6")
    ax.spines["bottom"].set_color("#d8cdb6")


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(IMG / nome, bbox_inches="tight")
    plt.close(fig)


X, y = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=6,
    n_redundant=3,
    class_sep=1.3,
    flip_y=0.05,
    weights=[0.75, 0.25],
    random_state=42,
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

modelo = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42),
)
modelo.fit(X_tr, y_tr)
prob = modelo.predict_proba(X_te)[:, 1]
pred = (prob >= 0.5).astype(int)

# ------------------------------------------------- metricas
print("Acurácia          :", round(accuracy_score(y_te, pred), 3))
print("Acurácia balanceada:", round(balanced_accuracy_score(y_te, pred), 3))
print("Precisão          :", round(precision_score(y_te, pred), 3))
print("Recall            :", round(recall_score(y_te, pred), 3))
print("F1                :", round(f1_score(y_te, pred), 3))
print("MCC               :", round(matthews_corrcoef(y_te, pred), 3))
print("Kappa             :", round(cohen_kappa_score(y_te, pred), 3))
print("ROC-AUC           :", round(roc_auc_score(y_te, prob), 3))
print("PR-AUC (AP)       :", round(average_precision_score(y_te, prob), 3))
print("Log loss          :", round(log_loss(y_te, prob), 3))
print("Brier             :", round(brier_score_loss(y_te, prob), 3))
print("Matriz de confusão:\n", confusion_matrix(y_te, pred))

# ------------------------------------------------- matriz de confusao
cm = confusion_matrix(y_te, pred)
fig, ax = plt.subplots(figsize=(5.6, 4.9), dpi=160)
im = ax.imshow(cm, cmap="YlGn", vmin=0, vmax=cm.max())
for i in range(2):
    for j in range(2):
        cor = "white" if cm[i, j] > cm.max() * 0.55 else "#2b2e23"
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=18, color=cor, fontweight="bold")
ax.set_xticks([0, 1], ["Previsto negativo", "Previsto positivo"])
ax.set_yticks([0, 1], ["Real negativo", "Real positivo"])
ax.set_title("Matriz de confusão (limiar 0,5)", loc="left", fontsize=13)
ax.grid(False)
salvar(fig, "matriz_confusao.png")

# ------------------------------------------------- distribuicao das probabilidades
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
ax.hist(prob[y_te == 0], bins=18, alpha=0.65, color=AZUL, label="classe 0 (negativo)", edgecolor="white")
ax.hist(prob[y_te == 1], bins=18, alpha=0.65, color=TERRA, label="classe 1 (positivo)", edgecolor="white")
ax.axvline(0.5, color=VERDE, linestyle="--", linewidth=1.4, label="limiar 0,5")
ax.set_xlabel("Probabilidade prevista de ser positivo")
ax.set_ylabel("Número de casos")
ax.set_title("Distribuição das probabilidades: onde as classes se sobrepõem", loc="left", fontsize=12.5)
ax.legend(frameon=False)
estilo(ax)
salvar(fig, "distribuicao_probabilidades.png")

# ------------------------------------------------- threshold
limiares = np.linspace(0.05, 0.95, 91)
precisao, recall, f1 = [], [], []
for t in limiares:
    p = (prob >= t).astype(int)
    precisao.append(precision_score(y_te, p, zero_division=0))
    recall.append(recall_score(y_te, p, zero_division=0))
    f1.append(f1_score(y_te, p, zero_division=0))

fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
ax.plot(limiares, precisao, color=AZUL, linewidth=1.8, label="precisão")
ax.plot(limiares, recall, color=TERRA, linewidth=1.8, label="recall")
ax.plot(limiares, f1, color=VERDE, linewidth=2.0, label="F1")
ax.axvline(0.5, color=CINZA, linestyle=":", linewidth=1.3)
ax.set_xlabel("Limiar de probabilidade")
ax.set_ylabel("Valor da métrica")
ax.set_ylim(0, 1.05)
ax.set_title("Precisão, recall e F1 mudam com o limiar", loc="left", fontsize=13)
ax.legend(frameon=False)
estilo(ax)
salvar(fig, "threshold_precisao_recall.png")

# ------------------------------------------------- ROC
fpr, tpr, _ = roc_curve(y_te, prob)
auc = roc_auc_score(y_te, prob)
fig, ax = plt.subplots(figsize=(5.8, 5.1), dpi=160)
ax.plot(fpr, tpr, color=AZUL, linewidth=2.0, label=f"AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], color=TERRA, linestyle="--", linewidth=1.3, label="aleatório")
ax.set_xlabel("Taxa de falsos positivos")
ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
ax.set_title("Curva ROC", loc="left", fontsize=13)
ax.legend(frameon=False, loc="lower right")
estilo(ax)
salvar(fig, "curva_roc.png")

# ------------------------------------------------- PR
pr_precisao, pr_recall, _ = precision_recall_curve(y_te, prob)
ap = average_precision_score(y_te, prob)
prevalencia = y_te.mean()
fig, ax = plt.subplots(figsize=(5.8, 5.1), dpi=160)
ax.plot(pr_recall, pr_precisao, color=TERRA, linewidth=2.0, label=f"AP = {ap:.3f}")
ax.axhline(prevalencia, color=AZUL, linestyle="--", linewidth=1.3,
           label=f"baseline = {prevalencia:.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precisão")
ax.set_title("Curva precisão–recall", loc="left", fontsize=13)
ax.legend(frameon=False, loc="lower left")
estilo(ax)
salvar(fig, "curva_pr.png")

# ------------------------------------------------- calibracao
frac_pos, prob_media = calibration_curve(y_te, prob, n_bins=6, strategy="quantile")
fig, ax = plt.subplots(figsize=(5.8, 5.1), dpi=160)
ax.plot(prob_media, frac_pos, marker="o", color=VERDE, linewidth=1.8, label="modelo")
ax.plot([0, 1], [0, 1], color=TERRA, linestyle="--", linewidth=1.3, label="calibração perfeita")
ax.set_xlabel("Probabilidade prevista")
ax.set_ylabel("Fração de positivos observada")
ax.set_title("Curva de calibração", loc="left", fontsize=13)
ax.legend(frameon=False, loc="upper left")
estilo(ax)
salvar(fig, "curva_calibracao.png")

print("Figuras de classificação geradas em:", IMG)

