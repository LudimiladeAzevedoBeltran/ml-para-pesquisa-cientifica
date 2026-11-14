"""Gera figuras das fichas KNN, SVM e Naive Bayes."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

RAIZ = Path(__file__).resolve().parents[1]
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


rng = np.random.default_rng(42)
X, y = make_classification(
    n_samples=220, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.2, flip_y=0.05, random_state=42,
)
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 250),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 250),
)

# KNN
fig, eixos = plt.subplots(1, 3, figsize=(12.0, 4.2), dpi=160, sharex=True, sharey=True)
for ax, k in zip(eixos, [1, 15, 50]):
    clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.18, cmap="RdYlGn")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlGn", s=20,
               edgecolor="white", linewidth=0.3)
    ax.set_title(f"k = {k}", loc="left", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    estilo(ax)
fig.suptitle("KNN: k pequeno decora; k grande suaviza demais", fontsize=12, x=0.02, ha="left")
salvar(fig, "knn_fronteira.png")

# SVM
clf = SVC(kernel="linear", C=1.0).fit(X, y)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=160)
ax.contourf(xx, yy, Z, levels=20, cmap="RdYlGn", alpha=0.18)
ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[TERRA, CINZA, TERRA],
           linestyles=["--", "-", "--"], linewidths=1.3)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlGn", s=22,
           edgecolor="white", linewidth=0.3)
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=140, facecolors="none", edgecolors="#2b2e23", linewidths=1.2,
           label="vetores de suporte")
ax.set_title("SVM: margem máxima e vetores de suporte", loc="left", fontsize=12)
ax.legend(frameon=False, loc="lower right")
ax.set_xticks([]); ax.set_yticks([])
estilo(ax)
salvar(fig, "svm_margem.png")

# Naive Bayes posterior
Xc, yc = make_classification(
    n_samples=300, n_features=2, n_informative=2, n_redundant=0,
    class_sep=0.7, flip_y=0.15, weights=[0.6, 0.4], random_state=42,
)
X_tr, X_te, y_tr, y_te = train_test_split(Xc, yc, test_size=0.3, stratify=yc, random_state=42)
nb = GaussianNB().fit(X_tr, y_tr)
prob = nb.predict_proba(X_te)[:, 1]
ordem = np.argsort(prob)
fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=160)
cores = [TERRA if y_te[i] == 1 else AZUL for i in ordem]
ax.scatter(range(len(ordem)), prob[ordem], c=cores, s=25, alpha=0.85, edgecolor="white", linewidth=0.3)
ax.axhline(0.5, color=CINZA, linestyle="--", linewidth=1.2)
ax.set_xlabel("observações ordenadas pela probabilidade prevista")
ax.set_ylabel("P(classe 1)")
ax.set_title("Naive Bayes: probabilidades previstas (cor = classe real)",
             loc="left", fontsize=11)
estilo(ax)
salvar(fig, "nb_posterior.png")

print("Figuras de modelos 2 geradas em:", IMG)

