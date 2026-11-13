"""Gera as figuras da Parte III (calibracao).

Uso:
    python src/gerar_figuras_calibracao.py

Saida:
    docs/img/curva_validacao.png
    docs/img/curva_aprendizado.png
    docs/img/grid_vs_random.png
    docs/img/optuna_historico.png
    docs/img/optuna_importancia.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    learning_curve,
    validation_curve,
)
from sklearn.tree import DecisionTreeRegressor

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


X, y = load_diabetes(return_X_y=True)

# ------------------------------------------------- curva de validacao
profundidades = np.arange(1, 21)
cv_curva = KFold(n_splits=5, shuffle=True, random_state=42)
treino, validacao = validation_curve(
    DecisionTreeRegressor(random_state=42),
    X,
    y,
    param_name="max_depth",
    param_range=profundidades,
    cv=cv_curva,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
treino = -treino
validacao = -validacao
melhor = int(np.argmin(validacao.mean(axis=1)))

fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
ax.plot(profundidades, treino.mean(axis=1), color=AZUL, linewidth=1.8, label="treino")
ax.fill_between(
    profundidades,
    treino.mean(axis=1) - treino.std(axis=1),
    treino.mean(axis=1) + treino.std(axis=1),
    color=AZUL,
    alpha=0.12,
)
ax.plot(profundidades, validacao.mean(axis=1), color=TERRA, linewidth=1.8, label="validação")
ax.fill_between(
    profundidades,
    validacao.mean(axis=1) - validacao.std(axis=1),
    validacao.mean(axis=1) + validacao.std(axis=1),
    color=TERRA,
    alpha=0.12,
)
ax.axvline(profundidades[melhor], color=VERDE, linestyle=":", linewidth=1.4)
ax.scatter(
    [profundidades[melhor]],
    [validacao.mean(axis=1)[melhor]],
    color=VERDE,
    zorder=5,
    s=45,
)
ax.annotate(
    f"melhor max_depth = {profundidades[melhor]}",
    xy=(profundidades[melhor], validacao.mean(axis=1)[melhor]),
    xytext=(profundidades[melhor] + 2, validacao.mean(axis=1)[melhor] + 6),
    fontsize=9,
    color=VERDE,
    arrowprops=dict(arrowstyle="->", color=VERDE, lw=1),
)
ax.set_xlabel("max_depth")
ax.set_ylabel("RMSE (média de 5 folds)")
ax.set_title("Curva de validação: onde o erro de validação é mínimo", loc="left", fontsize=13)
ax.legend(frameon=False)
estilo(ax)
salvar(fig, "curva_validacao.png")

# ------------------------------------------------- curva de aprendizado
tamanhos, tr_sc, va_sc = learning_curve(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X,
    y,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    shuffle=True,
    random_state=42,
)
tr_sc = -tr_sc
va_sc = -va_sc

fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
ax.plot(tamanhos, tr_sc.mean(axis=1), color=AZUL, linewidth=1.8, marker="o", label="treino")
ax.fill_between(
    tamanhos,
    tr_sc.mean(axis=1) - tr_sc.std(axis=1),
    tr_sc.mean(axis=1) + tr_sc.std(axis=1),
    color=AZUL,
    alpha=0.12,
)
ax.plot(tamanhos, va_sc.mean(axis=1), color=TERRA, linewidth=1.8, marker="o", label="validação")
ax.fill_between(
    tamanhos,
    va_sc.mean(axis=1) - va_sc.std(axis=1),
    va_sc.mean(axis=1) + va_sc.std(axis=1),
    color=TERRA,
    alpha=0.12,
)
ax.set_xlabel("Tamanho do conjunto de treino")
ax.set_ylabel("RMSE (média de 5 folds)")
ax.set_title("Curva de aprendizado: o gap diminui com mais dados?", loc="left", fontsize=13)
ax.legend(frameon=False)
estilo(ax)
salvar(fig, "curva_aprendizado.png")

# ------------------------------------------------- grid vs random
rng = np.random.default_rng(42)
valores = np.linspace(0.05, 0.95, 6)
gx, gy = np.meshgrid(valores, valores)
rx = rng.uniform(0.05, 0.95, 36)
ry = rng.uniform(0.05, 0.95, 36)

fig, eixos = plt.subplots(1, 2, figsize=(9.2, 4.6), dpi=160, sharex=True, sharey=True)
for ax, (x_vals, y_vals, titulo) in zip(
    eixos,
    [
        (gx.ravel(), gy.ravel(), "Grid search: 6 × 6 = 36 pontos"),
        (rx, ry, "Random search: 36 pontos"),
    ],
):
    ax.scatter(x_vals, y_vals, s=34, color=AZUL, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("hiperparâmetro 1")
    ax.set_ylabel("hiperparâmetro 2")
    ax.set_title(titulo, fontsize=11, loc="left")
    estilo(ax)
fig.suptitle("Por que random search cobre melhor o espaço", fontsize=13, color=VERDE, x=0.02, ha="left")
salvar(fig, "grid_vs_random.png")

# ------------------------------------------------- Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objetivo(trial):
    modelo = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 50, 200, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20, log=True),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        random_state=42,
        n_jobs=-1,
    )
    scores = cross_val_score(
        modelo,
        X,
        y,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    return -scores.mean()


estudo = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
estudo.optimize(objetivo, n_trials=40)

tabela = estudo.trials_dataframe()
valores = tabela["value"].to_numpy()
melhor_ate = np.minimum.accumulate(valores)

fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
ax.scatter(tabela["number"], valores, s=22, color=AZUL, alpha=0.55, label="trial")
ax.step(tabela["number"], melhor_ate, where="post", color=TERRA, linewidth=1.8, label="melhor até aqui")
ax.set_xlabel("Trial")
ax.set_ylabel("RMSE (CV 3-fold)")
ax.set_title("Optuna: histórico da otimização", loc="left", fontsize=13)
ax.legend(frameon=False)
estilo(ax)
salvar(fig, "optuna_historico.png")

importancias = optuna.importance.get_param_importances(estudo)
nomes = list(importancias.keys())[::-1]
valores_imp = [importancias[n] for n in nomes]

fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
ax.barh(nomes, valores_imp, color=VERDE, alpha=0.85)
ax.set_xlabel("Importância relativa")
ax.set_title("Optuna: quais hiperparâmetros mais importam", loc="left", fontsize=13)
estilo(ax)
salvar(fig, "optuna_importancia.png")

print("Figuras de calibração geradas em:", IMG)
print("Melhor trial:", estudo.best_params, "| RMSE:", round(estudo.best_value, 2))
