"""Gera os dados sintéticos dos projetos guiados.

Uso:
    python src/gerar_dados_projetos.py

Saída:
    data/exemplo_ch4.csv
    data/exemplo_classificacao.csv
    data/exemplo_cluster.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
DATA.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- regressão
rng = np.random.default_rng(42)
n = 200
temperatura = rng.normal(25, 4, n)
umidade = rng.normal(60, 15, n)
vento = rng.normal(3, 1.2, n)
pressao = rng.normal(1013, 8, n)
ruido = rng.normal(0, 25, n)
ch4 = (
    1800
    + 6 * (temperatura - 25)
    - 1.5 * (umidade - 60)
    - 20 * (vento - 3)
    + 0.8 * (pressao - 1013)
    + 0.08 * (temperatura - 25) ** 2
    + ruido
)
df_ch4 = pd.DataFrame(
    {
        "temperatura": temperatura.round(2),
        "umidade": umidade.round(2),
        "vento": vento.round(2),
        "pressao": pressao.round(2),
        "ch4": ch4.round(2),
    }
)
df_ch4.to_csv(DATA / "exemplo_ch4.csv", index=False)

# ---------------------------------------------------------------- classificação
rng = np.random.default_rng(7)
n = 800
temperatura = rng.normal(25, 4, n)
umidade = rng.normal(60, 15, n)
vento = rng.normal(3, 1.2, n)
ndvi = rng.normal(0.6, 0.15, n)
dist_estrada = rng.gamma(2, 5, n)


def z(x):
    return (x - x.mean()) / x.std()


logit = (
    -2.5
    + 1.2 * z(temperatura)
    + 0.8 * z(umidade)
    - 0.6 * z(vento)
    - 1.0 * z(ndvi)
    + 0.5 * z(dist_estrada)
)
p = 1 / (1 + np.exp(-logit))
risco = rng.binomial(1, p)

df_cls = pd.DataFrame(
    {
        "temperatura": temperatura.round(2),
        "umidade": umidade.round(2),
        "vento": vento.round(2),
        "ndvi": ndvi.round(3),
        "dist_estrada": dist_estrada.round(2),
        "risco": risco,
    }
)
df_cls.to_csv(DATA / "exemplo_classificacao.csv", index=False)

# ---------------------------------------------------------------- clusterização
rng = np.random.default_rng(11)
centros = np.array(
    [
        [28, 70, 2.0, 0.70],
        [22, 50, 5.0, 0.40],
        [30, 40, 1.5, 0.30],
        [18, 80, 3.5, 0.80],
    ]
)
desvios = np.array([2.5, 8.0, 0.8, 0.08])
partes = []
for centro in centros:
    partes.append(rng.normal(centro, desvios, size=(75, 4)))
cluster = np.vstack(partes)
df_cluster = pd.DataFrame(
    cluster,
    columns=["temperatura", "umidade", "vento", "ndvi"],
).round(3)
df_cluster.to_csv(DATA / "exemplo_cluster.csv", index=False)

print("Dados gerados em:", DATA)
print("CH4:", df_ch4.shape, "| classificação:", df_cls.shape,
      "| positivos:", round(df_cls["risco"].mean(), 3),
      "| cluster:", df_cluster.shape)
