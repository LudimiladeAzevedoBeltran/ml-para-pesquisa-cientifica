# Machine Learning para Pesquisa Científica

Material aberto para aprender **métodos de machine learning aplicados à pesquisa científica**:
escolher o modelo, calibrar com critério, avaliar com a métrica adequada e interpretar
resultados com base em figuras diagnósticas.

> **Idioma:** português (versão em inglês planejada).
> **Status:** Fase 8 — fundamentos, modelos, calibração, avaliação, interpretação, projetos, reprodutibilidade, extras e visualização de dados (distribuição, relação, tempo, espaço, categorias, multivariado, incerteza, 3D, cores, diagnóstico de modelos e gráficos enganosos).

[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-4a5937.svg)](LICENSE)
[![Texto: CC BY 4.0](https://img.shields.io/badge/texto-CC%20BY%204.0-b96c4c.svg)](LICENSE-docs)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-4a5937.svg)](https://www.python.org/)
[![R 4.x](https://img.shields.io/badge/R-4.x-4a5937.svg)](https://www.r-project.org/)
[![Site: Quarto](https://img.shields.io/badge/site-Quarto-b96c4c.svg)](docs/)

---

## O que você encontra aqui

| Parte | Conteúdo |
|---|---|
| Fundamentos | o que é ML, como aprende, tipos de tarefa, pipeline, armadilhas |
| Modelos | mapa de decisão + fichas padronizadas: quando usar, hiperparâmetros, bordas, calibração |
| Calibração | manual, grid/random search, Optuna e boas práticas |
| Avaliação | R², R² ajustado, RMSE, MAE, MAPE, acurácia, precisão, recall, F1, ROC/PR |
| Interpretação | laboratório de figuras com "semáforo": o que olhar e o que é erro |
| Projetos | do primeiro modelo em 30 min a projetos completos |
| Reprodutibilidade | sementes, versões, Git/GitHub e checklists |
| Visualização | distribuição, relação, 3D interativo e acessibilidade de cores |

## Estrutura

```
12_Projeto_Machine-learning_site/
├─ docs/                  # site (Quarto)
│  ├─ _quarto.yml
│  ├─ index.qmd
│  ├─ styles.css
│  ├─ img/
│  └─ pt/
│     ├─ 00-como-usar.qmd
│     ├─ 01-fundamentos/
│     ├─ 02-modelos/
│     ├─ 03-calibracao/
│     ├─ 04-avaliacao/
│     ├─ 05-interpretacao/
│     ├─ 06-projetos/
│     ├─ 07-reprodutibilidade/
│     ├─ 08-extras/
│     └─ 09-graficos/
├─ data/                  # exemplo_ch4.csv, exemplo_classificacao.csv, exemplo_cluster.csv
├─ notebooks/             # cadernos executáveis (em construção)
├─ src/                   # geração de dados e figuras
├─ requirements.txt
├─ 00_PLANEJAMENTO_SITE_ML.md
├─ CITATION.cff
├─ LICENSE                # código: MIT
└─ LICENSE-docs           # texto/figuras: CC BY 4.0
```

## Rodar o site localmente

Pré-requisitos: [Quarto](https://quarto.org) e Python 3.10+.

```bash
pip install -r requirements.txt
cd docs
quarto render
```

O site sai em `docs/_site`. Para pré-visualizar com recarregamento automático:

```bash
cd docs
quarto preview
```

> **Se o `quarto render` falhar com `os error 303` ("file is being deleted")**: é o
> Google Drive ou o antivírus segurando as pastas temporárias do Quarto. Feche o preview,
> pause a sincronização por um minuto ou renderize a partir de uma cópia local. O conteúdo
> não é afetado.

## Código em Python e R

Cada página de código tem duas abas: **Python** (referência executável, com
`scikit-learn` e `optuna`) e **R** (`base R` + `ggplot2` + `rpart`, com `glmnet`,
`randomForest` e `gbm` quando a ficha pede). Na v1, o R não exige `tidymodels`; quando um
exemplo precisar de pacote extra, isso é avisado na própria página.

## Publicação (GitHub Pages)

O workflow `.github/workflows/publish.yml` renderiza `docs/` e publica `docs/_site` no
GitHub Pages a cada push na branch `main`.

O repositório já está publicado em
`https://github.com/LudimiladeAzevedoBeltran/ml-para-pesquisa-cientifica`. Para publicar:

1. Em **Settings → Pages → Build and deployment**, escolha **GitHub Actions**.
2. Faça um push na `main` ou rode o workflow manualmente em **Actions**.

O site fica em `https://ludimiladeazevedobeltran.github.io/ml-para-pesquisa-cientifica/`.
Se a branch padrão não for `main`, ajuste o gatilho do workflow.

## Escopo da v1

Entram: fundamentos, modelos clássicos, calibração, avaliação (regressão, classificação e
clusterização), interpretação, projetos guiados, reprodutibilidade, extras e visualização
de dados (distribuição, relação, 3D e acessibilidade).

Ficam para a v2: versão em inglês, deep learning, séries temporais, NLP e as demais fichas
de modelo.

## Citação

> Beltran, L. A. (2026). *Machine Learning para Pesquisa Científica: métodos, avaliação e interpretação*. https://github.com/LudimiladeAzevedoBeltran/ml-para-pesquisa-cientifica

Ver `CITATION.cff`. O arquivo prepara o depósito no Zenodo; ao criar uma *release*,
o repositório ganha um DOI citável.

## Licença

- **Código:** MIT (ver `LICENSE`).
- **Texto e figuras:** CC BY 4.0 (ver `LICENSE-docs`).

## Autora

**Ludimila de Azevedo Beltran** — ORCID [0009-0003-5070-9472](https://orcid.org/0009-0003-5070-9472)

