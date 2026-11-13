# 00 — Planejamento do site de Machine Learning

> **Autora do plano:** Penelope Garcia (revisora técnica)
> **Para:** Ludimila de Azevedo Beltran
> **Data:** 2026-11-13
> **Status:** proposta inicial para decisão — nada aqui está fechado.

---

## 1. Missão do site

Ensinar **métodos de machine learning aplicados à pesquisa científica**, com uma promessa
específica:

> "Ao final, a pessoa consegue escolher um método, calibrar com critério, avaliar com a
> métrica adequada e interpretar resultados a partir de figuras diagnósticas — sabendo o
> que cada escolha implica."

Não é um curso de matemática. Não é uma coleção de snippets. É um **manual de decisão**:
o leitor chega com uma pergunta e sai sabendo qual caminho seguir, quais são as armadilhas
e como defender o resultado.

**O diferencial:** a parte de interpretação não é um apêndice. É um **laboratório** com
figura pronta, código reprodutível e um "semáforo" dizendo o que olhar, o que é aceitável,
o que é alerta e o que é erro fatal.

---

## 2. Público e trilhas

O site deve servir três públicos sem virar três sites. A solução é **trilhas** com a mesma
base de conteúdo, mudando a ordem e a profundidade.

| Trilha | Público | Pré-requisito | Promessa | Tempo |
|---|---|---|---|---|
| **A — Do zero** | Nunca programou ou só usa planilha | Nenhum | Entender o que é ML e rodar o primeiro modelo | ~6 h |
| **B — Já programa** | Sabe Python, nunca treinou modelo | Lógica, funções, pandas básico | Escolher, calibrar e avaliar com critério | ~10 h |
| **C — Já usa ML** | Treina modelos, mas quer defender resultados | scikit-learn básico | Diagnosticar, comparar e relatar sem overclaim | ~8 h |

Cada página declara no topo: **nível**, **tempo de leitura**, **pré-requisitos** e
**objetivo de aprendizagem** (uma frase mensurável).

---

## 3. Princípios didáticos (as regras da casa)

1. **Uma página = uma decisão.** Se a página não ajuda a decidir algo, ela é enciclopédia.
2. **Toda afirmação tem exemplo.** Conceito sem código ou figura é definição de dicionário.
3. **Toda figura tem leitura.** Não basta mostrar o gráfico: dizer o que olhar, o que é normal,
   o que é alerta e o que é erro.
4. **Toda ficha de modelo tem bordas.** Faixa típica, default, o que acontece no limite inferior,
   o que acontece no limite superior, e quando usar escala log.
5. **Toda página tem um exercício com solução escondida.** Aprender é errar com segurança.
6. **Nada de vazamento.** O site ensina a armadilha antes de ensinar o modelo.
7. **Reprodutibilidade é conteúdo, não anexo.** Seed, versão, split e código aparecem no corpo.
8. **Sem clichê de IA.** Nada de "é importante notar", "papel crucial", "mergulhar".
9. **Português claro, termo técnico em inglês entre parênteses na primeira aparição.**
10. **O leitor nunca fica sem o próximo passo.** Cada página termina com "para onde ir agora".

---

## 4. Arquitetura do site (mapa das partes)

```
INÍCIO
├─ Como usar o site
├─ Trilhas (A, B, C)
├─ Mapa de decisão ("qual modelo? qual métrica?")
└─ Glossário rápido

PARTE I — FUNDAMENTOS
├─ 1. O que é machine learning
├─ 2. Como o ML aprende (dados → perda → otimização → generalização)
├─ 3. Tipos de aprendizado e de tarefa
├─ 4. O pipeline completo
└─ 5. Dados e armadilhas (leakage, desbalanceamento, missing, escala)

PARTE II — MODELOS (catálogo de fichas)
├─ Baseline e Dummy
├─ Regressão linear
├─ Regressão regularizada (Ridge, Lasso, Elastic Net)
├─ Regressão logística
├─ KNN
├─ Naive Bayes
├─ SVM
├─ Árvore de decisão
├─ Random Forest
├─ Gradient Boosting (XGBoost, LightGBM, CatBoost)
├─ Redes neurais (MLP)
├─ K-means
├─ DBSCAN e hierárquico
└─ PCA (e t-SNE/UMAP como vizinhos)

PARTE III — CALIBRAÇÃO
├─ Parâmetro vs hiperparâmetro
├─ Calibração manual (como ler curvas e ajustar na mão)
├─ Grid search e random search
├─ Optuna (TPE, pruner, espaço de busca)
└─ Boas práticas (CV aninhada, orçamento, sementes, bordas)

PARTE IV — AVALIAÇÃO
├─ Métricas de regressão
├─ Métricas de classificação
├─ Métricas de clusterização
├─ Comparação estatística de modelos
└─ Qual métrica usar? (tabela de decisão)

PARTE V — INTERPRETAÇÃO (laboratório)
├─ Ajuste em regressão (previsto vs observado)
├─ Resíduos (4 diagnósticos)
├─ Curvas de aprendizado e validação
├─ Importância, permutação e SHAP
├─ Classificação (matriz, ROC, PR, calibração)
├─ Clusterização (silhueta, cotovelo, dispersão)
├─ Séries temporais (forecast com incerteza)
└─ Como relatar o resultado

PARTE VI — PROJETOS GUIADOS
├─ Primeiro modelo em 30 minutos
├─ Projeto de regressão
├─ Projeto de classificação
└─ Projeto de clusterização

PARTE VII — REPRODUTIBILIDADE E BOAS PRÁTICAS
├─ Estrutura de projeto
├─ Sementes, versões e ambientes
├─ Git e GitHub
└─ Checklists (anti-leakage, relato, revisão)

PARTE VIII — EXTRAS
├─ Glossário PT–EN
├─ FAQ
├─ Erros mais comuns
├─ Exercícios
├─ Folhas de cola
└─ Bibliografia
```

---

## 5. Detalhamento página a página

### Parte I — Fundamentos

**1. O que é machine learning**
- Definição operacional (aprender padrão de dados para prever/descrever, não programar regra).
- Origem: Turing, McCulloch-Pitts, perceptron, inverno da IA, renascimento com dados e computação.
- ML × estatística × IA × ciência de dados: o que é fronteira e o que é moda.
- O que ML **não** é: não descobre causalidade sozinho, não substitui desenho de estudo.
- Exemplo mínimo: prever vs explicar.

**2. Como o ML aprende**
- Vocabulário: `X`, `y`, feature, target, amostra, treino/validação/teste.
- Função de perda e otimização (intuição, sem cálculo pesado).
- Viés–variância e o trade-off.
- Generalização: por que o erro de treino mente.
- Overfitting e underfitting com figuras.
- Por que separar teste **antes** de qualquer decisão.

**3. Tipos de aprendizado e de tarefa**
- Supervisionado, não supervisionado, semissupervisionado, auto-supervisionado, por reforço.
- Tarefas: regressão, classificação binária/multiclasse, clusterização, redução de dimensionalidade,
  séries temporais, detecção de anomalias.
- Tabela: tarefa → exemplo → modelo inicial → métrica inicial.
- A confusão clássica: regressão vs classificação; cluster vs classificação.

**4. O pipeline completo**
- Pergunta → dados → EDA → pré-processamento → split → baseline → modelo → calibração →
  avaliação → interpretação → relato/deploy.
- O que pode dar errado em cada etapa (mapa de risco).
- O princípio "baseline primeiro".
- Onde entra o teste e por que ele é sagrado.

**5. Dados e armadilhas**
- Leakage temporal, espacial, de seleção, de target e de pré-processamento.
- Desbalanceamento: o que é, quando importa, o que fazer.
- Missing: MCAR/MAR/MNAR, imputação e o perigo de imputar antes do split.
- Outliers: detectar, entender e decidir (não excluir por reflexo).
- Escala e transformação.
- Multicolinearidade.
- Dados agrupados, espaciais e temporais: por que a CV comum falha.
- Checklist anti-leakage (reutilizável).

### Parte II — Modelos

Cada modelo em uma **ficha padronizada** (template no §6). A página índice é um catálogo
com filtros: tarefa, interpretabilidade, custo, tamanho de dados, se aceita missing, se
precisa de escala.

Modelos e o que cada um ensina:

| Família | Modelos | Ideia central | Interpretabilidade | Custo |
|---|---|---|---|---|
| Baseline | Dummy, média, persistência | Referência mínima | Total | Nulo |
| Linear | Regressão linear, Ridge, Lasso, Elastic Net | Soma ponderada de features | Alta | Baixo |
| Linear generalizado | Regressão logística | Probabilidade via função logística | Alta | Baixo |
| Instância | KNN | Vizinhança | Média | Médio |
| Probabilístico | Naive Bayes | Teorema de Bayes com independência | Média | Baixo |
| Margem | SVM | Separar com a maior margem | Baixa/média | Médio |
| Árvore | Árvore de decisão | Perguntas em sequência | Alta | Baixo |
| Ensemble bagging | Random Forest | Muitas árvores + média | Média | Médio/alto |
| Ensemble boosting | Gradient Boosting, XGBoost, LightGBM, CatBoost | Corrigir erros em sequência | Média (com SHAP) | Alto |
| Rede neural | MLP | Camadas de transformações | Baixa | Alto |
| Partição | K-means | Minimizar distância ao centro | Média | Baixo |
| Densidade | DBSCAN | Regiões densas | Média | Médio |
| Hierárquico | Ward, average, complete | Árvore de fusões | Alta | Médio |
| Projeção | PCA, t-SNE, UMAP | Comprimir dimensões | Média | Baixo/médio |

### Parte III — Calibração

**1. Parâmetro vs hiperparâmetro**
- Parâmetro: aprendido no treino (coeficientes, pesos).
- Hiperparâmetro: escolhido antes (profundidade, `k`, `C`, `n_estimators`).
- Por que não se calibra no teste.
- O custo de calibrar demais.

**2. Calibração manual**
- Como ler curva de aprendizado (treino vs validação).
- Como ler curva de validação (hiperparâmetro vs score).
- Receita: isolar um hiperparâmetro, variar, olhar as duas curvas, decidir.
- Exemplo completo com `max_depth` e `n_estimators`.
- Quando parar: platô, virada, custo.
- O erro de olhar só o score de validação.

**3. Grid e random search**
- Grid: quando faz sentido (poucos hiperparâmetros, faixa curta).
- Random: por que funciona melhor com muitos hiperparâmetros.
- Orçamento de buscas.
- Escala log para hiperparâmetros positivos.

**4. Optuna**
- O que é: framework de otimização bayesiana.
- TPE em intuição: explora e explota, aprende com trials anteriores.
- `study`, `trial`, `suggest_float`, `suggest_int`, `suggest_categorical`.
- `log=True` e por que importa.
- Pruner: parar trials ruins cedo.
- Espaço de busca: como escolher faixas a partir do default.
- Exemplo completo: 50 trials, validação cruzada, `best_params`.
- Erros comuns: espaço largo demais, vazamento do teste, `n_trials` insuficiente,
  não fixar semente, comparar trials com folds diferentes.

**5. Boas práticas**
- CV aninhada (tuning dentro do fold).
- Holdout final intocado.
- Sementes e variabilidade entre folds.
- Registro dos trials (tabela e figura).
- Como reportar a incerteza do melhor score.
- Quando o ganho de tuning não compensa.

### Parte IV — Avaliação

**1. Métricas de regressão**
- R², R² ajustado, RMSE, MAE, MedAE, MAPE, sMAPE, MSLE.
- O que cada uma mede, unidade, faixa, sensibilidade a outliers.
- Tabela "quando usar / quando não usar".
- Diagnóstico de resíduos (liga com a Parte V).
- Erros: R² fora da amostra, R² negativo, MAPE com zero, comparar RMSE entre escalas diferentes.

**2. Métricas de classificação**
- Acurácia, acurácia balanceada, precisão, recall, especificidade, F1, MCC, kappa.
- Matriz de confusão.
- ROC-AUC, PR-AUC, log loss, Brier.
- Curva de calibração.
- O problema da acurácia em dados desbalanceados.
- Quando otimizar recall vs precisão (com exemplos de custo).
- Multiclasse: macro, micro, weighted.

**3. Métricas de clusterização**
- Inércia, silhueta, Calinski-Harabasz, Davies-Bouldin.
- Métricas internas vs externas.
- O problema de escolher `k` por uma métrica só.
- Estabilidade de cluster (bootstrap).

**4. Comparação estatística de modelos**
- Média de folds não basta.
- Intervalo de confiança da métrica.
- Teste de Wilcoxon, Friedman, Nemenyi (intuição e quando usar).
- Correção para múltiplas comparações.

**5. Qual métrica usar?**
- Fluxograma de decisão: tarefa → custo do erro → métrica.
- Tabela-resumo imprimível.
- "A métrica escolhida depois de ver o resultado não vale."

### Parte V — Interpretação (o laboratório)

Cada página segue o **template do §8**: pergunta, código, figura pronta, o que olhar,
semáforo, erro comum, exercício.

Figuras obrigatórias:

| Página | Figuras | O que o leitor aprende |
|---|---|---|
| Ajuste em regressão | previsto vs observado, linha 1:1 | ver sub/superestimação e dispersão |
| Resíduos | resíduos vs ajustados, QQ, histograma, escala-localização | violação de linearidade, heterocedasticidade, normalidade |
| Curvas | curva de aprendizado, curva de validação | overfitting, underfitting, ponto de virada |
| Importância | barras, permutação, SHAP summary/waterfall/dependence | o que pesa, o que é artefato, o que é confundimento |
| Classificação | matriz de confusão, ROC, PR, calibração | escolher limiar, custo do erro, confiança |
| Cluster | dispersão, silhueta, cotovelo, dendrograma | número de clusters, separação, instabilidade |
| Séries temporais | série + previsão + intervalo, resíduos temporais | tendência, sazonalidade, incerteza |
| Relato | tabela de resultados + texto modelo | escrever sem overclaim |

### Parte VI — Projetos guiados

Cada projeto é um caderno executável com a mesma espinha:
pergunta → dados → EDA → baseline → modelos → calibração → avaliação → interpretação →
relatório → checklist de reprodutibilidade.

1. **Primeiro modelo em 30 minutos** — regressão simples, do CSV ao diagnóstico.
2. **Projeto de regressão** — dados ambientais (ex.: série de CH₄ ou variável climática).
3. **Projeto de classificação** — classes desbalanceadas e escolha de limiar.
4. **Projeto de clusterização** — quantos grupos e o que eles significam.

### Parte VII — Reprodutibilidade

- Estrutura de pastas, `requirements.txt`, ambiente.
- Sementes, versões, splits salvos.
- Git/GitHub e GitHub Pages.
- Checklists: anti-leakage, relato de modelo, revisão de figura.
- Model card e data card (versão simples).

### Parte VIII — Extras

- Glossário PT–EN com âncoras.
- FAQ ("preciso de GPU?", "qual modelo começar?", "deep learning sempre ganha?").
- Erros mais comuns, com "antes/depois".
- Exercícios por parte, com solução escondida.
- Folhas de cola (modelos, métricas, hiperparâmetros).
- Bibliografia comentada.

---

## 6. Template de ficha de modelo

Toda ficha de modelo deve ter exatamente estes blocos:

```markdown
# <Nome do modelo>

**Família:** ...
**Tarefa:** regressão | classificação | cluster | projeção
**Em uma frase:** ...
**Intuição:** ... (analogia curta)

## Quando usar
- ...
## Quando NÃO usar
- ...
## Pressupostos
- ...

## Hiperparâmetros
| Nome | O que controla | Faixa típica | Default | Borda inferior | Borda superior | Escala | Dica |
|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | linear/log | ... |

## Como calibrar
- Manual: ...
- Optuna: ...
- Orçamento sugerido: ...

## Código mínimo
```python
...
```

## Métricas recomendadas
- ...

## Erros comuns
- ...

## Interpretabilidade
- ...

## Custo computacional
- ...

## Referências
- ...

## Exercício
- ...
```

### Exemplo resumido: Random Forest (regressão)

| Hiperparâmetro | O que controla | Faixa típica | Default | Borda inferior | Borda superior | Escala |
|---|---|---|---|---|---|---|
| `n_estimators` | número de árvores | 100–1000 | 100 | instável, alta variância | custo sem ganho | linear |
| `max_depth` | profundidade máxima | 3–30 ou `None` | `None` | underfitting | overfitting | linear |
| `min_samples_leaf` | mínimo por folha | 1–50 | 1 | overfitting | underfitting | log |
| `min_samples_split` | mínimo para dividir | 2–50 | 2 | overfitting | underfitting | log |
| `max_features` | features por divisão | `sqrt`, `log2`, 0.1–1.0 | 1.0 (regressão) | mais viés | mais variância | categórica/linear |
| `max_samples` | fração no bootstrap | 0.5–1.0 | 1.0 | mais viés | sem ganho | linear |
| `bootstrap` | amostragem com reposição | `True`/`False` | `True` | — | — | categórica |

**Regra de borda:** se o melhor trial cair no limite da faixa, **a faixa está errada** —
refaça o espaço, não comemore o resultado.

---

## 7. Template de ficha de métrica

```markdown
# <Métrica>

**Tarefa:** regressão | classificação | cluster
**Fórmula (ou intuição):** ...
**Unidade:** ...
**Faixa:** ...
**Melhor valor:** ...
**O que mede:** ...
**O que NÃO mede:** ...

## Quando usar
- ...
## Quando NÃO usar
- ...
## Sensibilidade
- outliers: ...
- escala: ...
- desbalanceamento: ...

## Exemplo numérico
| Observado | Previsto |
|---|---|
| ... | ... |

## Interpretação
- valor bom / aceitável / ruim (com contexto)
## Erros comuns
- ...
```

### Exemplo resumido: RMSE

- **O que mede:** raiz da média dos erros ao quadrado; penaliza erros grandes.
- **Unidade:** a mesma do target.
- **Faixa:** 0 a ∞; melhor = 0.
- **Quando usar:** quando erros grandes são caros e a escala é interpretável.
- **Quando não usar:** target com outliers extremos que dominam; comparação entre escalas diferentes.
- **Erro comum:** comparar RMSE de targets com unidades diferentes.

---

## 8. Template do laboratório de interpretação

```markdown
# <Figura>: como ler

**Pergunta que a figura responde:** ...
**Código reprodutível:** ...
**Figura pronta:** ...

## O que a figura mostra
...

## O que olhar (checklist)
- [ ] ...
- [ ] ...

## Semáforo
| Sinal | Significado | Ação |
|---|---|---|
| Verde | ... | ... |
| Amarelo | ... | ... |
| Vermelho | ... | ... |

## Interpretação errada comum
...

## O que fazer se...
- ...

## Exercício
"Dado este gráfico, qual é o diagnóstico?"
```

---

## 9. Ideias complementares (para acrescentar)

1. **Quickstart "do zero ao primeiro modelo em 30 min"** — âncora para quem chega perdido.
2. **Mapa de decisão interativo** — "qual modelo?", "qual métrica?", "posso usar CV comum?".
3. **Semáforo de figura** — o diferencial didático da Parte V.
4. **"Diagnostique esta figura"** — quiz com 3 gráficos: um saudável, um suspeito, um desastroso.
5. **Simulador de hiperparâmetros** — sliders (Plotly/JS) mostrando o efeito de `max_depth` ou `k`.
6. **Calculadora de métricas** — entra matriz de confusão, sai precisão/recall/F1.
7. **Checklist anti-leakage imprimível** — usado em todo projeto.
8. **"Antes e depois"** — a mesma análise feita errada e corrigida, lado a lado.
9. **"Erros que eu já vi"** — pequenos casos reais (anonimizados) de overclaim.
10. **"Do artigo ao código"** — reproduzir uma figura de um paper e diagnosticar.
11. **Fichas de modelo com selo de custo** — baixo/médio/alto, para escolher com orçamento.
12. **Glossário PT–EN** — resolve a bagunça de termos.
13. **Bibliografia comentada** — não só a referência, mas "leia este capítulo para X".
14. **Autoavaliação por parte** — 5 perguntas, feedback imediato.
15. **"Modo revisor"** — ensinar a criticar uma análise de ML (casa com o painel de revisores).
16. **Datasets de prática** — um clássico e um ambiental (conexão com a pesquisa da Ludimila).
17. **Caderno de campo** — template de projeto para preencher.
18. **Versão EN** — se o site for divulgado internacionalmente.
19. **DOI no Zenodo + CITATION.cff** — o site vira material citável.
20. **CI que executa os notebooks** — garante que o código do site não apodrece.

---

## 10. Estrutura técnica do repositório (Quarto, como o TROPOMI)

```
12_Projeto_Machine-learning_site/
├─ README.md
├─ LICENSE                 # código: MIT
├─ LICENSE-docs            # texto/figuras: CC BY 4.0
├─ CITATION.cff
├─ requirements.txt
├─ data/                   # dados de exemplo (pequenos)
├─ notebooks/              # cadernos executáveis dos projetos
├─ src/                    # funções reutilizáveis (métricas, plots, tuning)
├─ docs/
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
│     └─ 08-extras/
└─ .github/workflows/      # render + execução dos notebooks
```

Recursos do Quarto a usar: busca, sidebar docked, TOC, callouts (dica/atenção/perigo),
código copiável, abas (`tabset`), spoiler para solução de exercício
(`callout-note collapse="true"`), figuras com legenda, equações leves, dark mode.

---

## 11. Roadmap em fases

| Fase | Entrega | Critério de pronto |
|---|---|---|
| **0 — Decisões** | Este plano + respostas às perguntas do §13 | Escopo, idioma e público fechados |
| **1 — Esqueleto** | Quarto renderizando, navbar, 5 páginas fundacionais, 1 ficha de modelo, 1 ficha de métrica, 1 laboratório | Site navegável e publicado |
| **2 — Núcleo** | Partes I a IV completas | Toda ficha de modelo e métrica com exemplo executável |
| **3 — Interpretação** | Parte V + 3 quizzes | Cada figura com semáforo e exercício |
| **4 — Projetos** | Parte VI + dados | Um projeto roda do início ao fim |
| **5 — Polimento** | Extras, EN (se for o caso), DOI, CI | Site citável e reprodutível |

**Regra de ouro do roadmap:** publicar feio e cedo. Um site no ar com 5 páginas boas vale
mais que 40 páginas em rascunho.

---

## 12. Critérios de qualidade por página

Uma página só é considerada pronta quando tem:

- [ ] Um objetivo de aprendizagem mensurável no topo.
- [ ] Nível, tempo e pré-requisitos declarados.
- [ ] Pelo menos um exemplo executável (código testado).
- [ ] Pelo menos uma figura interpretada (quando fizer sentido).
- [ ] Um checklist de erro ou armadilha.
- [ ] Um exercício com solução escondida.
- [ ] Links para o glossário e para a próxima página.
- [ ] Revisão técnica: números conferidos, código rodado, referência real.

---

## 13. Decisões fechadas

| Tema | Decisão |
|---|---|
| Idioma | **Português agora; EN depois** (estrutura já preparada para tradução) |
| Público | **Três trilhas** (A do zero, B já programa, C já usa ML) |
| Matemática | **Intuição + fórmulas comentadas** (sem exigir cálculo) |
| Código | **Python + R** (abas `tabset` com as duas versões) |
| Dados | **Clássicos + ambientais** (iris/diabetes + CH₄/clima) |
| Interatividade | **Estáticas + alguns Plotly** |
| Escopo v1 | **Núcleo** (sem deep learning e sem séries temporais na v1) |
| Início | **Esqueleto + fundamentos** |
| Autoria | Ludimila de Azevedo Beltran — ORCID 0009-0003-5070-9472 |
| Licenças | **MIT** (código) + **CC BY 4.0** (texto e figuras), como no TROPOMI |
| Nome do site | **ML para Pesquisa Científica** |

**Pendência menor:** criar o repositório no GitHub com o slug `ml-para-pesquisa-cientifica`
e confirmar o domínio do GitHub Pages.

### 13.1 Consequências das decisões

- Toda página de código terá **duas abas**: `Python` e `R`. O Python é a referência
  executável; o R espelha o mesmo resultado com `base R` + `ggplot2` + `rpart` (sem exigir
  `tidymodels` na v1).
- A trilha A não exige R; a trilha B/C pode escolher a aba de preferência.
- O laboratório de interpretação usa figuras estáticas; só o mapa de decisão e o
  simulador de hiperparâmetros ganham Plotly.
- A v1 não promete deep learning nem séries temporais; essas páginas ficam como
  "em breve" no roadmap, sem link quebrado.

---

## 14. Próximo passo

Fases 1 a 5 entregues: a v1 está completa — fundamentos, modelos, calibração, avaliação,
interpretação, projetos guiados, reprodutibilidade, extras e publicação no GitHub Pages.
O próximo passo é a v2: versão em inglês, deep learning, séries temporais, NLP e as demais
fichas de modelo.

> "A ideia está boa. Agora vamos tirar do papel antes que ela vire mais um documento
> bonito na pasta." — Garcia

---

## 15. Status das Fases 1 a 5 (entregues)

| Item | Arquivo | Estado |
|---|---|---|
| Planejamento | `00_PLANEJAMENTO_SITE_ML.md` | pronto |
| Estrutura do repositório | `README.md`, licenças, `requirements.txt`, `CITATION.cff` | pronto |
| Configuração do site | `docs/_quarto.yml`, `docs/index.qmd`, `docs/styles.css` | pronto |
| Como usar + trilhas | `docs/pt/00-como-usar.qmd` | pronto |
| Fundamentos 1–5 | `docs/pt/01-fundamentos/` | pronto |
| Mapa de decisão | `docs/pt/02-modelos/mapa-de-decisao.qmd` | pronto |
| Catálogo de modelos | `docs/pt/02-modelos/index.qmd` | pronto |
| Ficha de modelo (regressão linear) | `docs/pt/02-modelos/02-regressao-linear.qmd` | pronto |
| Ficha Ridge / Lasso / Elastic Net | `docs/pt/02-modelos/03-ridge-lasso.qmd` | pronto |
| Ficha Regressão logística | `docs/pt/02-modelos/04-regressao-logistica.qmd` | pronto |
| Ficha Árvore de decisão | `docs/pt/02-modelos/05-arvore-decisao.qmd` | pronto |
| Ficha Random Forest | `docs/pt/02-modelos/06-random-forest.qmd` | pronto |
| Ficha Gradient boosting | `docs/pt/02-modelos/07-gradient-boosting.qmd` | pronto |
| Figuras de modelos | `docs/img/ridge_lasso_caminho.png`, `logistica_sigmoide.png`, `arvore_decisao.png`, `random_forest_oob.png`, `gradient_boosting_curvas.png` + `src/gerar_figuras_modelos.py` | pronto |
| Métricas de regressão | `docs/pt/04-avaliacao/01-regressao.qmd` | pronto |
| Laboratório de interpretação | `docs/pt/05-interpretacao/01-ajuste-regressao.qmd` | pronto |
| Figuras do laboratório | `docs/img/previsto_observado_*.png` + `src/gerar_figuras.py` | pronto |
| Parâmetro vs hiperparâmetro | `docs/pt/03-calibracao/01-parametros-vs-hiperparametros.qmd` | pronto |
| Calibração manual | `docs/pt/03-calibracao/02-calibracao-manual.qmd` | pronto |
| Grid e random search | `docs/pt/03-calibracao/03-grid-random-search.qmd` | pronto |
| Optuna | `docs/pt/03-calibracao/04-optuna.qmd` | pronto |
| Boas práticas de calibração | `docs/pt/03-calibracao/05-boas-praticas.qmd` | pronto |
| Figuras de calibração | `docs/img/curva_*.png`, `grid_vs_random.png`, `optuna_*.png` + `src/gerar_figuras_calibracao.py` | pronto |
| Métricas de classificação | `docs/pt/04-avaliacao/02-classificacao.qmd` | pronto |
| ROC, PR e calibração | `docs/pt/04-avaliacao/03-roc-pr-calibracao.qmd` | pronto |
| Figuras de classificação | `docs/img/matriz_confusao.png`, `curva_roc.png`, `curva_pr.png`, `curva_calibracao.png`, `threshold_precisao_recall.png`, `distribuicao_probabilidades.png` + `src/gerar_figuras_classificacao.py` | pronto |
| Publicação no GitHub Pages | `.github/workflows/publish.yml` | pronto |
| Visão geral dos projetos | `docs/pt/06-projetos/index.qmd` | pronto |
| Primeiro modelo em 30 min | `docs/pt/06-projetos/01-primeiro-modelo-30min.qmd` | pronto |
| Projeto de regressão | `docs/pt/06-projetos/02-regressao.qmd` | pronto |
| Projeto de classificação | `docs/pt/06-projetos/03-classificacao.qmd` | pronto |
| Projeto de clusterização | `docs/pt/06-projetos/04-clusterizacao.qmd` | pronto |
| Dados dos projetos | `data/exemplo_ch4.csv`, `data/exemplo_classificacao.csv`, `data/exemplo_cluster.csv` + `src/gerar_dados_projetos.py` | pronto |
| Figuras dos projetos | `docs/img/projeto_*.png`, `residuos_4paineis.png` + `src/gerar_figuras_projetos.py` | pronto |
| Estrutura, ambiente e sementes | `docs/pt/07-reprodutibilidade/01-estrutura-ambiente.qmd` | pronto |
| Git, GitHub e model card | `docs/pt/07-reprodutibilidade/02-git-model-card.qmd` | pronto |
| Checklists finais | `docs/pt/07-reprodutibilidade/03-checklists.qmd` | pronto |
| Glossário PT–EN | `docs/pt/08-extras/01-glossario.qmd` | pronto |
| FAQ | `docs/pt/08-extras/02-faq.qmd` | pronto |
| Folhas de cola | `docs/pt/08-extras/03-folhas-de-cola.qmd` | pronto |
| Bibliografia comentada | `docs/pt/08-extras/04-bibliografia.qmd` | pronto |
| Erros e exercícios | `docs/pt/08-extras/05-erros-e-exercicios.qmd` | pronto |
| Site renderizado | `docs/_site/` | pronto |

**Próxima fase:** concluir as fichas restantes (Baseline, KNN, Naive Bayes, SVM, MLP,
K-means, DBSCAN/hierárquico e PCA). Depois, v2: versão em inglês, deep learning, séries
temporais e NLP. Antes disso, ativar o GitHub Pages.

> Nota operacional: o `quarto render` dentro do Google Drive pode falhar com `os error 303`
> por bloqueio de sincronização. Renderize a partir de uma cópia local ou pause a
> sincronização. O conteúdo não é afetado.
