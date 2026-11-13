# Revisão da v1 — roteiro para a Ludimila

> **Status:** v1 completa · 31 páginas · 0 links quebrados.
> **Site local:** `docs/_site/index.html`
> **Objetivo desta revisão:** encontrar o que atrapalha o leitor — não reescrever o site inteiro.

---

## 1. Como abrir

Abra `docs/_site/index.html` no navegador. Para navegar com busca e sidebar, use um servidor
local:

```bash
cd docs/_site
python -m http.server 8000
```

Depois acesse `http://localhost:8000`.

---

## 2. Roteiro sugerido (60–90 min)

| Ordem | Página | O que observar |
|---|---|---|
| 1 | Home | a promessa está clara? as trilhas fazem sentido? |
| 2 | Como usar | o leitor sabe por onde começar? |
| 3 | Fundamentos 1–5 | a progressão é suave? algum salto? |
| 4 | Regressão linear | a ficha responde "quando usar" e "como calibrar"? |
| 5 | Calibração 1–5 | manual, grid/random e Optuna estão didáticos? |
| 6 | Métricas de regressão | R², RMSE, MAE e MAPE estão claros? |
| 7 | Classificação + ROC/PR | a armadilha da acurácia e o limiar ficam evidentes? |
| 8 | Interpretação | o semáforo e os 4 casos ensinam a olhar? |
| 9 | Projetos | o ciclo completo fecha? os números batem? |
| 10 | Reprodutibilidade + extras | checklists, glossário e FAQ são úteis? |

Não precisa ler tudo em ordem. Escolha uma trilha e siga.

---

## 3. Lentes de revisão

### Conteúdo
- Alguma afirmação está errada, desatualizada ou sem fonte?
- Falta algum conceito que o leitor vai precisar?
- Algum exemplo é artificial demais ou não representa o problema?

### Didática
- O objetivo da página aparece antes do conteúdo?
- O exemplo vem antes da fórmula?
- O exercício testa o que a página ensinou?
- A página termina com um próximo passo claro?

### Navegação
- A sidebar reflete a ordem real?
- Algum link leva para a página errada?
- A busca encontra os termos principais?
- O leitor consegue voltar ao início sem se perder?

### Código
- O código roda como está? (os blocos foram testados, mas confira os caminhos)
- O resultado impresso bate com o texto?
- As abas Python e R contam a mesma história?
- Falta algum aviso de dependência?

### Figuras
- A legenda é autossuficiente?
- Os eixos têm unidade?
- A interpretação está no texto, não só na figura?
- A figura mostra o que a métrica esconde?

### Tom
- A voz da Garcia está calorosa e direta, sem clichê de IA?
- Algum trecho está professoral demais ou vago?
- Algum aviso importante está escondido no meio do texto?

### Consistência
- Os números entre páginas coincidem (ex.: regressão 25,31; classificação 0,12)?
- As unidades estão padronizadas (ppb, °C, m/s, %)?
- Os nomes dos modelos e métricas estão iguais em todo o site?

---

## 4. Registro de problemas

Copie a tabela para um arquivo ou me envie no chat:

| # | Página | Onde (seção/parágrafo) | Problema | Sugestão | Prioridade |
|---|---|---|---|---|---|
| 1 | | | | | alta/média/baixa |
| 2 | | | | | |
| 3 | | | | | |

**Prioridade:**
- **Alta:** impede entender ou reproduzir.
- **Média:** atrapalha, mas dá para seguir.
- **Baixa:** estilo, typo, preferência.

---

## 5. O que já sei que pode melhorar

Estas são limitações conscientes da v1:

- **Uma ficha de modelo** (regressão linear). Faltam Ridge/Lasso, logística, árvore,
  Random Forest, boosting, KNN, SVM, MLP, K-means, DBSCAN e PCA.
- **Um laboratório de interpretação** (previsto vs observado). Faltam resíduos, curvas de
  aprendizado, importância/SHAP, classificação, cluster e séries temporais.
- **Sem versão em inglês.**
- **Sem deep learning, séries temporais e NLP.**
- **Sem notebooks executáveis** ligados às páginas.
- **Sem interatividade** além do plano estático; o mapa de decisão e o simulador de
  hiperparâmetros ainda não existem.
- **R com splits diferentes do Python** em alguns exemplos; a página avisa, mas os números
  não coincidem.
- **Dados sintéticos** nos projetos; servem para ensinar, não para publicar.

---

## 6. Perguntas que eu quero que você responda

1. A promessa do site ("do zero ao diagnóstico") está sendo cumprida?
2. A ordem das partes funciona para a sua trilha (A, B ou C)?
3. Qual página você abandonaria por tédio ou confusão?
4. Qual página você leria de novo?
5. Falta algum tema que você considera obrigatório?
6. O tom está na medida, ou está professoral demais?
7. Alguma figura não te disse nada?
8. O que você mandaria para um aluno iniciante hoje?

---

## 7. Como me enviar o feedback

Formato mais útil:

```
Página: pt/03-calibracao/04-optuna.qmd
Onde: seção "Como ler a importância"
Problema: não ficou claro o que é importância relativa
Sugestão: dar um exemplo numérico antes da figura
Prioridade: média
```

Com isso eu corrijo sem mexer no que já está bom.

---

## 8. Critério de "v1 aprovada"

- [ ] O leitor iniciante consegue rodar o primeiro modelo sem travar.
- [ ] O leitor intermediário entende calibração e métricas.
- [ ] O leitor avançado confia nos checklists e na interpretação.
- [ ] Nenhum link quebrado.
- [ ] Nenhum número inconsistente entre páginas.
- [ ] O tom está consistente.
- [ ] As limitações estão declaradas.

> "Revisar não é reescrever. É achar o que trava o leitor e tirar do caminho." — Garcia
