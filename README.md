# Previsão de Risco de Crédito com Machine Learning

## Sobre o projeto

Este projeto foi desenvolvido para prever se um cliente pode se tornar inadimplente em um empréstimo.

A variável que será prevista é a coluna `loan_status`:

* `0`: cliente adimplente, ou seja, paga o empréstimo corretamente;
* `1`: cliente inadimplente, ou seja, apresenta risco de não pagar o empréstimo.

O objetivo é ajudar uma instituição financeira a tomar decisões mais seguras antes de liberar um empréstimo.

---

## Problema de negócio

Antes de liberar um empréstimo, o banco precisa analisar o risco de cada cliente.

O modelo pode cometer dois tipos principais de erro:

* **Falso positivo:** o modelo classifica um bom pagador como inadimplente. Nesse caso, o banco pode recusar um cliente confiável e perder uma oportunidade de negócio.
* **Falso negativo:** o modelo classifica um cliente inadimplente como seguro. Nesse caso, o banco pode liberar o empréstimo e sofrer prejuízo com a falta de pagamento.

Neste projeto, o falso negativo é considerado o erro mais perigoso, porque pode causar uma perda financeira direta para o banco. Porém, uma quantidade muito alta de falsos positivos também não é desejada, pois pode fazer a empresa rejeitar muitos bons clientes.

---

## Objetivo

O objetivo deste projeto é construir e comparar dois modelos de Machine Learning:

* K-Nearest Neighbors, conhecido como KNN;
* Árvore de Decisão.

Os modelos foram avaliados para descobrir qual apresenta o melhor equilíbrio entre:

* identificação de clientes inadimplentes;
* quantidade de erros;
* desempenho nos dados de teste;
* capacidade de generalização;
* impacto financeiro para a instituição.

---

## Base de dados

Foi utilizada a base `credit_risk_dataset.csv`.

A base possui informações pessoais, profissionais e financeiras dos clientes, além de dados relacionados ao empréstimo solicitado.

A base original possui:

* **32.581 linhas**;
* **12 colunas**.

A variável-alvo é `loan_status`.

---

## Dicionário de dados

| Coluna                       | Descrição                                                                                           |
| ---------------------------- | --------------------------------------------------------------------------------------------------- |
| `person_age`                 | Idade do cliente                                                                                    |
| `person_income`              | Renda anual do cliente                                                                              |
| `person_home_ownership`      | Situação de moradia do cliente                                                                      |
| `person_emp_length`          | Tempo de emprego do cliente em anos                                                                 |
| `loan_intent`                | Finalidade do empréstimo                                                                            |
| `loan_grade`                 | Classificação de risco do empréstimo                                                                |
| `loan_amnt`                  | Valor solicitado no empréstimo                                                                      |
| `loan_int_rate`              | Taxa de juros do empréstimo                                                                         |
| `loan_status`                | Situação do empréstimo: 0 para adimplente e 1 para inadimplente                                     |
| `loan_percent_income`        | Proporção entre o valor do empréstimo e a renda do cliente                                          |
| `cb_person_default_on_file`  | Indica se o cliente possui inadimplência anterior registrada                                        |
| `cb_person_cred_hist_length` | Tempo do histórico de crédito do cliente                                                            |
| `comprometimento_renda`      | Percentual da renda comprometida pelo empréstimo, calculado por `(loan_amnt / person_income) * 100` |

---

## Etapas do projeto

O projeto foi dividido nas seguintes etapas:

1. carregamento da base de dados;
2. análise exploratória dos dados;
3. análise dos tipos das variáveis;
4. identificação de valores nulos;
5. identificação e remoção de duplicatas;
6. análise e tratamento de inconsistências;
7. criação da coluna `comprometimento_renda`;
8. separação entre treino e teste;
9. transformação das variáveis categóricas;
10. balanceamento das classes;
11. escalonamento das variáveis utilizadas pelo KNN;
12. teste de diferentes valores de K;
13. teste de diferentes profundidades da Árvore de Decisão;
14. análise de overfitting;
15. comparação dos modelos;
16. análise das matrizes de confusão;
17. escolha do modelo final.

---

## Análise exploratória dos dados

Durante a análise exploratória, foram verificados:

* quantidade de linhas e colunas;
* tipos das variáveis;
* estatísticas descritivas;
* valores nulos;
* linhas duplicadas;
* distribuição da variável-alvo;
* distribuição das idades;
* correlação entre as variáveis numéricas;
* presença de outliers.

A análise mostrou que a variável-alvo é desbalanceada, pois existem mais clientes adimplentes do que inadimplentes.

Também foram encontrados valores ausentes, registros duplicados e valores extremos em algumas variáveis.

Essas informações foram importantes para definir a preparação dos dados antes do treinamento dos modelos.

---

## Tratamento dos dados

### Duplicatas

As linhas duplicadas foram removidas para evitar que os mesmos registros tivessem peso maior durante o treinamento.

### Inconsistências

Foram removidos registros com valores logicamente impossíveis, como:

* idade menor que 18 anos;
* idade maior que 100 anos;
* tempo de emprego negativo;
* tempo de emprego maior do que o tempo possível de trabalho do cliente.

### Valores nulos

Os valores nulos das variáveis numéricas foram preenchidos com a mediana.

A mediana foi escolhida porque é menos afetada por valores extremos do que a média.

Para evitar vazamento de dados, as medianas foram calculadas somente no conjunto de treino e depois aplicadas no conjunto de teste.

### Outliers

Os modelos receberam tratamentos diferentes.

Para o KNN, foi aplicado clipping entre os percentis 1 e 99, pois esse algoritmo calcula distâncias e é muito sensível a valores extremos.

A Árvore de Decisão recebeu os valores financeiros originais, pois esse modelo é mais resistente a diferenças de escala e outliers.

---

## Engenharia de atributos

Foi criada a coluna `comprometimento_renda`.

A fórmula utilizada foi:

```python
comprometimento_renda = (loan_amnt / person_income) * 100
```

Essa variável representa o percentual da renda anual do cliente que corresponde ao valor solicitado no empréstimo.

O cálculo foi protegido contra divisão por zero, valores nulos e valores infinitos.

---

## Separação entre treino e teste

A base foi separada em:

* 80% para treino;
* 20% para teste.

Foi utilizado o parâmetro `stratify=y` para preservar a proporção das classes nos dois conjuntos.

O conjunto de teste não foi balanceado, pois ele deve representar uma situação mais próxima dos dados reais.

---

## Transformação das variáveis categóricas

As colunas de texto foram transformadas em valores numéricos utilizando One-Hot Encoding.

O encoder foi ajustado somente com os dados de treino e depois aplicado aos dados de teste.

Também foi utilizado `handle_unknown='ignore'` para evitar erros caso apareça no teste uma categoria que não estava presente no treino.

---

## Balanceamento das classes

Como a base possui mais clientes adimplentes do que inadimplentes, foi aplicado o `RandomUnderSampler`.

O balanceamento foi realizado somente no conjunto de treino.

O conjunto de teste foi mantido com sua distribuição original para garantir uma avaliação mais realista.

---

## Escalonamento

O `StandardScaler` foi aplicado somente nas variáveis contínuas utilizadas pelo KNN.

O KNN precisa de escalonamento porque utiliza a distância entre os registros.

A Árvore de Decisão não recebeu escalonamento, pois esse algoritmo realiza divisões nos valores das variáveis e não depende da unidade de medida.

---

## Modelos testados

### KNN

Foram testados os seguintes valores de K:

* 3;
* 5;
* 7;
* 9.

O melhor resultado foi obtido com:

* **K = 7**.

### Árvore de Decisão

Foram testadas as seguintes profundidades:

* 3;
* 5;
* 7;
* 9;
* sem limite de profundidade.

O melhor resultado foi obtido com:

* **profundidade máxima = 9**.

---

## Análise de overfitting

Os modelos foram avaliados nos dados de treino e de teste.

Quando um modelo apresenta desempenho muito alto no treino e desempenho menor no teste, existe um sinal de overfitting.

A Árvore sem limite de profundidade apresentou um desempenho muito alto no treino, mas perdeu capacidade de generalização no teste.

Por isso, foi escolhida uma Árvore com profundidade limitada, que apresentou um melhor equilíbrio entre treino e teste.

---

## Resultados finais

| Modelo            |     Configuração | Acurácia | Recall da classe 1 | F1 da classe 1 | Falsos positivos | Falsos negativos |
| ----------------- | ---------------: | -------: | -----------------: | -------------: | ---------------: | ---------------: |
| KNN               |            K = 7 |   82,55% |             77,86% |         66,13% |              817 |              314 |
| Árvore de Decisão | profundidade = 9 |   89,48% |             76,73% |         76,14% |              352 |              330 |

O KNN identificou uma quantidade um pouco maior de clientes inadimplentes, apresentando menos falsos negativos.

Porém, o KNN também apresentou uma quantidade muito maior de falsos positivos, classificando muitos bons pagadores como clientes de risco.

A Árvore de Decisão apresentou:

* maior acurácia;
* maior F1 da classe inadimplente;
* menos falsos positivos;
* bom recall;
* melhor equilíbrio geral entre os erros.

---

## Veredito de negócio

O modelo recomendado para este projeto foi a **Árvore de Decisão com profundidade máxima igual a 9**.

O KNN conseguiu identificar alguns clientes inadimplentes adicionais, mas gerou muitos falsos positivos. Isso poderia fazer o banco recusar uma grande quantidade de bons clientes e perder oportunidades de negócio.

A Árvore apresentou uma quantidade muito menor de falsos positivos, mantendo um recall próximo ao do KNN e alcançando um desempenho geral superior.

Por esse motivo, a Árvore de Decisão apresentou o melhor equilíbrio entre proteção contra inadimplência e manutenção de bons clientes.

Antes de utilizar o modelo em uma situação real, o banco deveria informar o custo financeiro real de cada falso positivo e falso negativo. Com esses valores, seria possível ajustar a decisão final de acordo com a estratégia da instituição.

---

## Resumo executivo

A análise dos dados mostrou que a base possui desbalanceamento, valores ausentes, duplicatas e alguns registros extremos.

Após a limpeza e preparação dos dados, foram comparados os modelos KNN e Árvore de Decisão.

O KNN apresentou bom recall para a classe inadimplente, mas produziu muitos falsos positivos.

A Árvore de Decisão apresentou maior acurácia, melhor F1 e uma quantidade muito menor de bons clientes classificados incorretamente como inadimplentes.

Com base nos resultados estatísticos e no impacto dos erros para o banco, a Árvore de Decisão com profundidade 9 foi escolhida como o melhor modelo deste projeto.

---

## Tecnologias utilizadas

* Python;
* Pandas;
* NumPy;
* Matplotlib;
* Seaborn;
* Scikit-learn;
* Imbalanced-learn;
* Jupyter Notebook.

---

## Como executar o projeto

1. Clone ou baixe este repositório.
2. Coloque o arquivo `credit_risk_dataset.csv` na mesma pasta do notebook.
3. Instale as bibliotecas necessárias.
4. Abra o notebook no Jupyter Notebook ou Visual Studio Code.
5. Execute as células na ordem apresentada.

Para instalar as dependências, utilize:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

---

## Estrutura do repositório

```text
projeto-risco-credito/
├── credit_risk_dataset.csv
├── code_final_project.ipynb
├── README.md
└── requirements.txt
```

---

## Autor

**Miguel Magro Martins**
