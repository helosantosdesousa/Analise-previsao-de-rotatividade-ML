# Projeto Final Bootcamp Dados: Análise e Previsão de Rotatividade de Funcionários com Machine Learning 👩‍💻
Este projeto é o trabalho final do bootcamp "Cientista de Dados" da Data Girls de 2025.

# Resumo
A rotatividade de funcionários representa custos altos de recrutamento e perda de conhecimento. Este projeto buscou identificar os fatores que mais influenciam a saída de colaboradores e criar um modelo preditivo para antecipar casos de risco.

# Objetivo do projeto
O objetivo do projeto foi analisar e modelar os fatores que influenciam a rotatividade de funcionários (Attrition) usando o conjunto de dados "IBM HR Analytics Attrition". O trabalho visou desenvolver modelos de Machine Learning capazes de prever com precisão se um colaborador tem alta probabilidade de deixar a empresa.

# Etapas da análise
1. **Análise Exploratória de Dados (EDA):** Uma análise inicial foi realizada para identificar padrões, verificar a qualidade dos dados e entender as características dos colaboradores. Além disso foram criados gráficos com boxplot que mostraram a relação de attrition com outros fatores
2. **Limpeza e Pré-processamento dos Dados:** Esta fase incluiu o tratamento de outliers, conversão de tipos de dados (como transformar 'Yes'/'No' em booleanos), verificação de valores nulos, remoção de duplicatas, normalização e padronização de variáveis numéricas.
3. **Encoding de Variáveis Categóricas:** As colunas categóricas (como estado civil e departamento) foram convertidas em valores numéricos para que pudessem ser utilizadas nos modelos de Machine Learning.
4. **Modelagem e Otimização:** Foram implementados modelos de classificação Support Vector Classifier (SVC) e Random Forest Classifier.
5. **Validação de dados:** Otimizações de hiperparâmetros (Grid Search, Randomized Search e Bayesian Optimization) e validação cruzada (K-Fold, Stratified K-Fold, Repeated K-Fold e Leave-One-Out) foram utilizadas para garantir a robustez e o bom desempenho dos modelos.

# Principais achados e insights
Com base na análise, os fatores que mais influenciam a rotatividade são:

| Fator                         | Descrição                                              | Porcentagem no Total (%) | Taxa de Rotatividade (%) |
|-------------------------------|--------------------------------------------------------|--------------------------|--------------------------|
| Viagens a Trabalho Frequentes  | Funcionários que viajam frequentemente a trabalho      | 18,84                    | 24,91                    |
| Idade                         | Colaboradores mais jovens (18-24 e 25-34 anos)         | 63.29%                        | Maior tendência          |
| Satisfação com o Ambiente      | Insatisfação no nível 1 (pior nível de satisfação)     | 19.32%                        | 25,35                    |
| Satisfação com o Ambiente      | Maior satisfação (nível 4)                                       | 30.34%                        | 13,45                    |
| Satisfação no Trabalho         | Nível mais baixo de satisfação (nível 1)               | 19.66%                        | 22,84                    |
| Equilíbrio Vida Pessoal/Trabalho | Funcionários sem equilíbrio entre vida pessoal e trabalho | 5.44%           | 31,25                    |
| Relações Interpessoais         | Menor satisfação com relacionamentos                   | 18.78%                        | Maior taxa de rotatividade|
| Opção de compra de ações         | Nível de opção de compra de ações, sendo 0 sem opção de compra                   | 42.93%                        | 24.41% |

Os dados da tabela mostram um retrato claro de quais perfis têm mais chances de deixar a empresa. O colaborador com maior risco de saída costuma ser jovem, entre 18 e 34 anos, viaja frequentemente a trabalho e sente pouco equilíbrio entre sua vida profissional e pessoal. Muitas vezes, ele também está insatisfeito com o ambiente, com suas atividades diárias ou até com as relações interpessoais no trabalho. Um ponto curioso é que benefícios financeiros como opções de compra de ações parecem ter um peso importante: quem não recebe esse benefício apresenta taxas de saída muito mais altas, enquanto níveis intermediários de ações estão ligados a uma permanência maior. Esses achados reforçam que retenção não depende apenas de salário, mas de um conjunto de fatores que envolvem qualidade de vida, reconhecimento e oportunidades.

## Perfil de colaborador tem maior propensão a sair da empresa?
O perfil do colaborador com maior propensão a sair da empresa inclui os funcionários:
  - Viajam com frequência.
  - Mais jovens, entre 18 e 34 anos.
  - Possuem baixo nível de satisfação com o ambiente de trabalho e com o equilíbrio entre vida profissional e pessoal.

# Modelo de ML
O projeto estabeleceu o objetivo de desenvolver modelos de Machine Learning para prever a saída de funcionários com boa precisão. Os modelos escolhidos para a classificação preditiva de rotatividade foram  ```Support Vector Classifier (SVC) -> 86,60% de acurácia ``` e ```Random Forest Classifier -> 85,31% de acurácia```.

  - Modelo SVC: Um algoritmo de classificação que busca encontrar a melhor fronteira de separação entre as classes, maximizando a margem entre elas. Ele é especialmente eficaz quando as classes são separáveis e pode utilizar diferentes funções de kernel para capturar relações lineares e não lineares nos dados.

  - Modelo Random Forest: Um conjunto de múltiplas árvores de decisão que trabalham em conjunto para fazer previsões. Cada árvore é treinada com uma amostra aleatória dos dados e das variáveis, e o resultado final é obtido por votação. Ele é robusto contra overfitting, lida bem com dados complexos e fornece métricas de importância das variáveis.

## Avaliação dos modelos
**Validação cruzada**
  - SVC Otimizado (Stratified K-Fold): Acurácia Média = 0.8660
  - Random Forest Otimizado (Stratified K-Fold): Acurácia Média = 0.8531

**Acurácia no Conjunto de Teste**
  - SVC Otimizado: Acurácia = 0.8912
  - Random Forest Otimizado: Acurácia = 0.8435
    
## Análise de Erros do Modelo e Impacto para o Negócio
Na previsão de rotatividade, falsos positivos indicam funcionários que o modelo prevê que sairão, mas permanecem, podendo gerar ações de retenção desnecessárias. Falsos negativos são mais críticos, pois representam funcionários que saem sem que a empresa intervenha, resultando em custos de substituição e perda de conhecimento. Em geral, reduzir falsos negativos é prioridade, mesmo que aumente um pouco os falsos positivos. Métricas como Recall e Precisão ajudam a equilibrar esse impacto.

## Importância das Variáveis nos Modelos
A análise da importância das variáveis revela quais fatores mais influenciam a saída dos funcionários. Isso confirma insights da análise exploratória e orienta ações de RH, como priorizar melhorias em fatores de alta relevância (ex.: satisfação no trabalho ou tempo sem promoção). Também pode simplificar o modelo, removendo variáveis pouco significativas, sem perda relevante de desempenho.
    
    
# Recomendações práticas
Com base nos principais achados do projeto, as seguintes recomendações podem ser feitas para reduzir a rotatividade:
- **Revisão das Políticas de Viagens:** Avaliar e, se possível, reduzir a frequência de viagens a trabalho, especialmente para o grupo de funcionários que viaja com frequência, que apresenta a maior taxa de desligamento.
- **Foco na Satisfação do Ambiente:** Implementar iniciativas para melhorar o ambiente de trabalho, pois a insatisfação com ele está fortemente correlacionada com a saída de colaboradores.
- **Melhora no Equilíbrio entre Vida e Trabalho:** Desenvolver políticas e programas que ajudem a melhorar o equilíbrio entre vida profissional e pessoal, pois a falta de equilíbrio é um fator significativo de rotatividade.
- **Apoio a Funcionários Jovens:** Criar programas de mentoria e desenvolvimento para os colaboradores mais jovens, que mostram maior propensão a sair da empresa.
Com a aplicação desses modelos, a empresa pode antecipar com até 89% de precisão quais colaboradores têm maior risco de sair, permitindo ações preventivas e reduzindo custos com desligamentos e contratações.
