# Ferramenta de Análise Causal com Propensity Score Matching (PSM)

## Visão Geral

Esta é uma aplicação web interativa, construída com Streamlit, projetada para facilitar a realização de análises de inferência causal a partir de dados observacionais. O objetivo principal da ferramenta é permitir que usuários, mesmo sem conhecimento aprofundado em programação, possam aplicar a técnica de **Propensity Score Matching (PSM)** para criar um grupo de controle estatisticamente comparável a um grupo de tratamento.

Ao balancear as covariáveis entre os grupos, a ferramenta ajuda a mitigar vieses de seleção, permitindo uma estimativa mais acurada do efeito causal de uma intervenção (tratamento).

## Funcionalidades

- **Upload Interativo de Dados:** Carregue seus próprios datasets no formato `.csv`.
- **Seleção de Variáveis:** Interface amigável para selecionar as colunas de tratamento, identificador único e as covariáveis que serão usadas no modelo.
- **Cálculo de Propensity Score:** Utiliza regressão logística para calcular a probabilidade de cada observação pertencer ao grupo de tratamento.
- **Pareamento Híbrido Avançado:** Em vez de um pareamento simples, a ferramenta implementa uma robusta técnica em duas etapas para garantir a melhor qualidade dos pares sem descartar unidades:
    1.  **Pré-seleção (k-NN):** Para cada unidade tratada, encontra os 5 "vizinhos" mais próximos do grupo de controle com base no *propensity score*.
    2.  **Refinamento (Distância de Mahalanobis):** Dentre os candidatos, seleciona o par final que possui a menor distância de Mahalanobis, garantindo a máxima similaridade no conjunto completo de covariáveis.
- **Análise de Balanceamento:**
    - **Testes Estatísticos:** Executa testes T, Mann-Whitney U e Qui-quadrado para avaliar o balanceamento das covariáveis antes e depois do pareamento.
    - **Visualização Gráfica:** Gera gráficos de densidade (para variáveis numéricas) e de barras (para categóricas) para uma comparação visual do balanceamento.
- **Diagnóstico Detalhado do Pareamento:**
    - **Qualidade do Par:** Cada par é classificado como 'Bom' ou 'Alerta' com base em um *caliper* (regra de Rosenbaum & Rubin), permitindo uma avaliação da qualidade sem descartar dados.
    - **Melhores e Piores Pares:** Apresenta tabelas detalhadas mostrando os 10 melhores e piores pareamentos para uma análise qualitativa profunda.
    - **Explicabilidade do Modelo (SHAP):** Gera um gráfico SHAP para mostrar quais variáveis mais influenciam a probabilidade de tratamento, oferecendo transparência ao modelo de *propensity score*.
- **Download de Resultados:**
    - **Dataset Pareado e Mapa de Pares:** Exporte o dataset final com as unidades pareadas.
    - **Relatório Completo:** Baixe um arquivo `.xlsx` com o sumário estatístico, mapa de pares, melhores/piores pares e a importância das variáveis do modelo logístico.

## Como Usar

### 1. Pré-requisitos

- Python 3.9+
- `pip` (gerenciador de pacotes)

### 2. Configuração do Ambiente

É altamente recomendado criar um ambiente virtual para isolar as dependências do projeto e evitar conflitos.

1.  **Navegue até a pasta do projeto:**
    ```bash
    cd psm-causality
    ```

2.  **Crie o ambiente virtual:**
    *Este comando cria uma pasta `venv` dentro do diretório do projeto.*
    ```bash
    python -m venv venv
    ```

3.  **Ative o ambiente virtual:**
    -   No Windows (PowerShell):
        ```powershell
        .\\venv\\Scripts\\Activate.ps1
        ```
    -   No macOS e Linux:
        ```bash
        source venv/bin/activate
        ```
    *Você saberá que o ambiente está ativo pois o nome `(venv)` aparecerá no início do seu prompt do terminal.*

4.  **Instale as dependências:**
    Com o ambiente virtual ativado, instale as bibliotecas listadas no `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Executando a Aplicação

Com o ambiente virtual ainda ativo, inicie a aplicação Streamlit:

```bash
streamlit run main.py
```

A aplicação será aberta automaticamente no seu navegador.

### 4. Passos na Aplicação

1.  **Carregue seus Dados:** Na barra lateral, faça o upload do seu arquivo `.csv`.
2.  **Selecione as Colunas:**
    - **Coluna de Tratamento:** A coluna que identifica o grupo (ex: 0 ou 1, 'A' ou 'B').
    - **Covariáveis (X):** As variáveis que você acredita que influenciam a alocação no tratamento.
    - **Coluna Identificadora (ID):** A coluna que contém um ID único para cada linha.
3.  **Gere o Dataset Pareado:** Clique no botão para iniciar a análise.
4.  **Analise os Resultados:** Selecione cada covariável no menu suspenso para ver a comparação lado a lado do balanceamento antes e depois do PSM.
5.  **Baixe os Resultados:** Use os botões na barra lateral para baixar o mapa de pares e os sumários estatísticos.

## Estrutura do Projeto

```
psm-causality/
├── data/                  # Pasta para armazenar os datasets
│   ├── groupon.csv
│   └── ...
├── utils/                 # Módulos de suporte
│   ├── psm_logic.py         # Funções de backend para a lógica do PSM
│   └── generate_data_sdv.py # Script para gerar dados sintéticos
├── main.py                # Código principal da aplicação Streamlit
├── requirements.txt       # Lista de dependências Python
└── README.md              # Este arquivo
```
