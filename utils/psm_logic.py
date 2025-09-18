# --- Importações ---
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from scipy.spatial.distance import cdist
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import numpy as np
import shap
from shap import links


# --------------------------------------------------------------------------
# Funções de Análise de Balanceamento
# --------------------------------------------------------------------------

def run_balance_tests(df: pd.DataFrame, treatment_col: str, covariates: list) -> pd.DataFrame:
    """
    Executa testes de balanceamento estatístico entre os grupos de tratamento e controle.

    Para cada covariável, aplica o teste apropriado:
    - T-test e Mann-Whitney U para variáveis numéricas contínuas.
    - Teste Qui-quadrado para variáveis categóricas ou numéricas binárias.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados.
        treatment_col (str): O nome da coluna que define o tratamento.
        covariates (list): A lista de nomes das colunas de covariáveis a serem testadas.

    Returns:
        pd.DataFrame: Um DataFrame com os resultados dos testes (Variável, Teste, P-Valor).
    """
    treatment_values = df[treatment_col].unique()
    if len(treatment_values) != 2:
        raise ValueError(f"A coluna de tratamento '{treatment_col}' deve ter exatamente 2 grupos.")
    
    # Define os grupos de tratamento e controle com base nos valores da coluna
    treatment_group_val = treatment_values[0]
    control_group_val = treatment_values[1]

    treatment_data = df[df[treatment_col] == treatment_group_val]
    control_data = df[df[treatment_col] == control_group_val]

    results = []

    for var in covariates:
        # Variáveis numéricas com mais de 2 valores únicos são tratadas como contínuas
        if pd.api.types.is_numeric_dtype(df[var]) and df[var].nunique() > 2:
            group1 = treatment_data[var].dropna()
            group2 = control_data[var].dropna()
            
            # T-test para comparar médias
            t_stat, t_p = ttest_ind(group1, group2)
            results.append({'Variável': var, 'Teste': 'T-test (médias)', 'P-Valor': t_p})
            
            # Mann-Whitney U para comparar distribuições
            if len(group1) > 0 and len(group2) > 0:
                u_stat, u_p = mannwhitneyu(group1, group2)
                results.append({'Variável': var, 'Teste': 'Mann-Whitney U (distribuições)', 'P-Valor': u_p})
        
        # Variáveis categóricas ou numéricas binárias
        else:
            contingency_table = pd.crosstab(df[var], df[treatment_col])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            results.append({'Variável': var, 'Teste': 'Chi-quadrado (proporções)', 'P-Valor': p})

    return pd.DataFrame(results).sort_values(by=['Variável', 'Teste']).reset_index(drop=True)


def generate_single_balance_plot(df: pd.DataFrame, treatment_col: str, var: str) -> plt.Figure:
    """
    Gera um gráfico de distribuição para uma única covariável para visualizar o balanceamento.

    - Gráfico de densidade (KDE) para variáveis numéricas contínuas.
    - Gráfico de barras de proporção para variáveis categóricas.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados.
        treatment_col (str): O nome da coluna de tratamento.
        var (str): O nome da covariável a ser plotada.

    Returns:
        plt.Figure: O objeto da figura do Matplotlib contendo o gráfico.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Gráfico de densidade para variáveis numéricas contínuas
    if pd.api.types.is_numeric_dtype(df[var]) and df[var].nunique() > 2:
        sns.kdeplot(data=df, x=var, hue=treatment_col, fill=True, ax=ax, common_norm=False)
        ax.set_title(f'Distribuição de {var}')
    # Gráfico de barras para variáveis categóricas
    else:
        plot_df = df.groupby(treatment_col)[var].value_counts(normalize=True).mul(100).rename('percent').reset_index()
        sns.barplot(data=plot_df, x=var, y='percent', hue=treatment_col, ax=ax)
        ax.set_title(f'Proporção de {var}')
        ax.set_ylabel('Percentual (%)')
    
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Pipeline Principal do PSM
# --------------------------------------------------------------------------

def perform_psm_analysis(
    df: pd.DataFrame, 
    treatment_col: str, 
    covariates: list, 
    id_col: str,
    outcome_cols: list = None
) -> dict:
    """
    Executa o pipeline completo de Propensity Score Matching e gera diagnósticos.

    O processo inclui:
    1. Pré-processamento e Encoding.
    2. Cálculo do Propensity Score.
    3. Pareamento 1-para-1 com reposição.
    4. Cálculo do Caliper (regra de Rosenbaum & Rubin) para diagnóstico.
    5. Geração de sumário estatístico do pareamento.
    6. Identificação dos 10 piores pareamentos.
    7. Extração dos coeficientes do modelo para análise de importância.
    8. Cálculo e visualização SHAP para explicabilidade do modelo.
    """
    # --- 1. Pré-processamento e Encoding ---
    df_processed = df.copy()
    
    # Separar covariáveis em numéricas e categóricas
    numerical_vars = [var for var in covariates if pd.api.types.is_numeric_dtype(df[var]) and df[var].nunique() > 2]
    categorical_vars = [var for var in covariates if var not in numerical_vars]

    # Padronizar variáveis numéricas
    if numerical_vars:
        scaler = StandardScaler()
        df_processed[numerical_vars] = scaler.fit_transform(df_processed[numerical_vars])

    # One-Hot Encode para variáveis categóricas
    if categorical_vars:
        df_processed = pd.get_dummies(df_processed, columns=categorical_vars, drop_first=True, dtype=int)

    # Definir a variável dependente (y) para o modelo logístico
    treatment_group = df[treatment_col].unique()[0]
    y = (df_processed[treatment_col] == treatment_group).astype(int)
    
    # Construir a matriz de features (X) com as variáveis processadas
    dummy_cols = [c for c in df_processed.columns if any(c.startswith(f"{cov}_") for cov in categorical_vars)]
    final_covariates = numerical_vars + dummy_cols
    X = df_processed[final_covariates]
    X = sm.add_constant(X, prepend=False)

    # --- 2. Calcular Propensity Score ---
    logit_model = sm.Logit(y, X.astype(float))
    result = logit_model.fit(disp=0)
    df_processed['ps'] = result.predict(X)

    # --- 7. Extrair Importância das Variáveis (Coeficientes) ---
    feature_importance_df = pd.DataFrame({
        'Variável': result.params.index,
        'Coeficiente': result.params.values,
        'P-Valor': result.pvalues.values
    }).set_index('Variável')
    
    # Remover a constante e calcular a importância absoluta para ordenação
    feature_importance_df = feature_importance_df.drop('const', errors='ignore')
    feature_importance_df['Importância (Absoluta)'] = feature_importance_df['Coeficiente'].abs()
    feature_importance_df = feature_importance_df.sort_values(by='Importância (Absoluta)', ascending=False)

    # --- 3. Pareamento Híbrido: KNN no PS + Distância de Mahalanobis nas Covariáveis ---
    treatment_df = df_processed[y == 1]
    control_df = df_processed[y == 0]
    
    # Prepara os dados para o cálculo da distância de Mahalanobis
    X_covariates = df_processed[final_covariates]
    control_covariates = X_covariates.loc[control_df.index]
    
    try:
        inv_cov_matrix = np.linalg.inv(np.cov(control_covariates.T))
    except np.linalg.LinAlgError:
        inv_cov_matrix = np.linalg.pinv(np.cov(control_covariates.T))

    # --- Execução do Pareamento ---
    # 1. Encontrar os k vizinhos mais próximos no espaço do propensity score para pré-selecionar
    K_NEIGHBORS = 5 
    nn_ps = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='ball_tree', n_jobs=-1)
    nn_ps.fit(control_df[['ps']])
    
    # Encontra os k candidatos para cada tratado
    _, candidate_indices_ps = nn_ps.kneighbors(treatment_df[['ps']])
    
    # Listas para armazenar os pares finais
    treatment_indices = treatment_df.index
    matched_control_indices_list = []
    ps_distances = []
    
    # 2. Para cada tratado, encontrar o melhor par (menor dist. Mahalanobis) entre os k candidatos
    for i, treated_idx in enumerate(treatment_indices):
        candidate_control_indices_pos = candidate_indices_ps[i]
        actual_candidate_indices = control_df.index[candidate_control_indices_pos]
        
        treated_cov = X_covariates.loc[treated_idx].values.reshape(1, -1)
        candidate_covs = X_covariates.loc[actual_candidate_indices].values
        
        m_distances = cdist(treated_cov, candidate_covs, metric='mahalanobis', VI=inv_cov_matrix)
        
        best_candidate_pos = np.argmin(m_distances)
        best_control_idx = actual_candidate_indices[best_candidate_pos]
        
        matched_control_indices_list.append(best_control_idx)
        
        ps_dist = abs(df_processed.loc[treated_idx, 'ps'] - df_processed.loc[best_control_idx, 'ps'])
        ps_distances.append(ps_dist)

    matched_control_indices = pd.Index(matched_control_indices_list)

    matched_pairs = pd.DataFrame({
        'treatment_index': treatment_indices,
        'control_index': matched_control_indices,
        'distance': ps_distances
    })

    all_matched_indices = treatment_indices.union(matched_control_indices)
    df_matched = df.loc[all_matched_indices]

    # --- 4. Cálculo do Caliper para Diagnóstico (sem descarte) ---
    # Calcula o logit do propensity score para avaliar a qualidade do par
    ps = df_processed['ps']
    logit_ps = np.log(ps / (1 - ps + 1e-9))
    caliper_width = 0.2 * logit_ps.std()
    
    # A comparação é feita no logit do PS para a flag de qualidade
    logit_ps_treated = logit_ps.loc[matched_pairs['treatment_index']].values
    logit_ps_control = logit_ps.loc[matched_pairs['control_index']].values
    distance_logit = np.abs(logit_ps_treated - logit_ps_control)
    
    matched_pairs['qualidade_do_par'] = np.where(distance_logit <= caliper_width, 'Bom', 'Alerta (Fora do Caliper)')
    unidades_fora_do_caliper = (matched_pairs['qualidade_do_par'] != 'Bom').sum()

    # --- 5. Geração de Sumário Estatístico ---
    summary_stats = {
        'Unidades Tratadas (Original)': len(treatment_df),
        'Unidades de Controle (Original)': len(control_df),
        'Unidades de Controle Pareadas (Total)': len(matched_control_indices),
        'Unidades de Controle Pareadas (Únicas)': matched_control_indices.nunique(),
        'Unidades Fora do Caliper (Diagnóstico)': unidades_fora_do_caliper,
        'Total de Unidades no Dataset Pareado': len(df_matched)
    }
    summary_df = pd.DataFrame([summary_stats]).T.reset_index()
    summary_df.columns = ['Métrica', 'Valor']

    # --- 6. Identificação dos Piores Pareamentos ---
    worst_matches_pairs = matched_pairs.sort_values('distance', ascending=False).head(10).reset_index(drop=True)
    
    worst_treated_indices = worst_matches_pairs['treatment_index']
    worst_control_indices = worst_matches_pairs['control_index']

    # DataFrame base com IDs, PS scores e flags de qualidade
    base_worst_df = pd.DataFrame({
        'treatment_index': worst_treated_indices, # Adicionar índice
        f'tratado_{id_col}': df.loc[worst_treated_indices, id_col].values,
        'ps_tratado': df_processed.loc[worst_treated_indices, 'ps'].values,
        'control_index': worst_control_indices, # Adicionar índice
        f'controle_{id_col}': df.loc[worst_control_indices, id_col].values,
        'ps_controle': df_processed.loc[worst_control_indices, 'ps'].values,
        'distancia_ps': worst_matches_pairs['distance'].values,
        'qualidade_do_par': worst_matches_pairs['qualidade_do_par'].values
    })

    # Adicionar as covariáveis e variáveis de resultado para comparação direta
    cols_to_add = covariates[:] # Copia a lista de covariáveis
    if outcome_cols:
        # Adiciona as colunas de resultado no início para destaque
        cols_to_add = outcome_cols + cols_to_add

    treated_vars = df.loc[worst_treated_indices, cols_to_add].reset_index(drop=True)
    treated_vars = treated_vars.add_suffix('_tratado')

    control_vars = df.loc[worst_control_indices, cols_to_add].reset_index(drop=True)
    control_vars = control_vars.add_suffix('_controle')

    # Combinar todas as informações
    worst_matches_df = pd.concat([base_worst_df, treated_vars, control_vars], axis=1)
    
    # Reordenar colunas para melhor legibilidade
    id_ps_cols = list(base_worst_df.columns)
    
    # Intercalar colunas de resultado
    outcome_cols_interleaved = []
    if outcome_cols:
        outcome_cols_interleaved = [col for var in outcome_cols for col in [f'{var}_tratado', f'{var}_controle']]
        
    # Intercalar colunas de covariáveis
    covariate_cols_interleaved = [col for cov in covariates for col in [f'{cov}_tratado', f'{cov}_controle']]
    
    final_column_order = id_ps_cols + outcome_cols_interleaved + covariate_cols_interleaved
    worst_matches_df = worst_matches_df.reindex(columns=final_column_order)

    # --- 7. Identificação dos Melhores Pareamentos ---
    best_matches_pairs = matched_pairs.sort_values('distance', ascending=True).head(10).reset_index(drop=True)

    best_treated_indices = best_matches_pairs['treatment_index']
    best_control_indices = best_matches_pairs['control_index']

    # DataFrame base para os melhores pares
    base_best_df = pd.DataFrame({
        'treatment_index': best_treated_indices, # Adicionar índice
        f'tratado_{id_col}': df.loc[best_treated_indices, id_col].values,
        'ps_tratado': df_processed.loc[best_treated_indices, 'ps'].values,
        'control_index': best_control_indices, # Adicionar índice
        f'controle_{id_col}': df.loc[best_control_indices, id_col].values,
        'ps_controle': df_processed.loc[best_control_indices, 'ps'].values,
        'distancia_ps': best_matches_pairs['distance'].values,
        'qualidade_do_par': best_matches_pairs['qualidade_do_par'].values
    })

    # Adicionar covariáveis e resultados para os melhores pares
    best_treated_vars = df.loc[best_treated_indices, cols_to_add].reset_index(drop=True).add_suffix('_tratado')
    best_control_vars = df.loc[best_control_indices, cols_to_add].reset_index(drop=True).add_suffix('_controle')
    
    best_matches_df = pd.concat([base_best_df, best_treated_vars, best_control_vars], axis=1)
    best_matches_df = best_matches_df.reindex(columns=final_column_order) # Reusa a ordem das colunas

    # --- 8. Análise SHAP ---
    X_shap = X.drop('const', axis=1, errors='ignore')

    # NOVO: Cria uma versão de X para exibição com os valores originais (não padronizados)
    X_display = X_shap.copy()
    if numerical_vars:
        # Substitui as colunas numéricas padronizadas pelas originais do df
        X_display[numerical_vars] = df.loc[X_display.index, numerical_vars]

    # Extrair coeficientes e intercepto do modelo statsmodels
    coefs = result.params.drop('const', errors='ignore').values
    intercept = result.params.get('const', 0.0)
    
    # O explainer espera a tupla na ordem (coeficientes, intercepto)
    explainer = shap.LinearExplainer((coefs, intercept), X_shap, link=links.logit)
    shap_values = explainer.shap_values(X_shap)

    # Gerar o gráfico de resumo SHAP em memória
    fig_shap, ax_shap = plt.subplots()
    shap.summary_plot(shap_values, X_shap, show=False)
    fig_shap.tight_layout()
    
    # Converter a figura para bytes para passar para o Streamlit
    buf = BytesIO()
    fig_shap.savefig(buf, format="png")
    shap_summary_plot_bytes = buf.getvalue()
    plt.close(fig_shap) # Fechar a figura para liberar memória

    return {
        "df_matched": df_matched,
        "matched_pairs": matched_pairs,
        "summary_df": summary_df,
        "worst_matches_df": worst_matches_df,
        "best_matches_df": best_matches_df,
        "feature_importance_df": feature_importance_df,
        "shap_summary_plot": shap_summary_plot_bytes,
        "shap_explainer": explainer,
        "shap_values": shap_values,
        "X_shap": X_shap,
        "X_display": X_display,
        "numerical_vars": numerical_vars
    }


# --------------------------------------------------------------------------
# Funções Utilitárias
# --------------------------------------------------------------------------

def to_excel(dfs: dict) -> bytes:
    """
    Converte um dicionário de DataFrames para um arquivo Excel em memória.

    Args:
        dfs (dict): Um dicionário onde as chaves são os nomes das abas
                    e os valores são os DataFrames.

    Returns:
        bytes: Os dados do arquivo Excel em formato de bytes.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    processed_data = output.getvalue()
    return processed_data
