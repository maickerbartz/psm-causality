# --- Importações ---
import streamlit as st
import pandas as pd
from utils.psm_logic import (
    run_balance_tests, 
    perform_psm_analysis, 
    to_excel, 
    generate_single_balance_plot
)
import shap
import streamlit.components.v1 as components

# --------------------------------------------------------------------------
# Configuração da Página
# --------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Análise de PSM")


# --------------------------------------------------------------------------
# Título e Descrição
# --------------------------------------------------------------------------
st.title("Ferramenta de Análise Causal via Propensity Score Matching (PSM)")

st.markdown("""
Esta ferramenta utiliza o PSM para criar um dataset balanceado a partir de dados observacionais, 
controlando por vieses de seleção. É um passo crucial para análises de inferência causal.

**Como usar:**
1.  **Carregue seus dados** no formato CSV.
2.  **Selecione as colunas** apropriadas para tratamento, covariáveis e ID.
3.  **Execute a análise** e explore o balanceamento das variáveis.
4.  **Faça o download** dos resultados para análises futuras.
""")


# --------------------------------------------------------------------------
# Estado da Sessão (Session State)
# --------------------------------------------------------------------------
# Inicializa o estado da sessão para controlar a execução da análise
# e armazenar os resultados, evitando re-cálculos a cada interação.
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'results' not in st.session_state:
    st.session_state.results = None


# --------------------------------------------------------------------------
# Barra Lateral (Sidebar) para Inputs do Usuário
# --------------------------------------------------------------------------
df = None
with st.sidebar:
    st.header("1. Carregar Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    
    if df is not None:
        st.header("2. Selecionar Colunas")
        all_cols = df.columns.tolist()

        treatment_col = st.selectbox(
            "Coluna de Tratamento", 
            all_cols, 
            index=None, 
            placeholder="Selecione a coluna..."
        )
        
        potential_covariates = [c for c in all_cols if c != treatment_col]
        covariates = st.multiselect(
            "Covariáveis (X)", 
            options=potential_covariates, 
            default=None
        )

        id_col = st.selectbox(
            "Coluna Identificadora (ID)", 
            all_cols, 
            index=None, 
            placeholder="Selecione a coluna de ID..."
        )
        
        # --- NOVO: Seletor de Variáveis de Resultado ---
        potential_outcomes = [
            c for c in all_cols if c not in [treatment_col, id_col] + covariates
        ]
        outcome_cols = st.multiselect(
            "Variáveis de Resultado (Opcional)",
            options=potential_outcomes,
            default=None,
            help="Selecione as variáveis que você deseja comparar entre os grupos após o pareamento."
        )
        
        st.header("3. Iniciar Análise")
        run_analysis_button = st.button(
            "Gerar Dataset Pareado", 
            disabled=(not treatment_col or not covariates or not id_col)
        )
        
        if run_analysis_button:
            st.session_state.run_analysis = True
            st.session_state.results = None # Reseta resultados para uma nova análise
            # --- NOVO: Armazenar outcome_cols no estado da sessão ---
            st.session_state.outcome_cols = outcome_cols


# --------------------------------------------------------------------------
# Lógica Principal de Execução da Análise
# --------------------------------------------------------------------------
# A análise roda apenas UMA VEZ por clique no botão.
if st.session_state.run_analysis and st.session_state.results is None:
    if df is not None:
        with st.spinner("Analisando dados e realizando o pareamento..."):
            try:
                # --- ATUALIZADO: Passar outcome_cols para a função de análise ---
                outcome_cols_to_pass = st.session_state.get('outcome_cols', [])
                analysis_results = perform_psm_analysis(
                    df, treatment_col, covariates, id_col, outcome_cols_to_pass
                )
                
                # Calcula o balanceamento antes e depois
                balance_before = run_balance_tests(df, treatment_col, covariates)
                balance_after = run_balance_tests(analysis_results["df_matched"], treatment_col, covariates)
                
                # Armazena todos os resultados na sessão, desempacotando o dicionário
                st.session_state.results = {
                    "df": df,
                    "balance_before": balance_before,
                    "balance_after": balance_after,
                    "covariates": covariates,
                    "treatment_col": treatment_col,
                    "id_col": id_col,
                    **analysis_results  # Desempacota as chaves de analysis_results aqui
                }
                st.success("Pareamento concluído!")

            except Exception as e:
                st.error(f"Ocorreu um erro na análise: {e}")
                st.info("Dicas: Verifique se a coluna de tratamento tem apenas 2 grupos. Tente remover covariáveis que sejam IDs ou que sejam combinações lineares de outras.")
                st.session_state.run_analysis = False
                st.session_state.results = None # Garante que não tente exibir resultados parciais

# --------------------------------------------------------------------------
# Funções de Ajuda (Helpers)
# --------------------------------------------------------------------------
def st_shap(plot, height=None):
    """Função de ajuda para renderizar plots SHAP no Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --------------------------------------------------------------------------
# Área de Exibição de Resultados
# --------------------------------------------------------------------------
if st.session_state.get('results'):
    results = st.session_state.results
    
    st.header("Diagnósticos e Resultados do Pareamento")

    # --- Abas para organizar os resultados ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Sumário do Pareamento", 
        "Análise de Balanceamento",
        "Análise SHAP (Explicabilidade)",
        "Importância das Variáveis",
        "Análise de Piores Pares",
        "Análise de Melhores Pares"
    ])

    with tab1:
        st.subheader("Métricas Gerais do Pareamento")
        st.dataframe(results['summary_df'].style.format({'Valor': lambda x: f"{int(x):,d}".replace(',', '.')}))
        st.info(
            "**Unidades Tratadas Perdidas:** Com o método dos Vizinhos Mais Próximos (k=1) sem restrições, "
            "nenhuma unidade tratada é perdida, pois o algoritmo sempre encontrará um par.\n\n"
            "**Unidades Fora do Caliper:** Esta métrica utiliza a regra de Rosenbaum & Rubin (0.2 * desvio padrão do logit do PS) "
            "para identificar quantos pares possuem uma distância de propensity score tão grande que sua comparação pode não ser confiável. "
            "É uma sugestão de quantos tratados poderiam ser descartados para aumentar a qualidade do pareamento."
        )

    with tab2:
        st.subheader("Comparação do Balanceamento das Covariáveis")
        selected_var = st.selectbox(
            "Selecione uma Covariável para Análise Detalhada:",
            options=results['covariates'],
            key='balance_covariate_selector' # Chave única para o widget
        )
        if selected_var:
            # --- Guia de Interpretação dos Testes ---
            with st.expander("ℹ️ Como interpretar os testes de balanceamento?"):
                st.markdown("""
                O objetivo é verificar se os grupos de tratamento e controle são estatisticamente semelhantes *antes* e *depois* do pareamento. O ideal é que as diferenças desapareçam (ou diminuam) após o pareamento.

                - **T-test (médias):** Compara se as **médias** de uma variável numérica são iguais entre os dois grupos.
                
                - **Mann-Whitney U (distribuições):** Compara se as **distribuições** de uma variável numérica são as mesmas. É útil quando os dados não seguem uma distribuição normal.
                
                - **Chi-quadrado (proporções):** Compara as **proporções** de uma variável categórica entre os grupos.

                ---
                **Como ler o P-valor:**
                - **P-valor baixo (< 0.05):** Indica desequilíbrio (os grupos são diferentes).
                - **P-valor alto (> 0.05):** Indica equilíbrio (os grupos são semelhantes).

                **Idealmente:** Vemos p-valores baixos *antes* do pareamento e p-valores altos *depois*.
                """)

            # --- Comparação Lado a Lado ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Antes do Pareamento")
                st.dataframe(results['balance_before'][results['balance_before']['Variável'] == selected_var])
                fig_before = generate_single_balance_plot(results['df'], results['treatment_col'], selected_var)
                st.pyplot(fig_before)

            with col2:
                st.markdown("#### Depois do Pareamento")
                st.dataframe(results['balance_after'][results['balance_after']['Variável'] == selected_var])
                fig_after = generate_single_balance_plot(results['df_matched'], results['treatment_col'], selected_var)
                st.pyplot(fig_after)
    
    with tab3:
        st.subheader("Análise SHAP de Explicabilidade do Modelo")
        st.markdown("""
        O gráfico de resumo SHAP abaixo combina a importância das variáveis com seus efeitos. 
        Cada ponto no gráfico é uma observação para uma determinada variável.

        **Como interpretar o gráfico:**
        - **Posição no Eixo Y:** As variáveis são ordenadas da mais importante (topo) para a menos importante (base).
        - **Posição no Eixo X (Valor SHAP):** Mostra o impacto daquela variável na predição do propensity score.
            - Valores **positivos** empurram o score para **cima** (maior probabilidade de tratamento).
            - Valores **negativos** empurram o score para **baixo**.
        - **Cor do Ponto:** Representa o valor original da variável.
            - **Vermelho/Alto:** Valor alto da variável (ex: preço alto).
            - **Azul/Baixo:** Valor baixo da variável.
        
        **Exemplo:** Se a variável `price` estiver no topo e tiver muitos pontos vermelhos à direita do eixo zero, significa que preços altos (vermelho) têm um forte impacto positivo (à direita) no propensity score.
        """)
        st.image(results['shap_summary_plot'], use_column_width=True)

    with tab4:
        st.subheader("Importância das Variáveis (Coeficientes do Modelo)")
        st.markdown("""
        Esta tabela mostra os coeficientes do modelo de regressão logística usado para calcular o propensity score. 
        Eles representam o "peso" ou a "importância" de cada variável na determinação da probabilidade de pertencer ao grupo de tratamento.

        **Como interpretar a tabela:**
        - **Coeficiente:**
            - **Sinal (+/-):** Um coeficiente positivo significa que um aumento na variável aumenta a probabilidade de tratamento. Um negativo, diminui.
            - **Magnitude:** Como as variáveis numéricas foram padronizadas, podemos comparar suas magnitudes. Quanto maior o valor absoluto do coeficiente, maior o impacto da variável no modelo.
        - **P-Valor:** Um p-valor baixo (< 0.05) indica que o coeficiente é estatisticamente significante.
        """)
        st.dataframe(results['feature_importance_df'])

    with tab5:
        st.subheader("Análise dos 10 Piores Pareamentos")
        st.markdown(
            "Estes são os pares com a **maior distância** no Propensity Score. Distâncias grandes podem "
            "indicar a presença de outliers ou áreas de fraca sobreposição entre os grupos. "
            "A coluna `qualidade_do_par` aplica a regra do Caliper para um diagnóstico rápido."
        )
        # Ocultar colunas de índice da exibição principal
        display_cols_worst = [col for col in results['worst_matches_df'].columns if '_index' not in col]
        st.dataframe(results['worst_matches_df'][display_cols_worst])
        
        # --- Análise SHAP Local ---
        st.markdown("---")
        st.subheader("Análise Detalhada de um Par (SHAP)")
        
        pair_to_analyze_worst = st.selectbox(
            "Selecione um par da tabela acima para analisar:",
            options=results['worst_matches_df'].index,
            format_func=lambda x: f"Par {x+1}: {results['worst_matches_df'].loc[x, f'tratado_{results['id_col']}']} vs {results['worst_matches_df'].loc[x, f'controle_{results['id_col']}']}",
            key='worst_pair_selector'
        )
        
        if pair_to_analyze_worst is not None:
            explainer = results['shap_explainer']
            shap_values = results['shap_values']
            X_shap = results['X_shap']
            X_display = results['X_display']
            numerical_vars = results['numerical_vars']

            # Obter índices e posições
            pair_data = results['worst_matches_df'].loc[pair_to_analyze_worst]
            t_idx, c_idx = pair_data['treatment_index'], pair_data['control_index']
            t_pos, c_pos = X_shap.index.get_loc(t_idx), X_shap.index.get_loc(c_idx)

            # --- NOVO: Formatar os dados para exibição ---
            t_display_formatted = X_display.iloc[t_pos, :].copy()
            c_display_formatted = X_display.iloc[c_pos, :].copy()
            for var in numerical_vars:
                if var in t_display_formatted.index:
                    t_display_formatted[var] = f"{t_display_formatted[var]:,.0f}"
                if var in c_display_formatted.index:
                    c_display_formatted[var] = f"{c_display_formatted[var]:,.0f}"

            # Plot Tratado
            st.markdown(f"**Explicação para a unidade tratada ({pair_data[f'tratado_{results['id_col']}']})**")
            st_shap(shap.force_plot(explainer.expected_value, shap_values[t_pos, :], t_display_formatted, link='logit'), height=200)

            # Plot Controle
            st.markdown(f"**Explicação para a unidade de controle ({pair_data[f'controle_{results['id_col']}']})**")
            st_shap(shap.force_plot(explainer.expected_value, shap_values[c_pos, :], c_display_formatted, link='logit'), height=200)

            # --- NOVO: Tabela com valores SHAP numéricos ---
            st.markdown("**Contribuição Numérica de Cada Variável:**")
            with st.expander("Clique para ver os valores SHAP detalhados"):
                # Tratado
                t_shap_series = pd.Series(shap_values[t_pos, :], index=X_shap.columns)
                t_shap_df = pd.DataFrame({'Variável': t_shap_series.index, 'Valor SHAP': t_shap_series.values})
                t_shap_df['Impacto'] = t_shap_df['Valor SHAP'].abs()
                t_shap_df = t_shap_df.sort_values(by='Impacto', ascending=False).drop(columns=['Impacto'])
                
                # Controle
                c_shap_series = pd.Series(shap_values[c_pos, :], index=X_shap.columns)
                c_shap_df = pd.DataFrame({'Variável': c_shap_series.index, 'Valor SHAP': c_shap_series.values})
                c_shap_df['Impacto'] = c_shap_df['Valor SHAP'].abs()
                c_shap_df = c_shap_df.sort_values(by='Impacto', ascending=False).drop(columns=['Impacto'])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Tratado:** `{pair_data[f'tratado_{results['id_col']}']}`")
                    st.dataframe(t_shap_df.style.format({"Valor SHAP": "{:+.4f}"}))
                with col2:
                    st.markdown(f"**Controle:** `{pair_data[f'controle_{results['id_col']}']}`")
                    st.dataframe(c_shap_df.style.format({"Valor SHAP": "{:+.4f}"}))


    with tab6:
        st.subheader("Análise dos 10 Melhores Pareamentos")
        st.markdown(
            "Estes são os pares com a **menor distância** no Propensity Score. Eles representam "
            "os exemplos de pareamentos de maior sucesso, onde os grupos de tratamento e controle "
            "tinham unidades virtualmente idênticas em termos das covariáveis observadas."
        )
        # Ocultar colunas de índice da exibição principal
        display_cols_best = [col for col in results['best_matches_df'].columns if '_index' not in col]
        st.dataframe(results['best_matches_df'][display_cols_best])

        # --- Análise SHAP Local ---
        st.markdown("---")
        st.subheader("Análise Detalhada de um Par (SHAP)")

        pair_to_analyze_best = st.selectbox(
            "Selecione um par da tabela acima para analisar:",
            options=results['best_matches_df'].index,
            format_func=lambda x: f"Par {x+1}: {results['best_matches_df'].loc[x, f'tratado_{results['id_col']}']} vs {results['best_matches_df'].loc[x, f'controle_{results['id_col']}']}",
            key='best_pair_selector'
        )

        if pair_to_analyze_best is not None:
            explainer = results['shap_explainer']
            shap_values = results['shap_values']
            X_shap = results['X_shap']
            X_display = results['X_display']
            numerical_vars = results['numerical_vars']

            # Obter índices e posições
            pair_data = results['best_matches_df'].loc[pair_to_analyze_best]
            t_idx, c_idx = pair_data['treatment_index'], pair_data['control_index']
            t_pos, c_pos = X_shap.index.get_loc(t_idx), X_shap.index.get_loc(c_idx)

            # --- NOVO: Formatar os dados para exibição ---
            t_display_formatted = X_display.iloc[t_pos, :].copy()
            c_display_formatted = X_display.iloc[c_pos, :].copy()
            for var in numerical_vars:
                if var in t_display_formatted.index:
                    t_display_formatted[var] = f"{t_display_formatted[var]:,.0f}"
                if var in c_display_formatted.index:
                    c_display_formatted[var] = f"{c_display_formatted[var]:,.0f}"

            # Plot Tratado
            st.markdown(f"**Explicação para a unidade tratada ({pair_data[f'tratado_{results['id_col']}']})**")
            st_shap(shap.force_plot(explainer.expected_value, shap_values[t_pos, :], t_display_formatted, link='logit'), height=200)

            # Plot Controle
            st.markdown(f"**Explicação para a unidade de controle ({pair_data[f'controle_{results['id_col']}']})**")
            st_shap(shap.force_plot(explainer.expected_value, shap_values[c_pos, :], c_display_formatted, link='logit'), height=200)

            # --- NOVO: Tabela com valores SHAP numéricos ---
            st.markdown("**Contribuição Numérica de Cada Variável:**")
            with st.expander("Clique para ver os valores SHAP detalhados"):
                # Tratado
                t_shap_series = pd.Series(shap_values[t_pos, :], index=X_shap.columns)
                t_shap_df = pd.DataFrame({'Variável': t_shap_series.index, 'Valor SHAP': t_shap_series.values})
                t_shap_df['Impacto'] = t_shap_df['Valor SHAP'].abs()
                t_shap_df = t_shap_df.sort_values(by='Impacto', ascending=False).drop(columns=['Impacto'])
                
                # Controle
                c_shap_series = pd.Series(shap_values[c_pos, :], index=X_shap.columns)
                c_shap_df = pd.DataFrame({'Variável': c_shap_series.index, 'Valor SHAP': c_shap_series.values})
                c_shap_df['Impacto'] = c_shap_df['Valor SHAP'].abs()
                c_shap_df = c_shap_df.sort_values(by='Impacto', ascending=False).drop(columns=['Impacto'])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Tratado:** `{pair_data[f'tratado_{results['id_col']}']}`")
                    st.dataframe(t_shap_df.style.format({"Valor SHAP": "{:+.4f}"}))
                with col2:
                    st.markdown(f"**Controle:** `{pair_data[f'controle_{results['id_col']}']}`")
                    st.dataframe(c_shap_df.style.format({"Valor SHAP": "{:+.4f}"}))

    # --------------------------------------------------------------------------
    # Lógica de Download (na barra lateral)
    # --------------------------------------------------------------------------
    summary_to_download = {
        "Sumario_Pareamento": results['summary_df'],
        "Importancia_Variaveis": results['feature_importance_df'],
        "Piores_Pareamentos": results['worst_matches_df'],
        "Melhores_Pareamentos": results['best_matches_df'],
        "Balanceamento_Antes": results['balance_before'],
        "Balanceamento_Depois": results['balance_after'],
    }
    summary_excel = to_excel(summary_to_download)
    
    id_col_name = results['id_col']
    treatment_ids = results['df'].loc[results['matched_pairs']['treatment_index'], id_col_name].values
    control_ids = results['df'].loc[results['matched_pairs']['control_index'], id_col_name].values
    
    paired_ids_df = pd.DataFrame({
        f'tratado_{id_col_name}': treatment_ids,
        f'controle_{id_col_name}': control_ids
    })
    paired_ids_df.index.name = 'par_id'
    paired_ids_df = paired_ids_df.reset_index()

    csv_data = paired_ids_df.to_csv(index=False).encode('utf-8')

    st.sidebar.header("4. Download dos Resultados")
    st.sidebar.download_button(
        label="Baixar Diagnósticos (Excel)",
        data=summary_excel,
        file_name="diagnosticos_psm.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.sidebar.download_button(
        label="Baixar Pares Mapeados (CSV)",
        data=csv_data,
        file_name='pares_mapeados.csv',
        mime='text/csv',
    )

# Mensagem inicial se nenhum dado foi carregado
if df is None:
    st.info("Aguardando o upload de um arquivo CSV para começar.")
