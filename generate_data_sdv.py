# --- Importações ---
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import time

def generate_data_with_sdv(
    n_rows: int = 1_000_000, 
    source_file: str = 'psm/data/groupon_sintetico.csv', 
    output_file: str = 'psm/data/groupon_sdv_1M.csv'
) -> None:
    """
    Gera um grande dataset sintético usando a biblioteca SDV.

    O processo consiste em:
    1. Carregar os dados de origem.
    2. Aprender um modelo estatístico (GaussianCopula) a partir dos dados.
    3. Gerar (sample) um novo DataFrame com o número de linhas desejado.
    4. Salvar o resultado em um novo arquivo CSV.

    Args:
        n_rows (int): O número de linhas a serem geradas no dataset sintético.
        source_file (str): O caminho para o arquivo CSV de origem.
        output_file (str): O caminho onde o novo arquivo CSV será salvo.
    """
    print("--- Geração de Dados Sintéticos com SDV ---")
    
    # --- 1. Carregar Dados Reais ---
    try:
        real_data = pd.read_csv(source_file)
        print(f"Arquivo fonte '{source_file}' carregado com sucesso.")
    except FileNotFoundError:
        print(f"Erro: Arquivo fonte '{source_file}' não encontrado.")
        return

    # --- 2. Configurar Metadados ---
    # A SDV espera um objeto de metadados que descreva a estrutura dos dados.
    # Usamos a detecção automática e definimos a chave primária manualmente.
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    metadata.set_primary_key(column_name='deal_id')

    # --- 3. Treinar o Modelo Sintetizador ---
    # O GaussianCopulaSynthesizer é eficaz em aprender correlações entre variáveis.
    synthesizer = GaussianCopulaSynthesizer(metadata)
    
    print("Treinando o modelo SDV com os dados reais... (Isso pode levar alguns minutos)")
    start_train_time = time.time()
    synthesizer.fit(real_data)
    end_train_time = time.time()
    print(f"Treinamento concluído em {end_train_time - start_train_time:.2f} segundos.")

    # --- 4. Gerar Novos Dados (Amostragem) ---
    print(f"Gerando {n_rows:,} novas linhas... (Isso também pode levar um tempo)")
    start_sample_time = time.time()
    synthetic_data = synthesizer.sample(num_rows=n_rows)
    end_sample_time = time.time()
    print(f"Geração de dados concluída em {end_sample_time - start_sample_time:.2f} segundos.")

    # --- 5. Salvar o Arquivo ---
    print(f"Salvando o arquivo em '{output_file}'...")
    synthetic_data.to_csv(output_file, index=False)

    print("\\n---")
    print(f"Dataset sintético '{output_file}' com {len(synthetic_data):,} linhas criado com sucesso!")
    print("Cabeçalho do novo dataset:")
    print(synthetic_data.head())


if __name__ == '__main__':
    # Nota: A instalação e execução da SDV pode consumir bastante RAM.
    # Se encontrar problemas de memória, considere gerar um número menor de linhas.
    generate_data_with_sdv()
