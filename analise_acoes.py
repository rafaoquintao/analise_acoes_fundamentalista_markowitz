import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List
import fundamentus 
import logging
import sys
import random
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

np.random.seed(42)
random.seed(42)

# 1. Garante a pasta
os.makedirs("logs", exist_ok=True)
log_path = "logs/pipeline_analise_acoes.log"

# 2. Configuração com Flush forçado
# O segredo aqui é que vamos usar um StreamHandler apontando para o arquivo
# ou garantir que o FileHandler não use buffer.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Handler para o arquivo (Modo 'w' para sobrescrever e ver se funciona)
file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Handler para o console (GitHub Actions Logs)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def buscar_dados_financeiros(tickers: List[str]) -> pd.DataFrame:
    """
    Realiza a ingestão de indicadores fundamentelista (Camada Bronze).
    
    Args:
        tickers (List[str]): Lista de ações da carteira.
    Returns:
        pd.DataFrame: Dados brutos coletados.
    """
    lista_dados = []
    for t in tickers:
        try:
            logging.info(f'Buscando indicadores: {t}...')
            ativo = yf.Ticker(t)
            info = ativo.info
            
            # Coleta segura de dados usando .get() para evitar KeyErrors
            dados_ativo = {
                "ticker": t,
                "p_l": info.get("forwardPE"),
                "p_vp": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "roe": info.get("returnOnEquity"),
                "div_patrimonial": info.get("debtToEquity", 0) / 100 if info.get("debtToEquity") else 0,
                "liq_diaria": info.get("averageDailyVolume10Day"),
                "patr_liquido": info.get("totalCash"),
                "data_coleta": datetime.now().strftime("%Y-%m-%d"),
                "setor": info.get('sector'),
                "tipo": "ETF" if info.get("quoteType") == "ETF" else 
                        ("FII" if "11" in t and info.get("sector") == "Real Estate" else "Ação")
            }
            lista_dados.append(dados_ativo)
        except Exception as e:
            logging.error(f"Erro na ingestão de {t}: {e}")
            
    df = pd.DataFrame(lista_dados)
    os.makedirs("data_lake/bronze", exist_ok=True)
    df.to_parquet(f"data_lake/bronze/fundamentalista_{datetime.now().strftime('%Y%m%d')}.parquet", index=False)
    return df

def processar_dados_financeiros(tickers: List[str]) -> pd.DataFrame:
    """
    Coleta e limpa séries temporais de preços( Camada Silver).

    Args: 
        tickers (List[str]): Lista de ações da carteira.
    Returns:
        pd.DataFrame: Preços de fechamento limpos e preenchidos.
    """
    logging.info("Baixando histórico de preços...")
    # Coleta 2 anos de histórico de fechamento ajustado
    df_precos = yf.download(tickers, period="2y", progress=False)['Close']
    
    # Tratamento de dados (Data Cleaning)
    # ffill preenche gaps de dias sem negociação (comum em FIIs e ETFs)
    # bfill preenche gaps para ativos novos que não têm 2 anos de histórico
    df_limpo = df_precos.ffill().bfill()
    df_limpo = df_precos.dropna(axis=1, how="all") # Remove colunas que ficaram totalmente vazias (ativos que falharam no download)

    
    os.makedirs("data_lake/silver", exist_ok=True)
    df_limpo.to_parquet("data_lake/silver/precos_limpos.parquet")
    logging.info("Camada Silver: Preços históricos normalizados.")
    return df_limpo

def analisar_dados_financeiros(df_precos: pd.DataFrame) -> pd.DataFrame:
    """
    Otimização Numérica (Markowitz) com Regularização L2 para maior estabilidade.
    """
    logging.info(f"Otimizando carteira profissional para {len(df_precos.columns)} ativos...")

    # 1. Modelagem de Retorno e Risco
    # Usa o 'mean_historical_return' e o 'CovarianceShrinkage' para dados mais robustos
    mu = expected_returns.mean_historical_return(df_precos)
    S = risk_models.CovarianceShrinkage(df_precos).ledoit_wolf()

    # 2. Configuração do Otimizador
    ef = EfficientFrontier(mu, S)

    # 3. REGULARIZAÇÃO L2 
    # Gamma=0.1 evita que o modelo concentre tudo em poucos ativos
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)

    # 4. Busca pelo Sharpe Máximo
    try:
        _ = ef.max_sharpe()
        # Limpa pesos irrelevantes (menores que 1%)
        cleaned_weights = ef.clean_weights(cutoff=0.01)
        logging.info("Otimização concluída via Programação Quadrática (Max Sharpe).")
    except Exception as e:
        logging.warning(f"Falha no Max Sharpe ({e}). Calculando Variância Mínima...")
        _ = ef.min_volatility()
        cleaned_weights = ef.clean_weights()

    # 5. Formatação dos Resultados para a Camada Gold
    df_pesos_ideais = pd.DataFrame(
        list(cleaned_weights.items()), 
        columns=["ticker", "Peso_Ideal"]
    ).sort_values(by="Peso_Ideal", ascending=False)
    
    # Converte para escala 0-100
    df_pesos_ideais["Peso_Ideal"] = (df_pesos_ideais["Peso_Ideal"] * 100).round(2)

    # Persistência
    os.makedirs("data_lake/gold", exist_ok=True)
    df_pesos_ideais.to_parquet("data_lake/gold/alocacao_otimizada.parquet", index=False)
    
    return df_pesos_ideais

def validar_camada_bronze(caminho_arquivo: str) -> bool:
    """
    Realiza um check de qualidade nos dados recém coletados
    Args:
        caminho_arquivo (str): Caminho do arquivo Parquet na camada Bronze.
    Returns:
        bool: True se os dados passaram nos testes
    """
    try:
        df = pd.read_parquet(caminho_arquivo)
        # Check 1: O arquivo está vazio?
        if df.empty:
            logging.error("Falha: Arquivo Bronze está vazio.")
            return False
        # Check 2: Colunas essenciais presentes?
        colunas_essenciais = ["ticker", "data_coleta", "p_l", "p_vp",
                               "dividend_yield", "setor", "tipo", "roe",
                               "div_patrimonial", "liq_diaria", "patr_liquido"]
        for col in colunas_essenciais:
            if col not in df.columns:
                logging.error(f"Falha: Coluna essencial '{col}' ausente.")
                return False
        # Check 3: Temos valores nulos nos campos críticos?
        nulos = df['p_l'].isnull().sum()
        if nulos > 0:
            logging.warning(f"Aviso: Encontrados {nulos} tickers com P/L nulo.")
        
        logging.info(" Data quality check: Aprovado!")
        return True
    
    except Exception as e:
        logging.error(f"Falha na validação da camada Bronze: {e}")
        return False

def gerar_ranking_fundamentalista(df_bronze: pd.DataFrame) -> pd.DataFrame:
    """
    Cria um score de 0 a 10 usando ranking por percentil.
    """
    df = df_bronze.copy()

    # 1. Tratamento de nulos para não quebrar o ranking
    df['dividend_yield'] = df['dividend_yield'].fillna(0)
    # Para P/L, quanto menor melhor. Se for nulo ou negativo (prejuízo), colocamos um valor alto.
    df['p_l'] = df['p_l'].apply(lambda x: x if x > 0 else 999)

    # 2. Criar Rankings de 0 a 1 (Percentis)
    # rank(pct=True) transforma os valores em uma escala de 0 a 1
    df['rank_dy'] = df['dividend_yield'].rank(pct=True)
    df['rank_pl'] = df['p_l'].rank(pct=True, ascending=False) # Menor P/L ganha rank maior

    # 3. Score Final (Peso 50% para cada indicador)
    # Multiplicamos por 10 para a escala de 0 a 10
    df['score'] = ((df['rank_dy'] * 0.5) + (df['rank_pl'] * 0.5)) * 10
    
    # 4. Arredondamento para ficar "bonito" no dashboard
    df['score'] = df['score'].round(2)

    return df[['ticker', 'score', 'p_l', 'dividend_yield','preco']].sort_values(by='score', ascending=False)

def gerar_carteira_recomendada(df_universo: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica os filtros fundamentalistas rígidos para selelcionar os 20 tickers 'Elite '   
    """
    logging.info("Gerando carteira recomendada (Elite Fundamentalista)...")
    df = df_universo.copy()
    if df.empty:
        logging.error("Falha: Arquivo Universo B3 está vazio.")
        return pd.DataFrame()

    # Aplicando os filtros
    df_filtered = df[
        (df['p_l'] > 0) & (df['p_l'] < 15) &
        (df['p_vp'] < 1.5) &
        (df['div_patrimonial'] < 2) &
        (df['dividend_yield'] > 0.05) & (df['dividend_yield'] < 0.2) &
        (df['roe'] > 0.10) &
        (df['liq_diaria'] > 100000) &
        (df['patr_liquido'] > 100000000)
    ]
    recomendada = df_filtered.sort_values(by = 'roe', ascending = False).head(20)
    logging.info(f"Carteira Elite gerada com {len(recomendada)} ativos após aplicação dos filtros.")
    return recomendada

def analise_tecnica_rsi(ticker: str):
    """
    Calcula o Indice de Força Relativa (RSI) para identificar janelas de oportunidade.
    RSI < 30: Sobrevendido (Oportunidade de compra)
    RSI > 70: Sobrecomprado (Aguardar/Possível venda)
    """
    df = yf.download(ticker, period = "3mo", progress = False)
    # Caso o download falhe ou venha vazio
    if df.empty:
        logging.error(f"Falha ao baixar dados para {ticker}. Retornando RSI neutro.")
        return 50.0, "Erro nos Dados"
    
    delta = df['Close'].diff()
    # Separando os ganhos e perdas
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Cálculo da Média Suavizada de Wilder (alpha = 1/periodo)
    # O parâmetro 'com' (center of mass) é definido como period - 1 para equivaler ao Wilder
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()


    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    rsi_atual = float(df['RSI'].iloc[-1])

    status = "Compra (Sobrevendido)" if rsi_atual < 30 else \
             "Venda (Sobrecomprado)" if rsi_atual > 70 else "Neutro"
    return round(rsi_atual, 2), status

def buscar_dados_fundamentus() -> pd.DataFrame:
    """
    Busca TODOS  os ativos da B3 com seus indicadores fundamentalistas.
    Camada Bronze - Discovery Automático.
    """
    logging.info("Buscando dados do Fundamentus...")
    # Retorna um DataFrame onde o índice é o ticker
    try:    
        df = fundamentus.get_resultado()
        df = df.reset_index()
        # Padronizando as colunas para o filtro (mapping Fundamentus -> Nosso padrão)
        df = df.rename(columns={
            'papel': 'ticker', 
            'pl': 'p_l',
            'pvp': 'p_vp',
            'dy': 'dividend_yield',
            'divbpatr': 'div_patrimonial',
            'liq2m': 'liq_diaria',
            'patrliq': 'patr_liquido',
            'roe': 'roe',
        })
        # Adicionando o sufixo .SA para compatibilidade com yfinance se precisar
        df['ticker'] = df['ticker'].apply(lambda x: x + '.SA')

        os.makedirs("data_lake/bronze", exist_ok = True)
        df.to_parquet(f"data_lake/bronze/universo_b3.parquet", index =  False)
        logging.info("Sucesso: Dados do Fundamentus salvos na camada Bronze!")
    except Exception as e:
        logging.error(f"Erro ao buscar dados do Fundamentus: {e}")
    return df

def realizar_backtest_comparativo(tickers_elite: List[str], minha_carteira: List[str]):
    logging.info("Iniciando Backtest Comparativo Triplo (12 meses)...")
    
    # 1. Download
    todos_tickers = list(set(tickers_elite + minha_carteira + ['^BVSP']))
    df_precos = yf.download(todos_tickers, period="2y", progress=False)['Close']
    
    # 2. Tratamento anti-erro: Remove colunas totalmente vazias e preenche buracos
    df_precos = df_precos.dropna(axis=1, how='all').ffill().bfill()
    
    # 3. Normalização Base 100 
    # Usamos .iloc[0] mas garantimos que não seja zero/NaN para não explodir o cálculo
    df_norm = (df_precos / df_precos.iloc[0]) * 100
    
    # 4. Criamos um DataFrame NOVO só para o resultado final
    # Isso evita que colunas de tickers individuais "vazem" para o gráfico
    df_final = pd.DataFrame(index=df_norm.index)
    
    # Calculamos as médias apenas com os tickers que realmente foram baixados
    tickers_elite_ok = [t for t in tickers_elite if t in df_norm.columns]
    minha_carteira_ok = [t for t in minha_carteira if t in df_norm.columns]
    
    df_final['IBOVESPA'] = df_norm['^BVSP']
    df_final['CARTEIRA_ELITE'] = df_norm[tickers_elite_ok].mean(axis=1)
    df_final['MINHA_CARTEIRA'] = df_norm[minha_carteira_ok].mean(axis=1)
    
    # 5. Dropna final para garantir que o gráfico não tenha vácuo no início
    df_final = df_final.dropna()

    # Salva na Gold
    df_final.to_parquet("data_lake/gold/backtest_performance.parquet")
    logging.info(f"Backtest Triplo gerado com sucesso! Colunas: {df_final.columns.tolist()}")

# --- ORQUESTRAÇÃO ---
if __name__ == "__main__":
   # Definindo a carteira atual
    carteira_atual = [
        'BBAS3.SA', 'CASH3.SA', 'CMIG4.SA','EMBJ3.SA','GPUS11.SA','ISAE4.SA',
        'ITSA4.SA', 'ITUB4.SA', 'LAVV3.SA', 'PETR4.SA', 'PINE4.SA','POMO3.SA',
        'ROXO34.SA', 'VALE3.SA', 'VBBR3.SA', 'WEGE3.SA', 'NVDA','BUD', 'KO', 'AMZN',
        'NU', 'KO', 'DIS' 
    ]
    carteira_fii = [ 
        'CXCO11.SA', 'HGBS11.SA',
        'HGLG11.SA', 'XPCI11.SA','XPML11.SA' 
        ]
    
    # Discovery Automático do Universo de Ações da B3
    universo_df = buscar_dados_fundamentus()

    # Filtro Elite Fundamentalista
    df_elite = gerar_carteira_recomendada(universo_df)
    df_elite.to_parquet("data_lake/gold/carteira_elite.parquet", index = False)
    logging.info(f" Camada Ouro: Carteira Elite gerada com {len(df_elite)} ativos!")

    
    # Processamento da Carteira Atual
    df_bronze = buscar_dados_financeiros(carteira_atual)

    # validação de qualidade dos dados
    data_hoje = datetime.now().strftime('%Y%m%d')
    caminho_bronze = f"data_lake/bronze/fundamentalista_{data_hoje}.parquet"
    logging.info(f"Iniciando validação de qualidade: {caminho_bronze}...")

    if validar_camada_bronze(caminho_bronze):

        df_ranking = gerar_ranking_fundamentalista(df_bronze)
        os.makedirs("data_lake/gold", exist_ok=True)
        df_ranking.to_parquet("data_lake/gold/ranking_fundamentalista.parquet", index=False)
        
        # Processamento de preços e Markowitz
        df_silver = processar_dados_financeiros(carteira_atual)
        df_gold = analisar_dados_financeiros(df_silver)
        
        # Análise técnica (timing para ativos da atual)
        if not df_bronze.empty:
            logging.info("Calculando Timing (RSI) para carteira atual...")
            resultados_rsi = []
            for t in df_bronze['ticker'].tolist():
                rsi_valor, rsi_status = analise_tecnica_rsi(t)
                resultados_rsi.append({
                    "ticker": t,
                    "rsi": rsi_valor,
                    "status": rsi_status
                })
            df_timing = pd.DataFrame(resultados_rsi)
            df_timing.to_parquet("data_lake/gold/timing_elite.parquet", index = False)
            
            realizar_backtest_comparativo(df_elite['ticker'].tolist(), carteira_atual)

            logging.info("Pipeline de Análise de Ações concluído com sucesso!")
    else:
        logging.warning("Pipeline interrompido por falha na qualidade dos dados.")
    logging.info("--- Pipeline Finalizado com Sucesso ---")
    logging.shutdown() # Isso força o Python a fechar o arquivo e salvar tudo no disco
 