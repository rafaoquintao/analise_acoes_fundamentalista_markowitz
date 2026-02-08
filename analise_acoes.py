import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List
import fundamentus

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
            print(f'Buscando indicadores: {t}...')
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
            print(f"Erro na ingestão de {t}: {e}")
            
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
    print("Baixando histórico de preços...")
    # Coleta 2 anos de histórico de fechamento ajustado
    df_precos = yf.download(tickers, period="2y", progress=False)['Close']
    
    # Tratamento de dados (Data Cleaning)
    # ffill preenche gaps de dias sem negociação (comum em FIIs e ETFs)
    # bfill preenche gaps para ativos novos que não têm 2 anos de histórico
    df_limpo = df_precos.ffill().bfill()
    df_limpo = df_precos.dropna(axis=1, how="all") # Remove colunas que ficaram totalmente vazias (ativos que falharam no download)

    
    os.makedirs("data_lake/silver", exist_ok=True)
    df_limpo.to_parquet("data_lake/silver/precos_limpos.parquet")
    print("Camada Silver: Preços históricos normalizados.")
    return df_limpo

def analisar_dados_financeiros(df_precos: pd.DataFrame, num_portfolios: int = 5000) -> pd.DataFrame:
    """
    Calcula a Fronteira Eficiente e identifica a melhor alocação de ativos.
    
    Args:
        df_precos: DataFrame da camada Silver com preços limpos.
        num_portfolios: Quantidade de simulações.
    """
    retornos = np.log(df_precos / df_precos.shift(1)).dropna()
    media_retorno = retornos.mean() * 252
    matriz_cov = retornos.cov() * 252
    tickers = df_precos.columns

    resultados = np.zeros((3, num_portfolios))
    lista_pesos = []

    print(f"Otimizando carteira para {len(tickers)} ativos...")
    for i in range(num_portfolios):
        pesos = np.random.random(len(tickers))
        pesos /= np.sum(pesos)
        lista_pesos.append(pesos)
        
        retorno_p = np.sum(media_retorno * pesos)
        volatilidade_p = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos)))
        
        resultados[0,i] = retorno_p
        resultados[1,i] = volatilidade_p
        resultados[2,i] = retorno_p / volatilidade_p # Sharpe Ratio

    # 1. Gerar DataFrame da Fronteira Eficiente (Gráfico de dispersão)
    df_gold_simulacao = pd.DataFrame(resultados.T, columns=['Retorno', 'Volatilidade', 'Sharpe_Ratio'])
    df_gold_simulacao.to_parquet("data_lake/gold/simulacao_markowitz.parquet", index=False)

    # 2. Identificar o Melhor Portfólio (Sharpe Máximo) e seus pesos
    indice_melhor_sharpe = df_gold_simulacao['Sharpe_Ratio'].idxmax()
    melhores_pesos = lista_pesos[indice_melhor_sharpe]

    df_pesos_ideais = pd.DataFrame({
        "ticker": tickers,
        "Peso_Ideal": [round(p * 100, 2) for p in melhores_pesos]
    }).sort_values(by="Peso_Ideal", ascending=False)

    # Salva o arquivo que o Dashboard estava procurando
    os.makedirs("data_lake/gold", exist_ok=True)
    df_pesos_ideais.to_parquet("data_lake/gold/alocacao_otimizada.parquet", index=False)
    
    print("Sucesso: Tabela de alocação otimizada gerada na Gold!")
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
            print("Falha: Arquivo Bronze está vazio.")
            return False
        # Check 2: Colunas essenciais presentes?
        colunas_essenciais = ["ticker", "data_coleta", "p_l", "p_vp",
                               "dividend_yield", "setor", "tipo", "roe",
                               "div_patrimonial", "liq_diaria", "patr_liquido"]
        for col in colunas_essenciais:
            if col not in df.columns:
                print(f"Falha: Coluna essencial '{col}' ausente.")
                return False
        # Check 3: Temos valores nulos nos campos críticos?
        nulos = df['p_l'].isnull().sum()
        if nulos > 0:
            print(f"Aviso: Encontrados {nulos} tickers com P/L nulo.")
        
        print(" Data quality check: Aprovado!")
        return True
    
    except Exception as e:
        print(f"Falha na validação da camada Bronze: {e}")
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

    return df[['ticker', 'score', 'p_l', 'dividend_yield']].sort_values(by='score', ascending=False)

def gerar_carteira_recomendada(df_universo: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica os filtros fundamentalistas rígidos para selelcionar os 20 tickers 'Elite '   
    """
    df = df_universo.copy()
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
    print("Buscando dados do Fundamentus...")
    # Retorna um DataFrame onde o índice é o ticker
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
    return df

def realizar_backtest_comparativo( tickers_elite: List[str], minha_carteira: List[str]):
    """
    Compara o desempenho: Carteira Elite vs. Minha Carteira vs. IBOVESPA.
    """
    print("Iniciando Backtest Comparativo Triplo (12 meses)...")
    
    # Unificamos todos os tickers para um único download (otimiza performance)
    todos_tickers = list(set(tickers_elite + minha_carteira + ['^BVSP']))
    df_precos = yf.download(todos_tickers, period="1y", progress=False)['Close']
    
    # Tratamento inicial
    df_precos = df_precos.ffill()
    
    # Normalização Base 100 (Evolução relativa)
    df_norm = (df_precos / df_precos.iloc[0]) * 100
    
    # Cálculo das Médias (Estratégias Equipesadas)
    df_norm['CARTEIRA_ELITE'] = df_norm[tickers_elite].mean(axis=1)
    df_norm['MINHA_CARTEIRA'] = df_norm[minha_carteira].mean(axis=1)
    
    # Selecionamos apenas as colunas de comparação
    df_final = df_norm[['^BVSP', 'CARTEIRA_ELITE', 'MINHA_CARTEIRA']]
    
    # Salva na Gold
    df_final.to_parquet("data_lake/gold/backtest_performance.parquet")
    print(" Backtest Triplo gerado com sucesso!")

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
    print(f" Camada Ouro: Carteira Elite gerada com {len(df_elite)} ativos!")

    
    # Processamento da Carteira Atual
    df_bronze = buscar_dados_financeiros(carteira_atual)

    # validação de qualidade dos dados
    data_hoje = datetime.now().strftime('%Y%m%d')
    caminho_bronze = f"data_lake/bronze/fundamentalista_{data_hoje}.parquet"
    print(f"Iniciando validação de qualidade: {caminho_bronze}...")

    if validar_camada_bronze(caminho_bronze):

        df_ranking = gerar_ranking_fundamentalista(df_bronze)
        os.makedirs("data_lake/gold", exist_ok=True)
        df_ranking.to_parquet("data_lake/gold/ranking_fundamentalista.parquet", index=False)
        
        # Processamento de preços e Markowitz
        df_silver = processar_dados_financeiros(carteira_atual)
        df_gold = analisar_dados_financeiros(df_silver)
        
        # Análise técnica (timing para ativos da atual)
        if not df_bronze.empty:
            print("Calculando Timing (RSI) para carteira atual...")
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

            print("Pipeline de Análise de Ações concluído com sucesso!")
    else:
        print("Pipeline interrompido por falha na qualidade dos dados.")

