import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Dashboard Financeiro Inteligente",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CAMINHOS DOS ARQUIVOS (DATA LAKE) ---
PATH_RANKING = "data_lake/gold/ranking_fundamentalista.parquet"
PATH_PESOS = "data_lake/gold/alocacao_otimizada.parquet"
PATH_UNIVERSO = "data_lake/bronze/universo_b3.parquet"
PATH_TIMING = "data_lake/gold/timing_elite.parquet"
PATH_BACKTEST = "data_lake/gold/backtest_performance.parquet"
PATH_CUSTODIA = "data_lake/bronze/meus_ativos.csv"

# --- DICIONÁRIO DE CONVERSÃO STOCKS -> BDRs ---
MAPEAMENTO_BDR = {
    'NVDA': 'NVDC34', 'AMZN': 'AMZO34', 'DIS': 'DISB34',
    'KO': 'COCA34', 'BUD': 'ABUD34', 'F': 'FDMO34',
    'NU': 'ROXO34', 'AAPL': 'AAPL34', 'GOOGL': 'GOGL34'
}

st.title("📊 Gestão de Portfólio e Stock Discovery")
st.markdown("---")

# Verificação de arquivos críticos
if os.path.exists(PATH_RANKING) and os.path.exists(PATH_PESOS):
    # Carregamento de dados Gold
    df_rank = pd.read_parquet(PATH_RANKING)
    df_peso = pd.read_parquet(PATH_PESOS)
    
    # --- PROCESSAMENTO DE CUSTÓDIA E REBALANCEAMENTO ---
    if os.path.exists(PATH_CUSTODIA):
        df_custodia = pd.read_csv(PATH_CUSTODIA)
        df_custodia['ticker'] = df_custodia['ticker'].str.strip().str.upper()
        
        # Converte tickers americanos para BDRs antes do merge
        df_custodia['ticker'] = df_custodia['ticker'].apply(lambda x: MAPEAMENTO_BDR.get(x, x))
        
        # Merge com preços atuais da camada Gold
        df_rebal = pd.merge(df_custodia, df_rank[['ticker', 'preco']], on='ticker', how='left')
        df_rebal['preco'] = df_rebal['preco'].fillna(0)
        
        # Cálculo de Valor de Mercado
        df_rebal['Valor_Total_Ativo'] = df_rebal['quantidade'].astype(float) * df_rebal['preco']
        total_patrimonio = df_rebal['Valor_Total_Ativo'].sum()
        
        # Cálculo de Peso Atual
        df_rebal['Peso_Atual'] = (df_rebal['Valor_Total_Ativo'] / total_patrimonio * 100) if total_patrimonio > 0 else 0
        
        # Merge final com Markowitz
        df_final = pd.merge(df_peso, df_rebal[['ticker', 'Peso_Atual', 'quantidade']], on='ticker', how='outer').fillna(0)
    else:
        st.warning(f"Arquivo {PATH_CUSTODIA} não encontrado. Usando pesos zerados.")
        df_final = df_peso.copy()
        df_final['Peso_Atual'] = 0.0
        total_patrimonio = 0

    df_final['Diferenca'] = df_final['Peso_Ideal'] - df_final['Peso_Atual']
    df_total = pd.merge(df_final, df_rank, on='ticker', how='inner')

    # --- VISÃO 1: REBALANCEAMENTO E APORTES ---
    st.header("1. ⚖️ Rebalanceamento e Simulação de Aporte")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Patrimônio Total", f"R$ {total_patrimonio:,.2f}")
    
    # Identifica maiores gaps
    ticker_compra = df_final.loc[df_final['Diferenca'].idxmax(), 'ticker']
    ticker_venda = df_final.loc[df_final['Diferenca'].idxmin(), 'ticker']
    m2.metric("Aportar em (Maior Gap)", ticker_compra)
    m3.metric("Reduzir em (Maior Excesso)", ticker_venda)

    col_sim, col_graph = st.columns([1.2, 2])

    with col_sim:
        st.subheader("Simulador de Aporte")
        valor_aporte = st.number_input("Valor para investir (R$):", min_value=0.0, value=1000.0, step=100.0)
        
        df_compra_sug = df_final[df_final['Diferenca'] > 0].copy()
        if not df_compra_sug.empty:
            total_gap = df_compra_sug['Diferenca'].sum()
            df_compra_sug['Alocacao_R$'] = (df_compra_sug['Diferenca'] / total_gap) * valor_aporte
            df_compra_sug = pd.merge(df_compra_sug, df_rank[['ticker', 'preco']], on='ticker')
            
            # Arredondamento: Inteiro para B3, Float para Stocks/BDRs se necessário
            df_compra_sug['Qtd_Sugerida'] = (df_compra_sug['Alocacao_R$'] / df_compra_sug['preco']).astype(int)
            
            st.write(f"Com R$ {valor_aporte:,.2f}, compre:")
            st.dataframe(df_compra_sug.query("Qtd_Sugerida > 0")[['ticker', 'Qtd_Sugerida', 'Alocacao_R$']], use_container_width=True)
        else:
            st.info("Carteira já está equilibrada.")

    with col_graph:
        fig_rebal = px.bar(
            df_final, x='ticker', y='Diferenca',
            title="Desvio do Ideal (Peso Ideal - Peso Atual)",
            color='Diferenca', color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_rebal, use_container_width=True)

    # --- VISÃO 2: MATRIZ DE DECISÃO ---
    st.divider()
    st.header("2. 🎯 Matriz de Decisão Fundamentalista")
    
    fig_decisao = px.scatter(
        df_total, x='score', y='Peso_Ideal',
        size='dividend_yield', text='ticker', color='Diferenca',
        color_continuous_scale='RdYlGn',
        title="Score Fundamentos vs Alocação Sugerida (Cor = Necessidade de Compra)",
        labels={'score': 'Saúde Financeira (Score)', 'Peso_Ideal': '% Alocação Markowitz'}
    )
    st.plotly_chart(fig_decisao, use_container_width=True)

    # --- VISÃO 3: DISCOVERY & TIMING ---
    st.divider()
    st.header("3. 🔍 Discovery: Meus Ativos B3 & Timing")

    if os.path.exists(PATH_UNIVERSO) and os.path.exists(PATH_TIMING):
        df_universo = pd.read_parquet(PATH_UNIVERSO)
        df_timing = pd.read_parquet(PATH_TIMING)
        
        # Merge para mostrar fundamentos + timing (RSI)
        df_compra = pd.merge(df_universo, df_timing, on='ticker')
        
        st.write("Análise de Momento de Compra (RSI):")

        def color_status(val):
            color = 'green' if 'Compra' in val else 'red' if 'Venda' in val else 'white'
            return f'background-color: {color}; color: black'
        
        st.dataframe(
            df_compra[['ticker', 'p_l', 'roe', 'dividend_yield', 'rsi', 'status']]
            .style.applymap(color_status, subset=['status']),
            use_container_width=True
        )

else:
    st.error("Erro: Arquivos da camada Gold não encontrados. Execute o pipeline de dados primeiro.")

# --- VISÃO 4: BACKTEST ---
st.divider()
st.header("📈 Backtest: Performance Histórica")

if os.path.exists(PATH_BACKTEST):
    df_b = pd.read_parquet(PATH_BACKTEST)
    # Garante que os nomes fiquem iguais aos do seu README
    df_b.columns = ['IBOVESPA', 'ESTRATÉGIA ELITE', 'MINHA CARTEIRA']
    
    fig_b = px.line(
        df_b, 
        labels={'value': 'Evolução (Base 100)', 'index': 'Data'},
        title="Comparativo de Retorno Acumulado (Últimos 24 meses)",
        color_discrete_map={
            'IBOVESPA': '#808080',      # Cinza
            'CARTEIRA_ELITE': '#0000FF', # Azul
            'MINHA_CARTEIRA': '#00FF00'  # Verde
        }
    )
    
    # Melhora a legenda e eixos
    fig_b.update_layout(hovermode="x unified")
    st.plotly_chart(fig_b, use_container_width=True)
else:
    st.info("Execute o Backtest no script principal para visualizar os resultados aqui.")

st.sidebar.markdown("### ⚙️ Configurações")
st.sidebar.write("Os dados são atualizados conforme a execução do pipeline nas camadas Bronze, Silver e Gold.")