import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Dashboard Financeiro Intelligence",
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

def normalizar_ticker(t):
    """Ajusta tickers para o padrão da Gold (com .SA e BDRs)"""
    t = str(t).strip().upper()
    t = MAPEAMENTO_BDR.get(t, t)
    # Se for ativo brasileiro (número no ticker) e sem sufixo, adiciona .SA
    if any(char.isdigit() for char in t) and not t.endswith('.SA') and len(t) <= 6:
        return f"{t}.SA"
    return t

st.title("📊 Gestão de Portfólio e Stock Discovery")
st.markdown("---")

# Verificação de arquivos críticos da camada Gold
if os.path.exists(PATH_RANKING) and os.path.exists(PATH_PESOS):
    df_rank = pd.read_parquet(PATH_RANKING)
    df_peso = pd.read_parquet(PATH_PESOS)
    
    # Garantir que tickers na Gold também estejam normalizados para o merge
    df_rank['ticker'] = df_rank['ticker'].apply(normalizar_ticker)
    df_peso['ticker'] = df_peso['ticker'].apply(normalizar_ticker)

    # --- PROCESSAMENTO DE CUSTÓDIA ---
    if os.path.exists(PATH_CUSTODIA):
        df_custodia = pd.read_csv(PATH_CUSTODIA)
        df_custodia['ticker'] = df_custodia['ticker'].apply(normalizar_ticker)
        
        # Merge com preços (df_rank deve conter a coluna 'preco')
        df_rebal = pd.merge(df_custodia, df_rank[['ticker', 'preco']], on='ticker', how='left')
        df_rebal['preco'] = df_rebal['preco'].fillna(0)
        
        # Valor de Mercado e Patrimônio
        df_rebal['Valor_Total_Ativo'] = df_rebal['quantidade'].astype(float) * df_rebal['preco']
        total_patrimonio = df_rebal['Valor_Total_Ativo'].sum()
        
        # Peso Atual
        df_rebal['Peso_Atual'] = (df_rebal['Valor_Total_Ativo'] / total_patrimonio * 100) if total_patrimonio > 0 else 0
        
        # Merge final com Markowitz
        df_final = pd.merge(df_peso, df_rebal[['ticker', 'Peso_Atual', 'quantidade']], on='ticker', how='outer').fillna(0)
    else:
        st.error(f"Arquivo de custódia não encontrado em: {PATH_CUSTODIA}")
        df_final = df_peso.copy()
        df_final['Peso_Atual'] = 0.0
        total_patrimonio = 0

    df_final['Diferenca'] = df_final['Peso_Ideal'] - df_final['Peso_Atual']
    df_total = pd.merge(df_final, df_rank, on='ticker', how='inner')

    # --- VISÃO 1: MÉTRICAS E REBALANCEAMENTO ---
    st.header("1. ⚖️ Rebalanceamento e Simulação de Aporte")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Patrimônio Total", f"R$ {total_patrimonio:,.2f}")
    
    # Lógica de Gaps
    if not df_final.empty:
        ticker_compra = df_final.loc[df_final['Diferenca'].idxmax(), 'ticker']
        ticker_venda = df_final.loc[df_final['Diferenca'].idxmin(), 'ticker']
        m2.metric("Aportar em (Maior Gap)", ticker_compra)
        m3.metric("Reduzir em (Maior Excesso)", ticker_venda)

    col_sim, col_graph = st.columns([1.2, 2])

    with col_sim:
        st.subheader("Simulador de Aporte")
        valor_aporte = st.number_input("Valor para investir (R$):", min_value=0.0, value=1000.0, step=100.0)
        
        df_compra_sug = df_final[df_final['Diferenca'] > 0].copy()
        
        if not df_compra_sug.empty and valor_aporte > 0:
            total_gap = df_compra_sug['Diferenca'].sum()
            df_compra_sug['Alocacao_R$'] = (df_compra_sug['Diferenca'] / total_gap) * valor_aporte
            
            # Merge com preços
            df_compra_sug = pd.merge(df_compra_sug, df_rank[['ticker', 'preco']], on='ticker')
            df_compra_sug['Qtd_Sugerida'] = (df_compra_sug['Alocacao_R$'] / df_compra_sug['preco']).fillna(0).astype(int)
            
            # --- LÓGICA DE AGRUPAMENTO (EXECUTAR ANTES DE EXIBIR) ---
            df_exibicao = df_compra_sug.query("Qtd_Sugerida > 0").groupby('ticker').agg({
                'Qtd_Sugerida': 'sum',
                'Alocacao_R$': 'sum'
            }).reset_index()
            
           
            st.write(f"Com R$ {valor_aporte:,.2f}, compre:")
            st.dataframe(
                df_exibicao.sort_values(by='Alocacao_R$', ascending=False), 
                use_container_width=True,
                hide_index=True # Deixa o visual mais profissional
            )

    with col_graph:
        fig_rebal = px.bar(
            df_final.sort_values(by='Diferenca', ascending=False), 
            x='ticker', y='Diferenca',
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
        title="Score Fundamentos vs Alocação Sugerida",
        labels={'score': 'Saúde Financeira (Score)', 'Peso_Ideal': '% Alocação Ideal'}
    )
    st.plotly_chart(fig_decisao, use_container_width=True)

    # --- VISÃO 3: TIMING (RSI) ---
    if os.path.exists(PATH_TIMING):
        st.divider()
        st.header("3. 🔍 Discovery: Timing de Entrada (RSI)")
        df_timing = pd.read_parquet(PATH_TIMING)
        df_timing['ticker'] = df_timing['ticker'].apply(normalizar_ticker)
        
        def color_status(val):
            color = '#2ecc71' if 'Compra' in str(val) else '#e74c3c' if 'Venda' in str(val) else 'None'
            return f'background-color: {color}'
        
        st.dataframe(
            df_timing[['ticker', 'rsi', 'status']].style.applymap(color_status, subset=['status']),
            use_container_width=True
        )

    # --- VISÃO 4: BACKTEST ---
    if os.path.exists(PATH_BACKTEST):
        st.divider()
        st.header("📈 Backtest: Performance Histórica")
        df_b = pd.read_parquet(PATH_BACKTEST)
        
        fig_b = px.line(
            df_b, 
            title="Evolução Patrimonial (Base 100)",
            color_discrete_map={'IBOVESPA': 'gray', 'CARTEIRA_ELITE': 'blue', 'MINHA_CARTEIRA': 'green'}
        )
        fig_b.update_layout(hovermode="x unified")
        st.plotly_chart(fig_b, use_container_width=True)

else:
    st.error("Erro: Arquivos da camada Gold não encontrados. Execute o pipeline `analise_acoes.py` primeiro.")

st.sidebar.info("Atualizado em: " + pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"))