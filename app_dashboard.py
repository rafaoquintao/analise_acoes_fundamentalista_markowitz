import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title = "Dashboard de Análise de Ações",
    layout = "wide",
)

st.title("📊 Gestão de Portfólio e Stock Discovery")

# --- Caminho dos arquivos ---

PATH_RANKING = "data_lake/gold/ranking_fundamentalista.parquet"
PATH_PESOS = "data_lake/gold/alocacao_otimizada.parquet"
PATH_ELITE = "data_lake/gold/carteira_elite.parquet"
PATH_TIMING = "data_lake/gold/timing_elite.parquet"

if os.path.exists(PATH_RANKING) and os.path.exists(PATH_PESOS):
    # Lendo dados da Carteira atual
    df_rank = pd.read_parquet(PATH_RANKING)
    df_peso = pd.read_parquet(PATH_PESOS)
    df_total = pd.merge(df_peso, df_rank, on = 'ticker')

    # Visão 1: Carteira Atual 
    st.header("1. Minha Carteira Atual")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Sugestão de Pesos")
        st.dataframe(df_peso)

    with col2:
        fig_decisao = px.scatter(
            df_total, 
            x = 'score', 
            y = 'Peso_Ideal',
            size = 'dividend_yield',
            text = 'ticker',
            title = "Matriz de Decisão(Sua Carteira)",
            labels = {'score': 'Score Fundamentalista', 
                      'Peso_Ideal': '% Sugerido Markowitz'}
        )
        st.plotly_chart(fig_decisao, use_container_width = True)
    
    # Visão 2: Oportunidades
    st.divider()
    st.header("2. Discovery: Carteira Elite B3")

    if os.path.exists(PATH_ELITE):
        df_elite = pd.read_parquet(PATH_ELITE)
        df_timing = pd.read_parquet(PATH_TIMING)

        # Merge para mostrar fundamentos + timing
        df_compra = pd.merge(df_elite, df_timing, on = 'ticker')

        st.write(" Ativos da B3 que atendem aos critérios e o momento atual de compra (RSI):")

        # Estilização básica para o RSI
        def color_rsi(val):
            color = 'green' if 'Compra' in val else 'red' if 'Venda' in val else 'white'
            return f'background-color: {color}; color: black'
        
        st.dataframe(df_compra[['ticker', 'p_l', 'roe', 
                                'dividend_yield', 'rsi', 'status']].style.applymap(color_rsi, subset = ['status']))
else:
    st.warning("Aguardando a execução do pipeline... Executo o script principal primeiro.")

st.divider()
st.header("📈 Backtest: Performance da Estratégia")

PATH_BACKTEST = "data_lake/gold/backtest_performance.parquet"

if os.path.exists(PATH_BACKTEST):
    df_b = pd.read_parquet(PATH_BACKTEST)
    # Renomear para ficar mais amigável no gráfico
    # Melhorando os nomes das legendas
    df_b.columns = ['IBOVESPA', 'ESTRATÉGIA ELITE', 'MINHA CARTEIRA']
    
    fig_b = px.line(
        df_b, 
        labels={'value': 'Evolução (Base 100)', 'Date': 'Data'},
        color_discrete_map={
            'IBOVESPA': 'gray',
            'ESTRATÉGIA ELITE': 'blue',
            'MINHA CARTEIRA': 'green'
        }
    )
    st.plotly_chart(fig_b, use_container_width=True)
else:
    st.info("Rode o pipeline para ver o comparativo de performance.")