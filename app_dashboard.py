import streamlit as st
import pandas as pd
import plotly.express as px
import os
import yfinance as yf

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Dashboard Financeiro Intelligence", layout="wide")

# --- CAMINHOS DOS ARQUIVOS ---
PATH_RANKING = "data_lake/gold/ranking_fundamentalista.parquet"
PATH_PESOS = "data_lake/gold/alocacao_otimizada.parquet"
PATH_TIMING = "data_lake/gold/timing_elite.parquet"
PATH_BACKTEST = "data_lake/gold/backtest_performance.parquet"
PATH_CUSTODIA = "data_lake/bronze/meus_ativos.csv"

MAPEAMENTO_BDR = {
    "NVDA": "NVDC34", "AMZN": "AMZO34", "DIS": "DISB34",
    "KO": "COCA34", "BUD": "ABUD34", "F": "FDMO34",
    "NU": "ROXO34", "AAPL": "AAPL34", "GOOGL": "GOGL34",
}

def normalizar_ticker(t):
    t = str(t).strip().upper()
    t = MAPEAMENTO_BDR.get(t, t)
    if any(char.isdigit() for char in t) and not t.endswith(".SA") and len(t) <= 6:
        return f"{t}.SA"
    return t

def obter_cambio_usd_brl():
    try:
        return yf.Ticker("USDBRL=X").fast_info['lastPrice']
    except:
        return 5.15

st.title("📊 Gestão de Portfólio e Stock Discovery")
st.markdown("---")

if os.path.exists(PATH_RANKING) and os.path.exists(PATH_PESOS):
    df_rank = pd.read_parquet(PATH_RANKING).drop_duplicates(subset="ticker", keep="last")
    df_peso = pd.read_parquet(PATH_PESOS).drop_duplicates(subset="ticker", keep="last")
    
    df_rank["ticker"] = df_rank["ticker"].apply(normalizar_ticker)
    df_peso["ticker"] = df_peso["ticker"].apply(normalizar_ticker)

    # --- PROCESSAMENTO DE CUSTÓDIA ---
    if os.path.exists(PATH_CUSTODIA):
        df_custodia = pd.read_csv(PATH_CUSTODIA)
        df_custodia["ticker"] = df_custodia["ticker"].apply(normalizar_ticker)
        df_custodia = df_custodia.groupby("ticker")["quantidade"].sum().reset_index()

        taxa_dolar = obter_cambio_usd_brl()
        
        # Merge com a Gold para pegar o preço base
        df_rebal = pd.merge(df_custodia, df_rank[["ticker", "preco"]], on="ticker", how="left")

        # FORÇAR ATUALIZAÇÃO DE PREÇO PARA CHEGAR NOS 52K
        tickers_para_atu = df_rebal["ticker"].tolist()
        try:
            # Busca preços atuais para garantir os 52k
            precos_mercado = yf.download(tickers_para_atu, period="1d", progress=False)["Close"].iloc[-1]
            for t in tickers_para_atu:
                if t in precos_mercado:
                    df_rebal.loc[df_rebal["ticker"] == t, "preco"] = precos_mercado[t]
        except: pass

        def converter_moeda_local(row):
            t = str(row['ticker'])
            if not any(char.isdigit() for char in t) and not t.endswith(".SA"):
                return row['preco'] * taxa_dolar
            return row['preco']

        df_rebal['preco_convertido'] = df_rebal.apply(converter_moeda_local, axis=1)
        df_rebal['Valor_Total_Ativo'] = df_rebal['quantidade'] * df_rebal['preco_convertido']
        total_patrimonio = df_rebal['Valor_Total_Ativo'].sum()
        
        df_rebal["Peso_Atual"] = (df_rebal["Valor_Total_Ativo"] / total_patrimonio * 100) if total_patrimonio > 0 else 0

        df_final = pd.merge(df_peso, df_rebal[["ticker", "Peso_Atual", "quantidade"]], on="ticker", how="outer").fillna(0)
        df_final = df_final.groupby("ticker").agg({"Peso_Ideal": "max", "Peso_Atual": "sum", "quantidade": "sum"}).reset_index()
    else:
        df_final, total_patrimonio = df_peso.copy(), 0

    df_final["Diferenca"] = df_final["Peso_Ideal"] - df_final["Peso_Atual"]
    df_total = pd.merge(df_final, df_rank, on="ticker", how="inner")

    # --- VISÃO 1: REBALANCEAMENTO ---
    st.header("1. ⚖️ Rebalanceamento e Patrimônio")
    st.metric("Patrimônio Total", f"R$ {total_patrimonio:,.2f}")
    
    col_sim, col_graph = st.columns([1, 2])
    with col_sim:
        valor_aporte = st.number_input("Simular Aporte (R$):", min_value=0.0, value=1000.0)
        df_compra = df_final[df_final["Diferenca"] > 0].copy()
        if not df_compra.empty and valor_aporte > 0:
            total_gap = df_compra["Diferenca"].sum()
            df_compra["Alocacao_R$"] = (df_compra["Diferenca"] / total_gap) * valor_aporte
            df_compra = pd.merge(df_compra, df_rank[["ticker", "preco"]], on="ticker")
            df_compra["Qtd"] = (df_compra["Alocacao_R$"] / df_compra["preco"]).fillna(0).astype(int)
            st.dataframe(df_compra.query("Qtd > 0")[["ticker", "Qtd", "Alocacao_R$"]], hide_index=True)

    with col_graph:
        st.plotly_chart(px.bar(df_final.sort_values(by="Diferenca", ascending=False), x="ticker", y="Diferenca", color="Diferenca", color_continuous_scale="RdYlGn"), use_container_width=True)

    # --- VISÃO 2: MATRIZ DE DECISÃO (RESTAURADA) ---
    st.divider()
    st.header("2. 🎯 Matriz de Decisão: Markowitz x Score")
    fig_decisao = px.scatter(
        df_total.drop_duplicates(subset="ticker"),
        x="score", y="Peso_Ideal", size="dividend_yield", text="ticker", color="Diferenca",
        color_continuous_scale="RdYlGn", title="Saúde Financeira vs Alocação Alvo",
        labels={"score": "Score Fundamentalista", "Peso_Ideal": "% Alocação Ideal"}
    )
    st.plotly_chart(fig_decisao, use_container_width=True)

    # --- VISÃO 3: TIMING (RSI + P/VP) ---
    if os.path.exists(PATH_TIMING):
        st.divider()
        st.header("3. 🔍 Discovery: Timing e Valuation")
        df_timing = pd.read_parquet(PATH_TIMING)
        df_timing['ticker'] = df_timing['ticker'].apply(normalizar_ticker)
        
        # Merge com df_rank para trazer o P/L
        df_timing = pd.merge(df_timing, df_rank[['ticker', 'p_l']], on='ticker', how='left')
        df_timing = df_timing.drop_duplicates(subset='ticker', keep='last')

        def color_status(val):
            color = '#2ecc71' if 'Compra' in str(val) else '#e74c3c' if 'Venda' in str(val) else "#777474"
            return f'background-color: {color}'

        st.dataframe(
            df_timing[['ticker', 'rsi', 'p_l', 'status']].style.applymap(color_status, subset=['status']).format({"p_l": "{:.2f}"}),
            use_container_width=True, hide_index=True
        )

    # --- VISÃO 4: BACKTEST ---
    if os.path.exists(PATH_BACKTEST):
        st.divider()
        st.header("📈 Backtest")
        st.plotly_chart(px.line(pd.read_parquet(PATH_BACKTEST)).update_layout(hovermode="x unified"), use_container_width=True)

with st.expander("Detalhamento da Custódia"):
    st.dataframe(df_rebal[['ticker', 'quantidade', 'preco_convertido', 'Valor_Total_Ativo']])