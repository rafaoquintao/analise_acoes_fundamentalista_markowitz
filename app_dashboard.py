import streamlit as st
import pandas as pd
import plotly.express as px
import os
import yfinance as yf

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Dashboard Financeiro Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CAMINHOS DOS ARQUIVOS ---
PATH_RANKING = "data_lake/gold/ranking_fundamentalista.parquet"
PATH_PESOS = "data_lake/gold/alocacao_otimizada.parquet"
PATH_TIMING = "data_lake/gold/timing_elite.parquet"
PATH_BACKTEST = "data_lake/gold/backtest_performance.parquet"
PATH_CUSTODIA = "data_lake/bronze/meus_ativos.csv"

# --- DICIONÁRIO DE CONVERSÃO STOCKS -> BDRs ---
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
        usd_brl = yf.Ticker("USDBRL=X")
        return usd_brl.fast_info['lastPrice']
    except:
        return 5.10  # Fallback mais atualizado

st.title("📊 Gestão de Portfólio e Stock Discovery")
st.markdown("---")

if os.path.exists(PATH_RANKING) and os.path.exists(PATH_PESOS):
    df_rank = pd.read_parquet(PATH_RANKING).drop_duplicates(subset="ticker", keep="last")
    df_peso = pd.read_parquet(PATH_PESOS).drop_duplicates(subset="ticker", keep="last")

    df_rank["ticker"] = df_rank["ticker"].apply(normalizar_ticker)
    df_peso["ticker"] = df_peso["ticker"].apply(normalizar_ticker)

    if os.path.exists(PATH_CUSTODIA):
        df_custodia = pd.read_csv(PATH_CUSTODIA)
        df_custodia["ticker"] = df_custodia["ticker"].apply(normalizar_ticker)
        df_custodia = df_custodia.groupby("ticker")["quantidade"].sum().reset_index()

        taxa_dolar = obter_cambio_usd_brl()

        # Merge com preços
        df_rebal = pd.merge(df_custodia, df_rank[["ticker", "preco"]], on="ticker", how="left")

        # Busca preços faltantes individualmente (evita falha em lote)
        tickers_sem_preco = df_rebal[df_rebal["preco"].isnull() | (df_rebal["preco"] == 0)]["ticker"].tolist()
        if tickers_sem_preco:
            for t in tickers_sem_preco:
                try:
                    p = yf.Ticker(t).fast_info['lastPrice']
                    if (p is None or p == 0) and ".SA" not in t:
                        p = yf.Ticker(f"{t}.SA").fast_info['lastPrice']
                    df_rebal.loc[df_rebal['ticker'] == t, 'preco'] = p
                except: continue

        # Lógica de Moeda Corrigida
        def converter_moeda_local(row):
            t = str(row['ticker'])
            if not any(char.isdigit() for char in t) and not t.endswith(".SA"):
                return row['preco'] * taxa_dolar
            return row['preco']

        df_rebal["preco"] = df_rebal["preco"].fillna(0)
        df_rebal['preco_convertido'] = df_rebal.apply(converter_moeda_local, axis=1)
        df_rebal['Valor_Total_Ativo'] = df_rebal['quantidade'] * df_rebal['preco_convertido']
        total_patrimonio = df_rebal['Valor_Total_Ativo'].sum()
        
        df_rebal["Peso_Atual"] = (df_rebal["Valor_Total_Ativo"] / total_patrimonio * 100) if total_patrimonio > 0 else 0

        # Merge único com Markowitz
        df_final = pd.merge(df_peso, df_rebal[["ticker", "Peso_Atual", "quantidade"]], on="ticker", how="outer").fillna(0)
        df_final = df_final.groupby("ticker").agg({"Peso_Ideal": "max", "Peso_Atual": "sum", "quantidade": "sum"}).reset_index()
    else:
        st.error("Custódia não encontrada.")
        df_final, total_patrimonio = df_peso.copy(), 0

    df_final["Diferenca"] = df_final["Peso_Ideal"] - df_final["Peso_Atual"]
    df_total = pd.merge(df_final, df_rank, on="ticker", how="inner")

    # --- VISÃO 1: MÉTRICAS ---
    st.header("1. ⚖️ Rebalanceamento")
    m1, m2, m3 = st.columns(3)
    m1.metric("Patrimônio Total", f"R$ {total_patrimonio:,.2f}")
    if not df_final.empty:
        m2.metric("Aportar em", df_final.loc[df_final["Diferenca"].idxmax(), "ticker"])
        m3.metric("Reduzir em", df_final.loc[df_final["Diferenca"].idxmin(), "ticker"])

    col_sim, col_graph = st.columns([1, 2])
    with col_sim:
        valor_aporte = st.number_input("Valor do aporte (R$):", min_value=0.0, value=1000.0)
        df_compra = df_final[df_final["Diferenca"] > 0].copy()
        if not df_compra.empty and valor_aporte > 0:
            total_gap = df_compra["Diferenca"].sum()
            df_compra["Alocacao_R$"] = (df_compra["Diferenca"] / total_gap) * valor_aporte
            df_compra = pd.merge(df_compra, df_rank[["ticker", "preco"]], on="ticker")
            df_compra["Qtd"] = (df_compra["Alocacao_R$"] / df_compra["preco"]).fillna(0).astype(int)
            st.dataframe(df_compra.query("Qtd > 0")[["ticker", "Qtd", "Alocacao_R$"]], hide_index=True)

    with col_graph:
        st.plotly_chart(px.bar(df_final.sort_values(by="Diferenca", ascending=False), x="ticker", y="Diferenca", color="Diferenca", color_continuous_scale="RdYlGn"), use_container_width=True)

    # --- VISÃO 3: TIMING (RSI) ---
    if os.path.exists(PATH_TIMING):
        st.divider()
        st.header("3. 🔍 Discovery: Timing (RSI)")
        df_timing = pd.read_parquet(PATH_TIMING)
        df_timing['ticker'] = df_timing['ticker'].apply(normalizar_ticker)
        df_timing = df_timing.drop_duplicates(subset='ticker', keep='last')
        
        st.dataframe(df_timing[['ticker', 'rsi', 'status']].style.applymap(lambda x: f'background-color: {"#2ecc71" if "Compra" in str(x) else "#e74c3c" if "Venda" in str(x) else "#777474"}', subset=['status']), use_container_width=True, hide_index=True)

    # --- VISÃO 4: BACKTEST ---
    if os.path.exists(PATH_BACKTEST):
        st.divider()
        st.header("📈 Performance Histórica")
        st.plotly_chart(px.line(pd.read_parquet(PATH_BACKTEST)).update_layout(hovermode="x unified"), use_container_width=True)

with st.expander("Debug: Itens Zerados"):
    st.table(df_rebal[df_rebal['preco'] == 0][['ticker', 'quantidade']])