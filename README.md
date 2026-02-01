📊 Financial Intelligence & Portfolio Optimization Pipeline
Este projeto implementa um pipeline de dados ponta-a-ponta para análise de ativos da B3, utilizando arquitetura de Data Lake (Medallion Architecture). O objetivo é cruzar indicadores fundamentalistas (Value Investing), análise técnica (Timing) e otimização matemática de portfólios (Markowitz).

🚀 Destaques do Projeto
Performance Superior: O pipeline identificou que a carteira pessoal atual supera o IBOVESPA e o modelo filtrado "Elite", validando a estratégia de seleção de ativos.

Arquitetura de Dados: Separação clara em camadas Bronze (Raw), Silver (Cleaned) e Gold (Business/Analytics).

Multi-Sourcing: Integração híbrida entre as bibliotecas Fundamentus (para análise em escala) e yfinance (para séries temporais e ativos específicos).

🏗️ Arquitetura do Pipeline
O projeto segue a lógica de processamento em camadas para garantir integridade e escalabilidade:

Bronze (Ingestion): Captura de dados brutos do mercado (Ações, FIIs e ETFs).

Silver (Processing): Limpeza de dados, tratamento de nulos e normalização de séries históricas de preços.

Gold (Analytics): * Ranking Fundamentalista: Score baseado em P/L, ROE e Dividend Yield.

Markowitz Optimization: Cálculo da Fronteira Eficiente para maximizar o Sharpe Ratio.

Backtesting: Comparação histórica de performance entre estratégias e o Benchmark (IBOVESPA).

🛠️ Tecnologias Utilizadas
Linguagem: Python 3.10+

Processamento de Dados: Pandas, NumPy.

Visualização: Streamlit (Dashboard), Plotly (Gráficos Interativos).

Data Quality: Implementação de Quality Gates para validação de esquemas e dados nulos.

Persistência: Arquivos no formato Parquet (alta performance e compressão).

📈 Insights Gerados pelo Dashboard
O dashboard desenvolvido permite visualizar:

Matriz de Decisão: Relação entre a saúde financeira da empresa (Score) vs. o peso ideal sugerido pela matemática de risco.

Timing de Execução: Indicador RSI para evitar compras em zonas de sobrecompra.

Prova de Conceito (PoC): Gráfico de Backtest comparativo validando a tese de investimento.

🔧 Como Executar
Instale as dependências:

Bash
pip install -r requirements.txt
Execute o pipeline de dados:

Bash
python analise_acoes.py
Inicie o dashboard:

Bash
streamlit run app_dashboard.py
Conclusão do Backtest
Nota do Autor: Durante o desenvolvimento, observou-se que a carteira personalizada superou significativamente o modelo puramente quantitativo (Elite) e o índice de referência. Isso demonstra que a combinação de filtros algorítmicos com o julgamento qualitativo de setores estratégicos gera o maior Alpha para o investidor.