# Usa uma imagem leve do Python
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema necessárias para compilar algumas libs financeiras
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de dependências primeiro (otimiza o cache do Docker)
COPY requirements.txt .

# Instala as bibliotecas do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o conteúdo do seu projeto para dentro do container
COPY . .

# Cria a estrutura de pastas do Data Lake para evitar erros de permissão
RUN mkdir -p data_lake/bronze data_lake/silver data_lake/gold

# Comando para rodar o seu script principal
CMD ["python", "analise_acoes.py"]