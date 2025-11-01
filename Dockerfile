# Dockerfile (Receita do Ambiente)
# Este arquivo define a máquina virtual que rodará nosso código.
# (Garante que o Takashi rode exatamente no mesmo ambiente que eu).

# 1. Ponto de Partida (Imagem Base)
# Começamos com a imagem oficial do PyTorch 2.3.1 com CUDA 12.1.
# (Já vem com drivers da NVIDIA, perfeito para a 4070).
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 2. Diretório de Trabalho
# Define '/app' como a pasta principal dentro do container.
# Todo o nosso projeto ficará aqui.
WORKDIR /app

# 3. Otimizações do Python no Docker
# Variáveis de ambiente para o Python rodar "limpo":
# (PYTHONUNBUFFERED: Faz os 'prints' irem direto para o log, sem atraso)
# (PYTHONDONTWRITEBYTECODE: Impede o Python de criar arquivos .pyc)
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# 4. Instalação das Dependências
# Primeiro, copiamos *apenas* o 'requirements.txt' para dentro da imagem.
# (O Docker é inteligente, se este arquivo não mudar, ele não reinstala tudo).
COPY requirements.txt .

# Agora, rodamos os comandos de instalação.
# 1. apt-get update -> Atualiza o sistema Linux base
# 2. pip install -> Lê nosso requirements.txt e instala tudo
# 3. apt-get clean -> Limpa o lixo da instalação para a imagem ficar menor
RUN apt-get update && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 5. Cópia do Código do Projeto
# Com o ambiente pronto, copiamos *todo* o nosso código
# (os scripts .py, etc.) para a pasta '/app'.
COPY . .

# 6. Comando Padrão
# O comando que o container vai rodar quando for "ligado".
# 'sleep infinity' é um truque para manter o container vivo
# indefinidamente, permitindo que a gente conecte o VSCode nele.
CMD ["sleep", "infinity"]