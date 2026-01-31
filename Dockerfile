FROM python:3.10-slim

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    curl \
    zstd \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Install Ollama
# -------------------------
RUN curl -fsSL https://ollama.com/install.sh | sh

# -------------------------
# App setup
# -------------------------
WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# IMPORTANT: let Python see src/ as top-level
ENV PYTHONPATH=/app/src

# Ollama port
EXPOSE 11434

# -------------------------
# Start Ollama + app
# -------------------------
CMD ["bash", "-c", "ollama serve & sleep 5 && python -m src.main"]
