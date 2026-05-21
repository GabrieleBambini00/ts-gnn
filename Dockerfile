FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[all]" 2>/dev/null || \
    pip install --no-cache-dir torch torch-geometric scanpy anndata numpy scipy \
    matplotlib seaborn networkx pyyaml wandb

# Copy source code
COPY . .

# Install project
RUN pip install --no-cache-dir -e .

# Default: run the full pipeline
ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--config", "configs/default.yaml", "--skip-download"]
