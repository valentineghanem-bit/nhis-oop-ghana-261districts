# Dockerfile — NHIS OOP Ghana 261 Districts | AIPOCH v6.5
# Full computational environment: Python 3.12 + R 4.3 + all requirements
# Build:  docker build -t nhis-oop-ghana .
# Run:    docker run --rm -v $(pwd):/app nhis-oop-ghana bash run_all.sh

FROM python:3.12-slim

LABEL maintainer="Valentine Golden Ghanem <valentineghanem@gmail.com>"
LABEL org.opencontainers.image.title="NHIS OOP Ghana 261 Districts"
LABEL org.opencontainers.image.description="Spatial epidemiology and ML pipeline"
LABEL org.opencontainers.image.version="1.0.0"

# ─── SYSTEM PACKAGES ──────────────────────────────────────────────────────────

RUN apt-get update && apt-get install -y --no-install-recommends \
        # R runtime
        r-base \
        r-base-dev \
        # GDAL / GEOS / PROJ (required by geopandas + sf)
        libgdal-dev \
        gdal-bin \
        libgeos-dev \
        libproj-dev \
        proj-data \
        proj-bin \
        # Build tools
        build-essential \
        libpq-dev \
        # Utilities
        curl \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ─── PYTHON ENVIRONMENT ──────────────────────────────────────────────────────

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─── R PACKAGES ──────────────────────────────────────────────────────────────

RUN Rscript -e "\
    options(repos = c(CRAN = 'https://cloud.r-project.org')); \
    install.packages(c('spdep', 'spatialreg', 'sf', 'dplyr'), quiet = TRUE); \
    cat('R packages installed:\\n'); \
    cat('  spdep', as.character(packageVersion('spdep')), '\\n'); \
    cat('  spatialreg', as.character(packageVersion('spatialreg')), '\\n'); \
    cat('  sf', as.character(packageVersion('sf')), '\\n'); \
    cat('  dplyr', as.character(packageVersion('dplyr')), '\\n')"

# ─── PROJECT FILES ────────────────────────────────────────────────────────────

COPY . /app

# Ensure scripts are executable
RUN chmod +x run_all.sh

# Create output directories
RUN mkdir -p data/processed data/raw figures tables

# ─── VALIDATION ───────────────────────────────────────────────────────────────

# Syntax-check all Python scripts at build time
RUN find . -name "*.py" \
        -not -path "./.git/*" \
        -not -path "./dashboard/*" \
        -not -path "./poster/*" \
    | sort | xargs -I{} python -m py_compile {} \
    && echo "All Python scripts pass syntax check"

# Dependency consistency check
RUN pip check

# ─── RUNTIME ──────────────────────────────────────────────────────────────────

# Default: validate pipeline inputs; override with run_all.sh for full run
CMD ["python", "scripts/analysis_pipeline.py", "--validate"]
