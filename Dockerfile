FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN bash scripts/download_models.sh
ENV PYTHONPATH=/app/src OMP_NUM_THREADS=1 GLOG_minloglevel=2
CMD ["python", "-m", "pytest", "-q"]
