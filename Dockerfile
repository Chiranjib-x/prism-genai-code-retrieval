# CPU-only image. Torch is installed from the CPU wheel index first so pip
# does not pull the multi-GB CUDA build from PyPI.
#
#   docker build -t apps-retrieval .
#   docker run --rm -v hf-cache:/root/.cache apps-retrieval                     # full benchmark
#   docker run --rm -v hf-cache:/root/.cache apps-retrieval python search.py --qid q5001
#
# Candidate programs from the dataset are executed in sandboxed subprocesses.
# Once models and data are cached in the volume, add --network none (and a
# --memory limit) for defence in depth.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY *.py ./
CMD ["python", "pipeline.py", "--model", "Salesforce/SFR-Embedding-Code-400M_R"]
