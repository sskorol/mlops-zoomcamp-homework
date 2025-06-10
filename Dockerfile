FROM prefecthq/prefect:3-python3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PREFECT_API_URL=http://prefect-server:4200/api
ENV PYTHONPATH=/app

CMD ["python", "/app/src/serve_flows.py"]
