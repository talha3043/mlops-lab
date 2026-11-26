FROM python:3.12-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/__init__.py src/serve.py ./src/
COPY model ./model
ARG GIT_SHA=local
ENV GIT_SHA=$GIT_SHA \
    MODEL_DIR=/app/model
RUN useradd --create-home app
USER app
EXPOSE 8000
CMD ["uvicorn", "src.serve:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
