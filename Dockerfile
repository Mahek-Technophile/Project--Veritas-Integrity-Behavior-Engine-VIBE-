FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    openenv-core \
    openai \
    pydantic \
    fastapi \
    uvicorn

EXPOSE 7860

CMD ["uvicorn", "environment.env:app", "--host", "0.0.0.0", "--port", "7860"]