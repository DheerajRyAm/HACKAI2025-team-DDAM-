FROM python:3.11-slim AS base

WORKDIR /app


COPY requirements.txt /app
RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; \
    else echo "requirements.txt not found" && exit 1; \
    fi

COPY . .

CMD ["python", "app.py"]