FROM python:3.11-slim AS base

RUN apt update && apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir opencv-python-headless

WORKDIR /app

COPY requirements.txt /app
RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; \
    else echo "requirements.txt not found" && exit 1; \
    fi

COPY . .

CMD ["python", "app.py"]