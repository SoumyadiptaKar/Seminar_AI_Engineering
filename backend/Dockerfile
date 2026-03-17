FROM python:3.10-slim
WORKDIR /app

#sys deps - ultralytics needs these....
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libx11-6 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN chmod -R 777 * 
#force cpu only torch (may need to change this for openshift maybe????)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1 torchvision==0.20.1

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY utils/ ./utils/
RUN mkdir -p models
COPY models/weights.pt models/

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
