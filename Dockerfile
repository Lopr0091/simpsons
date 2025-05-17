FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Instala librerías del sistema necesarias para OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

CMD ["python", "interface.py"]
