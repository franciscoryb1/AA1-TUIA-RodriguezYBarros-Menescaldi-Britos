FROM python:3.9-slim

WORKDIR /app

RUN mkdir -p /app/output

COPY inferencia.py /app/inferencia.py
COPY preprocesamiento.py /app/preprocesamiento.py
COPY final_model_nn.h5 /app/final_model_nn.h5
COPY scaler.joblib /app/scaler.joblib
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "inferencia.py"]
