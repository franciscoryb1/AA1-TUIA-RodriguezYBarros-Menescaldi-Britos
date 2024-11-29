# Imagen base con Python 3.9
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar únicamente los archivos necesarios
COPY inferencia.py /app/inferencia.py
COPY preprocesamiento.py /app/preprocesamiento.py
COPY final_model_nn.h5 /app/final_model_nn.h5
COPY scaler.joblib /app/scaler.joblib
COPY requirements.txt /app/requirements.txt

# Actualizar pip antes de instalar las dependencias
RUN pip install --upgrade pip

# Instalar las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Comando para ejecutar el script de inferencia
CMD ["python", "inferencia.py"]
