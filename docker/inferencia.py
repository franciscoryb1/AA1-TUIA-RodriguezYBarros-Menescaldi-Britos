import os
import pandas as pd
from tensorflow.keras.models import load_model
from preprocesamiento import preprocesar_datos

# Ruta del modelo
MODEL_PATH = "./final_model_nn.h5"

# Cargar el modelo neuronal
model = load_model(MODEL_PATH)

def predecir(data):
    """
    Realiza la predicción utilizando el pipeline de preprocesamiento y el modelo neuronal.
    :param data: DataFrame con los datos de entrada.
    :return: DataFrame con la columna 'RainTomorrow' agregada (predicciones).
    """
    # Preprocesar los datos de entrada
    data_preprocesada = preprocesar_datos(data.copy())  # Copiar el DataFrame para no modificar el original

    # Realizar las predicciones con el modelo
    predicciones = model.predict(data_preprocesada)

    # Convertir las predicciones continuas (probabilidades) a clases binarias (0 o 1)
    clases = (predicciones > 0.5).astype("int32").flatten()  # Aplanar las predicciones a una lista o array

    # Agregar las predicciones al DataFrame original
    data['RainTomorrow'] = clases
    return data

if __name__ == "__main__":
    # Ruta del archivo CSV de entrada proporcionado por el usuario
    input_csv = "input_usuario.csv"

    # Ruta del archivo de salida
    output_csv = "predicciones_output.csv"

    try:
        # Verificar si el archivo de salida es un directorio
        if os.path.isdir(output_csv):
            raise ValueError(f"Error: '{output_csv}' es un directorio, no se puede escribir en él.")
        
        # Cargar los datos desde el CSV
        data = pd.read_csv(input_csv)
        print(f"Datos cargados exitosamente:\n{data.head()}")

        # Realizar la predicción
        print("Iniciando el preprocesamiento y la predicción...")
        resultado = predecir(data)
        
        # Guardar el resultado en un nuevo archivo CSV
        resultado.to_csv(output_csv, index=False)
        print(f"Predicciones guardadas en: {output_csv}")

    except Exception as e:
        print(f"Error procesando el archivo CSV: {e}")
