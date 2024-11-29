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
    # Ruta fija del archivo CSV de entrada dentro del contenedor
    input_csv = "input.csv"
    output_dir = "output"  # Directorio para guardar los resultados
    output_csv = os.path.join(output_dir, "predicciones_output.csv")

    try:
        # Crear el directorio de salida si no existe
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Cargar los datos desde el archivo CSV proporcionado
        data = pd.read_csv(input_csv)
        print(f"Datos cargados exitosamente desde {input_csv}:\n{data.head()}")

        # Realizar la predicción
        print("Iniciando el preprocesamiento y la predicción...")
        resultado = predecir(data)

        # Guardar el resultado en un archivo CSV dentro del directorio `output`
        resultado.to_csv(output_csv, index=False)
        print(f"Predicciones guardadas en: {output_csv}")

    except Exception as e:
        print(f"Error procesando el archivo CSV: {e}")
