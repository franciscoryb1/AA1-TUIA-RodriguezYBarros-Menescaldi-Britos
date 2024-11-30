import os
import pandas as pd
from tensorflow.keras.models import load_model
from preprocesamiento import preprocesar_datos

MODEL_PATH = "./final_model_nn.h5"

model = load_model(MODEL_PATH)

def predecir(data):
    """
    Realiza la predicción utilizando el pipeline de preprocesamiento y el modelo neuronal.
    :param data: DataFrame con los datos de entrada.
    :return: DataFrame con la columna 'RainTomorrow' agregada (predicciones).
    """
    data_preprocesada = preprocesar_datos(data.copy())  

    predicciones = model.predict(data_preprocesada)

    clases = (predicciones > 0.5).astype("int32").flatten()  

    data['RainTomorrow'] = clases
    return data

if __name__ == "__main__":
    input_csv = "input.csv"
    output_dir = "output"  
    output_csv = os.path.join(output_dir, "predicciones_output.csv")

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data = pd.read_csv(input_csv)
        print(f"Datos cargados exitosamente desde {input_csv}:\n{data.head()}")

        print("Iniciando el preprocesamiento y la predicción...")
        resultado = predecir(data)

        resultado.to_csv(output_csv, index=False)
        print(f"Predicciones guardadas en: {output_csv}")

    except Exception as e:
        print(f"Error procesando el archivo CSV: {e}")
