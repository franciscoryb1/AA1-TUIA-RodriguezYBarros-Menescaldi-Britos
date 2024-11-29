import pandas as pd
import numpy as np
import joblib

# Diccionarios necesarios para las transformaciones
direccion_a_grados = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
grupos_principales = {
    "N": ["N", "NNW", "NNE"],
    "S": ["S", "SSW", "SSE"],
    "E": ["E", "ENE", "ESE", "SE", "NE"],
    "W": ["W", "WNW", "WSW", "SW", "NW"],
}

# Ruta del scaler
SCALER_PATH = "./scaler.joblib"

# Cargar el scaler serializado
scaler = joblib.load(SCALER_PATH)

# Columnas a estandarizar
columnas_a_estandarizar = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
    'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
    'Temp9am', 'Temp3pm', 'WindGustDir_sin', 'WindGustDir_cos',
    'WindDir9am_sin', 'WindDir9am_cos', 'WindDir3pm_sin', 'WindDir3pm_cos'
]

def agrupar_direcciones(direccion):
    """
    Agrupa las direcciones en 'N', 'S', 'E', 'W', o 'Otro'.
    """
    for grupo, direcciones in grupos_principales.items():
        if direccion in direcciones:
            return grupo
    return "Otro"

def procesar_direcciones(data, columnas):
    """
    Agrupa direcciones y las convierte a sin y cos.
    """
    for col in columnas:
        # Agrupar direcciones
        data[col] = data[col].apply(agrupar_direcciones)
        # Crear columnas sin y cos
        data[f"{col}_sin"] = np.sin(np.deg2rad(data[col].map(direccion_a_grados)))
        data[f"{col}_cos"] = np.cos(np.deg2rad(data[col].map(direccion_a_grados)))
        # Eliminar la columna original
        data.drop(columns=[col], inplace=True)
    return data

def mapear_rain_columns(data, columnas):
    """
    Mapea 'Yes' a 1 y 'No' a 0 en las columnas especificadas.
    """
    for col in columnas:
        data[col] = data[col].map({'Yes': 1, 'No': 0})
    return data

def imputar_nulos(data):
    """
    Imputa valores nulos en el DataFrame:
    - Numéricos: mediana
    - Categóricos: moda
    """
    # Imputar valores nulos en columnas numéricas
    columnas_numericas = data.select_dtypes(include=["float64", "int64"]).columns
    for col in columnas_numericas:
        if data[col].isnull().any():
            mediana = data[col].median()
            data[col].fillna(mediana, inplace=True)

    # Imputar valores nulos en columnas categóricas
    columnas_categoricas = data.select_dtypes(include=["object"]).columns
    for col in columnas_categoricas:
        if data[col].isnull().any():
            moda = data[col].mode()[0]
            data[col].fillna(moda, inplace=True)

    return data

def escalar_datos(data):
    """
    Escala únicamente las columnas numéricas seleccionadas para estandarización.
    """
    data_copy = data.copy()
    data_copy[columnas_a_estandarizar] = scaler.transform(data_copy[columnas_a_estandarizar])
    return data_copy

def preprocesar_datos(data):
    """
    Aplica todas las transformaciones necesarias al dataset y garantiza que las columnas
    coincidan con las del conjunto de entrenamiento.
    """
    # Paso 1: Imputar valores nulos
    data = imputar_nulos(data)

    # Paso 2: Procesar columnas de direcciones
    columnas_direcciones = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    if set(columnas_direcciones).issubset(data.columns):
        data = procesar_direcciones(data, columnas_direcciones)

    # Paso 3: Mapear columnas categóricas de "Yes/No"
    columnas_yes_no = ['RainToday']
    if set(columnas_yes_no).issubset(data.columns):
        data = mapear_rain_columns(data, columnas_yes_no)

    # Paso 4: Escalar los datos
    data = escalar_datos(data)

    return data
