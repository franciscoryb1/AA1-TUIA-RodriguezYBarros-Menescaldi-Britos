import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Configuración de la página
# Configuración de la página - debe ser lo primero
st.set_page_config(
    page_title="Predicción de lluvia en Australia",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título principal
st.title('Predicción de lluvia en Australia')

# Carga de modelos
@st.cache_resource
def load_models():
    models = {
        "Grid Search model": joblib.load('Grid_model.joblib'),
        "Random Search model": joblib.load('Random_model.joblib'),
        "Optuna model": joblib.load('Optuna_model.joblib')
    }
    return models

with st.spinner('Cargando modelos...'):
    models = load_models()
st.success('Modelos cargados exitosamente!')

# Selección de modelo
selected_model = st.sidebar.selectbox(
    "Modelo utilizado:",
    options=list(models.keys())
)

# Sidebar
st.sidebar.title("Acerca de")
st.sidebar.info("Esta aplicación predice la probabilidad de lluvia en Australia basada en diversos factores meteorológicos.")

























# Procesameinto del dataset
# Cargamso el dataset original
file_path= 'weatherAUS.csv'
weatherdata = pd.read_csv(file_path, sep=',',engine='python')

# Hacemos la reduccion de dimencionalidad
# Seleccionamos aleatoriamente 10 ciudades de la columna Location usando una semilla fija
cities = weatherdata['Location'].sample(10, random_state=0)
weatherdata = weatherdata[weatherdata['Location'].isin(cities)]
weatherdata = weatherdata.drop('Location', axis=1)

# Eliminamos todas los registros que tengan más de 10 columnas nulas
weatherdata = weatherdata[weatherdata.isnull().sum(axis=1) < 10]

# Hacemos el split de los datos segun la fecha y eliminamos la columna 'Date'
X = weatherdata.drop(columns=['RainTomorrow'])
weatherdata.loc[:, "Date"] = pd.to_datetime(weatherdata["Date"])
X.loc[:, "Date"] = pd.to_datetime(X["Date"])
# Calcular las fechas que abarcan el 70% de los datos
fecha_70porciento = weatherdata['Date'].quantile(0.7)
# Dividir X e y en conjuntos de entrenamiento
X_train = X[X['Date'] <= fecha_70porciento].drop(columns='Date')

# MaxTemp
max_temp_model = LinearRegression()
train_not_null = X_train.dropna(subset=['MaxTemp', 'Temp3pm'])
max_temp_model.fit(train_not_null[['Temp3pm']], train_not_null['MaxTemp'])
# Imputar valores nulos en X_train solo cuando Temp3pm no es NaN
train_null_max_temp = X_train[X_train['MaxTemp'].isnull() & X_train['Temp3pm'].notnull()]
X_train.loc[train_null_max_temp.index, 'MaxTemp'] = max_temp_model.predict(train_null_max_temp[['Temp3pm']])

# Temp3pm
temp3pm_model = LinearRegression()
train_not_null = X_train.dropna(subset=['Temp3pm', 'MaxTemp'])  # Filtramos los datos no nulos
temp3pm_model.fit(train_not_null[['MaxTemp']], train_not_null['Temp3pm'])
# Imputar valores nulos en X_train solo cuando MaxTemp no es NaN
train_null_temp3pm = X_train[X_train['Temp3pm'].isnull() & X_train['MaxTemp'].notnull()]
X_train.loc[train_null_temp3pm.index, 'Temp3pm'] = temp3pm_model.predict(train_null_temp3pm[['MaxTemp']])
# Imputar valores faltantes restantes con la mediana en caso de que ambas variables tengan NaN
X_train['MaxTemp'] = X_train['MaxTemp'].fillna(X_train['MaxTemp'].median())
X_train['Temp3pm'] = X_train['Temp3pm'].fillna(X_train['Temp3pm'].median())

# Temp9am
temp9am_model = LinearRegression()
train_not_null = X_train.dropna(subset=['Temp9am', 'MaxTemp'])
temp9am_model.fit(train_not_null[['MaxTemp']], train_not_null['Temp9am'])

# MaxTemp 
train_null_temp9am = X_train[X_train['Temp9am'].isnull() & X_train['MaxTemp'].notnull()]
X_train.loc[train_null_temp9am.index, 'Temp9am'] = temp9am_model.predict(train_null_temp9am[['MaxTemp']])

# Temp9am
MinTemp_model = LinearRegression()
train_not_null = X_train.dropna(subset=['MinTemp', 'Temp9am'])
MinTemp_model.fit(train_not_null[['Temp9am']], train_not_null['MinTemp'])

# Temp9am
train_null_max_temp = X_train[X_train['MinTemp'].isnull() & X_train['Temp9am'].notnull()]
X_train.loc[train_null_max_temp.index, 'MinTemp'] = MinTemp_model.predict(train_null_max_temp[['Temp9am']])

# Humidity9am
mediana_humidity9am_train = X_train['Humidity9am'].median()
X_train['Humidity9am'] = X_train['Humidity9am'].fillna(mediana_humidity9am_train)

# Humidity3pm
mediana_Humidity3pm_train = X_train['Humidity3pm'].median()
X_train['Humidity3pm'] = X_train['Humidity3pm'].fillna(mediana_Humidity3pm_train)

# Pressure9am
mediana_Pressure9am_train = X_train['Pressure9am'].median()
X_train['Pressure9am'] = X_train['Pressure9am'].fillna(mediana_Pressure9am_train)

# Pressure3pm
mediana_Pressure3pm_train = X_train['Pressure3pm'].median()
X_train['Pressure3pm'] = X_train['Pressure3pm'].fillna(mediana_Pressure3pm_train)

# Sunshine
mediana_Sunshine_train = X_train['Sunshine'].median()
X_train['Sunshine'] = X_train['Sunshine'].fillna(mediana_Sunshine_train)

# Cloud3pm 
mediana_Cloud3pm_train = X_train['Cloud3pm'].median()
X_train['Cloud3pm'] = X_train['Cloud3pm'].fillna(mediana_Cloud3pm_train)

# Cloud9am 
mediana_Cloud9am_train = X_train['Cloud9am'].median()
X_train['Cloud9am'] = X_train['Cloud9am'].fillna(mediana_Cloud9am_train)

# Rainfall 
mediana_Rainfall_train = X_train['Rainfall'].median()
X_train['Rainfall'] = X_train['Rainfall'].fillna(mediana_Rainfall_train)

# Evaporation
mediana_Evaporation_train = X_train['Evaporation'].median()
X_train['Evaporation'] = X_train['Evaporation'].fillna(mediana_Evaporation_train)

# WindGustSpeed 
mediana_WindGustSpeed_train = X_train['WindGustSpeed'].median()
X_train['WindGustSpeed'] = X_train['WindGustSpeed'].fillna(mediana_WindGustSpeed_train)

# WindSpeed9am 
mediana_WindSpeed9am_train = X_train['WindSpeed9am'].median()
X_train['WindSpeed9am'] = X_train['WindSpeed9am'].fillna(mediana_WindSpeed9am_train)

# WindSpeed3pm
mediana_WindSpeed3pm_train = X_train['WindSpeed3pm'].median()
X_train['WindSpeed3pm'] = X_train['WindSpeed3pm'].fillna(mediana_WindSpeed3pm_train)

# WindGustDir
moda_WindGustDir = X_train['WindGustDir'].mode()[0]
X_train['WindGustDir'] = X_train['WindGustDir'].fillna(moda_WindGustDir)

# WindDir9am
moda_WindDir9am = X_train['WindDir9am'].mode()[0]
X_train['WindDir9am'] = X_train['WindDir9am'].fillna(moda_WindDir9am)

# WindDir3pm 
moda_WindDir3pm = X_train['WindDir3pm'].mode()[0]
X_train['WindDir3pm'] = X_train['WindDir3pm'].fillna(moda_WindDir3pm)

# RainToday
moda_RainToday = X_train['RainToday'].mode()[0]
X_train['RainToday'] = X_train['RainToday'].fillna(moda_RainToday)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


# Opciones para la dirección del viento y sus valores correspondientes
options_dir = {
    "North": "N", "North-Northeast": "NNE", "Northeast": "NE", "East-Northeast": "ENE",
    "East": "E", "East-Southeast": "ESE", "Southeast": "SE", "South-Southeast": "SSE",
    "South": "S", "South-Southwest": "SSW", "Southwest": "SW", "West-Southwest": "WSW",
    "West": "W", "West-Northwest": "WNW", "Northwest": "NW", "North-Northwest": "NNW"   
}

# Temperatura
st.header("Temperatura")
col1, col2, col3 = st.columns(3)
with col1:
    MaxTemp = st.number_input("Temperatura máxima", min_value=-40.0, max_value=50.0, value=20.0, step=0.1)
with col2:
    MinTemp = st.number_input("Temperatura mínima", min_value=-40.0, max_value=50.0, value=0.0, step=0.1)
with col3:
    Temp9am = st.number_input("Temperatura a las 9am", min_value=-40.0, max_value=50.0, value=2.0, step=0.1)
    Temp3pm = st.number_input("Temperatura a las 3pm", min_value=-40.0, max_value=50.0, value=25.0, step=0.1)

# Precipitación y Evaporación
st.header("Precipitación y Evaporación")
col1, col2, col3 = st.columns(3)
with col1:
    Rainfall = st.number_input("Cantidad de lluvia", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
with col2:
    Evaporation = st.number_input("Evaporación", min_value=0.0, max_value=100.0, value=13.5, step=0.1)
with col3:
    Sunshine = st.number_input("Horas de sol", min_value=0.0, max_value=24.0, value=6.8, step=0.1)

# Viento
st.header("Viento")
col1, col2 = st.columns(2)
with col1:
    WindGustDir = st.selectbox("Dirección de la ráfaga de viento", list(options_dir.keys()))
    WindGustSpeed = st.number_input("Velocidad de la ráfaga de viento", min_value=0, max_value=200, value=30)
with col2:
    WindDir9am = st.selectbox("Dirección del viento a las 9am", list(options_dir.keys()))
    WindSpeed9am = st.number_input("Velocidad del viento a las 9am", min_value=0, max_value=200, value=15)
    WindDir3pm = st.selectbox("Dirección del viento a las 3pm", list(options_dir.keys()))
    WindSpeed3pm = st.number_input("Velocidad del viento a las 3pm", min_value=0, max_value=200, value=20)

# Humedad
st.header("Humedad")
col1, col2 = st.columns(2)
with col1:
    Humidity9am = st.number_input("Humedad a las 9am", min_value=0, max_value=100, value=70)
with col2:
    Humidity3pm = st.number_input("Humedad a las 3pm", min_value=0, max_value=100, value=67)

# Presión
st.header("Presión")
col1, col2 = st.columns(2)
with col1:
    Pressure9am = st.number_input("Presión a las 9am", min_value=900, max_value=1100, value=1015)
with col2:
    Pressure3pm = st.number_input("Presión a las 3pm", min_value=900, max_value=1100, value=1010)

# Nubosidad
st.header("Nubosidad")
col1, col2 = st.columns(2)
with col1:
    Cloud9am = st.slider("Nubosidad a las 9am", min_value=0, max_value=9, value=5)
with col2:
    Cloud3pm = st.slider("Nubosidad a las 3pm", min_value=0, max_value=9, value=9)

# Lluvia
st.header("Lluvia")
RainToday = st.selectbox("¿Llovió hoy?", ['No', 'Yes'])
    
# Botón para realizar la predicción
if st.button("Realizar predicción", type="primary"):
    with st.spinner("Procesando datos y realizando predicción..."):
        
        data = {
            'MaxTemp': [MaxTemp], 'MinTemp': [MinTemp], 'Rainfall': [Rainfall],
            'Evaporation': [Evaporation], 'Sunshine': [Sunshine],
            'WindGustDir': [options_dir[WindGustDir]], 'WindGustSpeed': [WindGustSpeed],
            'WindDir9am': [options_dir[WindDir9am]], 'WindDir3pm': [options_dir[WindDir3pm]],
            'WindSpeed9am': [WindSpeed9am], 'WindSpeed3pm': [WindSpeed3pm],
            'Humidity9am': [Humidity9am], 'Humidity3pm': [Humidity3pm],
            'Pressure9am': [Pressure9am], 'Pressure3pm': [Pressure3pm],
            'Cloud9am': [Cloud9am], 'Cloud3pm': [Cloud3pm],
            'Temp9am': [Temp9am], 'Temp3pm': [Temp3pm],
            'RainToday': [RainToday]
        }

        user_input = pd.DataFrame(data)





























        # Copiamos el dataset
        X_train_plus_user = X_train.copy()
        # Agregamos el registro del usuario al dataset
        # Selecciona la primera fila de 'df'
        fila_a_agregar = user_input.iloc[0]  

        # Agrega la fila a 'weatherdata'
        X_train_plus_user.loc[len(X_train_plus_user)] = fila_a_agregar

        # Dummies
        def agrupar_direcciones(direccion):
            grupos_principales = {
                "N": ["N", "NNW", "NNE"],
                "S": ["S", "SSW", "SSE"],
                "E": ["E", "ENE", "ESE", "SE", "NE"],
                "W": ["W", "WNW", "WSW", "SW", "NW"],
            }

            for grupo, direcciones in grupos_principales.items():
                if direccion in direcciones:
                    return grupo

            return "Otro"

        # Mapeo de direcciones a ángulos
        direccion_a_grados = {'N': 0, 'E': 90, 'S': 180, 'W': 270}

        # Aplicar agrupamiento de direcciones en X_train y X_test
        X_train['WindGustDir'] = X_train['WindGustDir'].apply(agrupar_direcciones)
        X_train['WindDir9am'] = X_train['WindDir9am'].apply(agrupar_direcciones)
        X_train['WindDir3pm'] = X_train['WindDir3pm'].apply(agrupar_direcciones)
        X_train_plus_user['WindGustDir'] = X_train_plus_user['WindGustDir'].apply(agrupar_direcciones)
        X_train_plus_user['WindDir9am'] = X_train_plus_user['WindDir9am'].apply(agrupar_direcciones)
        X_train_plus_user['WindDir3pm'] = X_train_plus_user['WindDir3pm'].apply(agrupar_direcciones)
        
        # Ahora aplicamos las funciones trigonométricas (sin y cos) directamente sobre las columnas originales
        X_train['WindGustDir_sin'] = np.sin(np.deg2rad(X_train['WindGustDir'].map(direccion_a_grados)))
        X_train['WindGustDir_cos'] = np.cos(np.deg2rad(X_train['WindGustDir'].map(direccion_a_grados)))
        X_train['WindDir9am_sin'] = np.sin(np.deg2rad(X_train['WindDir9am'].map(direccion_a_grados)))
        X_train['WindDir9am_cos'] = np.cos(np.deg2rad(X_train['WindDir9am'].map(direccion_a_grados)))
        X_train['WindDir3pm_sin'] = np.sin(np.deg2rad(X_train['WindDir3pm'].map(direccion_a_grados)))
        X_train['WindDir3pm_cos'] = np.cos(np.deg2rad(X_train['WindDir3pm'].map(direccion_a_grados)))

        X_train_plus_user['WindGustDir_sin'] = np.sin(np.deg2rad(X_train_plus_user['WindGustDir'].map(direccion_a_grados)))
        X_train_plus_user['WindGustDir_cos'] = np.cos(np.deg2rad(X_train_plus_user['WindGustDir'].map(direccion_a_grados)))
        X_train_plus_user['WindDir9am_sin'] = np.sin(np.deg2rad(X_train_plus_user['WindDir9am'].map(direccion_a_grados)))
        X_train_plus_user['WindDir9am_cos'] = np.cos(np.deg2rad(X_train_plus_user['WindDir9am'].map(direccion_a_grados)))
        X_train_plus_user['WindDir3pm_sin'] = np.sin(np.deg2rad(X_train_plus_user['WindDir3pm'].map(direccion_a_grados)))
        X_train_plus_user['WindDir3pm_cos'] = np.cos(np.deg2rad(X_train_plus_user['WindDir3pm'].map(direccion_a_grados)))
        
        
        # Elimina las columnas originales si ya no son necesarias
        X_train.drop(columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'], inplace=True)
        
        X_train_plus_user.drop(columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'], inplace=True)
        
        
        # Mapeo de 'Si' y 'No' a 1 y 0 en las columnas que lo necesiten en X_train y X_test
        X_train['RainToday'] = X_train['RainToday'].map({'Yes': 1, 'No': 0})
        
        X_train_plus_user['RainToday'] = X_train_plus_user['RainToday'].map({'Yes': 1, 'No': 0})
        



        # Creamos el estandarizador, fiteamos el transform de train y luego lo aplicamos al los datos del usuario
        scaler = StandardScaler()
        # Seleccionar las columnas a estandarizar
        columnas_a_estandarizar = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
            'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
            'Temp9am', 'Temp3pm', 'WindGustDir_sin', 'WindGustDir_cos',
            'WindDir9am_sin', 'WindDir9am_cos', 'WindDir3pm_sin', 'WindDir3pm_cos']
        
        # Ajustar y transformar X_train
        X_train[columnas_a_estandarizar] = scaler.fit_transform(X_train[columnas_a_estandarizar]) 
        
        last_row = X_train_plus_user.tail(1)
        # Asegurarse de que last_row sea una copia independiente
        last_row = last_row.copy()

        # Aplicar la estandarización de manera segura
        last_row[columnas_a_estandarizar] = scaler.transform(last_row[columnas_a_estandarizar])

        st.write(X_train)
        st.write(last_row)

        modelo = models[selected_model]
        prediccion = modelo.predict(last_row)
        
        st.write(prediccion)
        
        # Manejar diferentes tipos de salida
        if isinstance(prediccion[0], np.ndarray):
            # Para la red neuronal
            probabilidad_lluvia = prediccion[0][0]
            lluvia_manana = "Yes" if probabilidad_lluvia > 0.5 else "No"
        else:
            # Para otros modelos que devuelven 'Yes' o 'No'
            lluvia_manana = prediccion[0]
            probabilidad_lluvia = None
        
    st.write(lluvia_manana)
    st.write(probabilidad_lluvia)
    
    st.header(f"Resultado de la Predicción usando {selected_model}", divider="rainbow")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if lluvia_manana == "Yes":
            st.image("https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExYnVxbHBnM2s5dDY2cnBxdzUyanF6YmdqcjdnODg4YWV0Y2RkZm05ZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pNn4hlkovWAHfpLRRD/giphy.gif", width=200)
        else:
            st.image("https://media.giphy.com/media/XWXnf6hRiKBJS/giphy.gif", width=200)
    
    with col2:
        st.subheader("Predicción para mañana:")
        st.markdown(f"<h1 style='text-align: center; color: {'#1E90FF' if lluvia_manana == 'Yes' else '#FFD700'};'>{'🌧️ Lloverá' if lluvia_manana == 'Yes' else '☀️ No lloverá'}</h1>", unsafe_allow_html=True)
        
        if probabilidad_lluvia is not None:
            st.metric(label="Probabilidad de lluvia", value=f"{probabilidad_lluvia*100:.1f}%")
        elif hasattr(modelo, 'predict_proba'):
            confianza = modelo.predict_proba(last_row)[0][1] * 100
            st.metric(label="Probabilidad de lluvia", value=f"{confianza:.1f}%")
        else:
            st.write("Probabilidad no disponible para este modelo.")
        
    st.info(f"Esta predicción se basa en los datos meteorológicos proporcionados y el modelo {selected_model} entrenado con datos históricos de Australia.")
