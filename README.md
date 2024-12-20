
# Predicción de lluvias en Australia 🌧️

  

Este proyecto tiene como objetivo predecir si lloverá al día siguiente en distintas ciudades de Australia, utilizando datos meteorológicos del dataset `weatherAUS.csv`. Desarrollamos y empaquetamos un modelo de red neuronal en un contenedor Docker para realizar inferencias de manera eficiente.

  

El contenedor incluye todo lo necesario: el modelo entrenado, el pipeline de preprocesamiento y las dependencias requeridas, permitiendo que las predicciones se ejecuten de forma directa con solo proporcionar un archivo CSV de entrada.

  

---

  

## Cómo ejecutar la aplicación con Docker 🐳

  

### **Requisitos previos**

1. Tener [Docker](https://www.docker.com/get-started) instalado en tu máquina.

2. Clonar este repositorio y ubicar la terminal en la carpeta docker del repositorio:

```bash

git clone <URL_DEL_REPOSITORIO>

cd <CARPETA_DEL_REPOSITORIO/docker>

```

  

### **Construir la imagen Docker**

Construir la imagen ejecutando:

```bash

docker  build  -t  inferencia_modelo  .

```


---

  

### **Preparar los archivos de entrada**

1. Crear un archivo CSV con los datos meteorológicos a analizar. El archivo debe tener las siguientes columnas y respetar este formato:

---

   | MinTemp | MaxTemp | Rainfall | Evaporation | Sunshine | WindGustDir | WindGustSpeed | WindDir9am | WindDir3pm | WindSpeed9am | WindSpeed3pm | Humidity9am | Humidity3pm | Pressure9am | Pressure3pm | Cloud9am | Cloud3pm | Temp9am | Temp3pm | RainToday |
   |---------|---------|----------|-------------|----------|-------------|---------------|------------|------------|--------------|--------------|-------------|-------------|-------------|-------------|----------|----------|---------|---------|-----------|
   | 12.3    | 22.4    | 1.2      | 4.5         | 9.3      | WNW         | 46            | WNW        | WSW        | 20           | 24           | 87          | 56          | 1015.6      | 1008.7      | 3        | 5        | 14.1    | 21.5    | Yes       |

---

-  **Descripción de las columnas**:

-  **MinTemp** y **MaxTemp**: Temperatura mínima y máxima (en grados Celsius).

-  **Rainfall**: Cantidad de lluvia en mm.

-  **Evaporation** y **Sunshine**: Cantidad de evaporación y horas de sol.

-  **WindGustDir** y **WindGustSpeed**: Dirección y velocidad de las ráfagas de viento.

-  **WindDir9am** y **WindDir3pm**: Dirección del viento a las 9 am y 3 pm.

-  **WindSpeed9am** y **WindSpeed3pm**: Velocidad del viento a las 9 am y 3 pm.

-  **Humidity9am** y **Humidity3pm**: Humedad relativa a las 9 am y 3 pm.

-  **Pressure9am** y **Pressure3pm**: Presión atmosférica a las 9 am y 3 pm.

-  **Cloud9am** y **Cloud3pm**: Cobertura nubosa a las 9 am y 3 pm.

-  **Temp9am** y **Temp3pm**: Temperatura a las 9 am y 3 pm.

-  **RainToday**: Indica si llovió el día actual (**Yes** o **No**).

  

Asegurarse que los valores sean coherentes y estén en el formato esperado para evitar errores durante la predicción.

  

2. Colocar este archivo en cualquier directorio local de tu pc.

  

---

  

### **Ejecutar el contenedor**

Para ejecutar el modelo y realizar predicciones:

```bash

docker  run  -v <RUTA_LOCAL_OUTPUT>:/app/output  -v <RUTA_LOCAL_CSV>:/app/input.csv  inferencia_modelo

```

  

- Reemplaza `<RUTA_LOCAL_OUTPUT>` con la ruta completa de la carpeta donde deseas guardar los resultados.

Ejemplo: `C:/Users/Francisco/Desktop/`

- Reemplaza `<RUTA_LOCAL_CSV>` con la ruta del archivo CSV de entrada.

Ejemplo: `C:/Users/Francisco/Desktop/user_input.csv`

  

#### Ejemplo de comando completo:

```bash

docker  run  -v  C:/Users/Francisco/Desktop:/app/output  -v  C:/Users/Francisco/Desktop/user_input.csv:/app/input.csv  inferencia_modelo

```

  

### **Resultados**

El contenedor:

1. Cargará los datos desde el archivo `input.csv`.

2. Realizará el preprocesamiento y la predicción.

3. Guardará un archivo llamado `predicciones_output.csv` en la carpeta especificada como salida.

  

---