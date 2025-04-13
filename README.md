📄 README.txt

🧠 Proyecto: Predicción de Obesidad con Machine Learning

Este proyecto tiene como objetivo entrenar un modelo de aprendizaje automático que, a partir de datos personales y de estilo de vida, pueda predecir el nivel de obesidad de una persona.

El modelo se entrena pero **no se utiliza para hacer predicciones individuales** en esta fase. El enfoque está en el **proceso de entrenamiento y evaluación** del modelo.

---

📁 Estructura del proyecto

ObesityPredictionProject/
│
├── data/
│   └── ObesityDataSet.csv         # Dataset original con 2111 registros
│
├── src/
│   ├── data_preprocessing.py      # Carga y preprocesamiento de los datos
│   └── model.py                   # Entrenamiento y evaluación del modelo
│
├── main.py                        # Script principal que ejecuta todo
└── README.txt                     # Documentación del proyecto

---

⚙️ ¿Qué hace el código?

1. **Carga de datos:** Se cargan 2111 registros desde un archivo CSV.
2. **Preprocesamiento:**
   - Se convierten las variables categóricas en variables numéricas.
   - Se imputan valores faltantes con la media.
   - Si aún quedan NaN, se reemplazan con ceros.
3. **División de datos:**
   - 70% para entrenamiento (1477 registros)
   - 25% para prueba/test (528 registros)
   - 5% para evaluación final (106 registros)
4. **Entrenamiento:** Se entrena un modelo de regresión lineal con los datos de entrenamiento.
5. **Evaluación del modelo:**
   - Se evalúa el modelo en los datos de prueba y de evaluación final.
   - Se reporta el **Error Cuadrático Medio (MSE)** como métrica.

---

📊 Resultados de ejemplo

Total de filas en el dataset original: 2111  
Aún existen valores NaN después de imputar con la media. Se completan con 0.  
Total de filas después del preprocesamiento: 2111  
Número de datos para entrenamiento: 1477  
Número de datos para prueba (test): 528  
Número de datos para evaluación final: 106  
Modelo entrenado exitosamente.  
Evaluación en conjunto de prueba:  
Error Cuadrático Medio (MSE): 0.2023  
Evaluación en conjunto de evaluación final:  
Error Cuadrático Medio (MSE): 0.1976  

---

✅ ¿Cómo saber que el modelo se está entrenando?

- Si aparece el mensaje `Modelo entrenado exitosamente`, significa que el modelo fue alimentado con los datos de entrenamiento y completó el proceso sin errores.
- Los valores del MSE indican que el modelo fue capaz de ajustar una función a los datos y luego evaluarse con otros subconjuntos.
