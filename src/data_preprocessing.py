import pandas as pd

def load_data(file_path):
    """
    Carga el archivo CSV y devuelve un DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print("Datos cargados exitosamente.")
        return data
    except Exception as e:
        print("Error al cargar los datos:", e)
        raise

def convert_target(y):
    """
    Convierte la variable objetivo 'NObeyesdad' a valores numéricos.
    
    Ajusta el diccionario de mapeo según los valores reales que contenga el dataset.
    """
    mapping = {
        "Insufficient_Weight": 0,
        "Normal_Weight": 1,
        "Overweight_Level_I": 2,
        "Overweight_Level_II": 3,
        "Obesity_Type_I": 4,
        "Obesity_Type_II": 5,
        "Obesity_Type_III": 6
    }
    if y.dtype == 'object':
        unique_vals = set(y.unique())
        if unique_vals.issubset(set(mapping.keys())):
            y = y.map(mapping)
        else:
            raise ValueError(f"Los valores de NObeyesdad no corresponden al mapeo esperado: {unique_vals}")
    return y

def preprocess_data(data):
    """
    Preprocesa el DataFrame para el modelo.
    
    Se asume que el dataset contiene las columnas:
    Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC,
    SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS, NObeyesdad

    La variable objetivo es 'NObeyesdad'.
    
    Se convierten las variables categóricas a variables dummy y se asegura que las demás columnas sean numéricas.
    Finalmente, se imputa cualquier valor faltante.
    """
    target = 'NObeyesdad'
    if target not in data.columns:
        raise ValueError(f"La columna objetivo '{target}' no se encuentra en los datos.")
    
    # Seleccionar las características: todas las columnas excepto la variable objetivo.
    features = data.columns.drop(target)
    
    # Crear X e y
    X = data[features].copy()
    y = data[target].copy()
    
    # Definir las columnas conocidas como categóricas (con valores no numéricos)
    categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'MTRANS']
    
    # Convertir a string las columnas categóricas para asegurar la correcta codificación.
    for col in categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype(str)
    
    # Convertir las columnas categóricas en variables dummy
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Convertir el resto de columnas que aún sean objeto a numérico; valores no convertibles se harán NaN.
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Imputar valores faltantes usando la media de la columna
    X.fillna(X.mean(), inplace=True)
    
    # Si aún existen NaN, se completan con 0 (alternativamente se podrían eliminar registros con dropna())
    if X.isnull().sum().sum() > 0:
        print("Aún existen valores NaN después de imputar con la media. Se completan con 0.")
        X.fillna(0, inplace=True)
    
    # Convertir la variable objetivo a valores numéricos mediante el mapeo
    y = convert_target(y)
    
    return X, y
