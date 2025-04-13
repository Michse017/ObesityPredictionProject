# src/model.py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model(X, y):
    """
    Entrena un modelo de regresión lineal utilizando los datos proporcionados.
    """
    model = LinearRegression()
    model.fit(X, y)
    print("Modelo entrenado exitosamente.")
    return model

def evaluate_model(model, X, y):
    """
    Evalúa el modelo calculando el Error Cuadrático Medio (MSE).
    """
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print("Evaluación del modelo:")
    print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
    return mse
