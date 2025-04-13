# main.py
from src.data_preprocessing import load_data, preprocess_data
from src.model import train_model, evaluate_model
from sklearn.model_selection import train_test_split

def main():
    # Ruta del archivo de datos
    data_path = "data/ObesityDataSet.csv"
    
    # Cargar el dataset
    data = load_data(data_path)
    
    # Preprocesar los datos: se extraen las características (X) y la variable objetivo (y)
    X, y = preprocess_data(data)
    
    # Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo de regresión lineal
    model = train_model(X_train, y_train)
    
    # Evaluar el modelo en el conjunto de prueba
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
