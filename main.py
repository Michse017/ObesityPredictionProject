# main.py
import matplotlib.pyplot as plt
from src.data_preprocessing import load_data, preprocess_data
from src.model import train_model, evaluate_model
from sklearn.model_selection import train_test_split

def plot_feature_importance(model, feature_names):
    coefficients = model.coef_
    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, coefficients, color='skyblue')
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance (Linear Regression Coefficients)')
    plt.axvline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_regression_results(y_true, y_pred, title="Regression Results"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='royalblue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # Línea ideal
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Ruta del archivo de datos
    data_path = "data/ObesityDataSet.csv"
    
    # Cargar el dataset
    data = load_data(data_path)
    print(f"Datos cargados exitosamente.")
    print(f"Total de filas en el dataset original: {len(data)}")
    
    # Preprocesar los datos
    X, y = preprocess_data(data)
    print(f"Total de filas después del preprocesamiento: {len(X)}")

    # División en 70% entrenamiento, 25% prueba, 5% evaluación
    # Paso 1: Separar 5% para evaluación final
    X_temp, X_eval, y_temp, y_eval = train_test_split(X, y, test_size=0.05, random_state=42)

    # Paso 2: De los datos restantes (95%), separar 25/95 ≈ 0.2632 para prueba
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2632, random_state=42)

    print(f"Número de datos para entrenamiento: {len(X_train)}")
    print(f"Número de datos para prueba (test): {len(X_test)}")
    print(f"Número de datos para evaluación final: {len(X_eval)}")

    # Entrenar el modelo
    model = train_model(X_train, y_train)

        # Mostrar la importancia de las variables (coeficientes) del modelo
    print("\nImportancia de características (coeficientes del modelo):")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")


    # Evaluar el modelo en el conjunto de prueba
    print("Evaluación en conjunto de prueba:")
    evaluate_model(model, X_test, y_test)

    # Evaluar el modelo en el conjunto de evaluación final
    print("Evaluación en conjunto de evaluación final:")
    evaluate_model(model, X_eval, y_eval)

    y_pred_test = model.predict(X_test)
    plot_regression_results(y_test, y_pred_test, title="Regression Results on Test Set")

    y_pred_eval = model.predict(X_eval)
    plot_regression_results(y_eval, y_pred_eval, title="Regression Results on Final Evaluation Set")

if __name__ == '__main__':
    main()
