import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
def hinge_loss(w, b, X, y, C):
    """
    Calcula la pérdida hinge para SVM
    """
    reg_term = 0.5 * np.sum(w**2)
    margins = y * (np.dot(X, w) + b)
    hinge_term = np.maximum(0, 1 - margins)
    loss_term = C * np.mean(hinge_term)
    return reg_term + loss_term

def train_svm_dinosaurios(X, y, C=1.0, eta=0.01, epochs=100, batch_size=32):
    """
    Entrena un SVM lineal para clasificación de dinosaurios

    Parámetros:
    -----------
    X : array-like, shape (n_samples, n_features)
        Características de los dinosaurios
    y : array-like, shape (n_samples,)
        Etiquetas de clase (-1, 1)
    C : float
        Parámetro de regularización
    eta : float
        Tasa de aprendizaje
    epochs : int
        Número de épocas
    batch_size : int
        Tamaño del mini-batch
    """
    n_samples, n_features = X.shape

    # Inicialización
    w = np.zeros(n_features)
    b = 0
    losses = []

    for epoch in range(epochs):
        # Mezclar datos
        X_shuffled, y_shuffled = shuffle(X, y)
        epoch_losses = []

        # Entrenamiento por mini-batches
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Calcular márgenes
            margins = y_batch * (np.dot(X_batch, w) + b)

            # Calcular gradientes
            mask = margins < 1

            # Actualizar gradientes según la condición del margen
            grad_w = np.where(mask.reshape(-1, 1),
                            w - C * y_batch.reshape(-1, 1) * X_batch,
                            w)
            grad_b = np.where(mask, -C * y_batch, 0).sum()

            # Actualizar parámetros
            w = w - eta * grad_w.mean(axis=0)
            b = b - eta * grad_b / len(y_batch)

            # Calcular pérdida para este batch
            batch_loss = hinge_loss(w, b, X_batch, y_batch, C)
            epoch_losses.append(batch_loss)

        # Registrar pérdida promedio de la época
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch + 1}/{epochs}, Pérdida: {avg_loss:.4f}")

    return w, b, losses

def plot_training_curve(losses):
    """
    Grafica la curva de entrenamiento
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Curva de Entrenamiento SVM - Clasificación de Dinosaurios')
    plt.grid(True)
    plt.savefig('training_curve_dinosaurios.png')
    plt.close()

def evaluate_svm(X, y, w, b):
    """
    Evalúa el modelo en el conjunto de datos
    """
    predictions = np.sign(np.dot(X, w) + b)
    accuracy = np.mean(predictions == y)
    return accuracy

if __name__ == "__main__":
    # Generar datos sintéticos para demostración
    np.random.seed(42)
    n_samples = 100
    n_features = 2

    # Intentar cargar datos del archivo, si existe
    try:
        df = pd.read_csv("data/train_linear.csv")
        X = df[['x1','x2']].to_numpy()
        y = df['y'].to_numpy()
        print("Usando datos del archivo train_linear.csv")
    except FileNotFoundError:
        # Usar datos sintéticos si no se encuentra el archivo
        print("Archivo train_linear.csv no encontrado. Usando datos sintéticos...")
        X = np.random.randn(n_samples, n_features)
        y = np.sign(X[:, 0] + 0.1 * X[:, 1] + 0.1 * np.random.randn(n_samples))

    # Parámetros del modelo
    C = 1.0
    eta = 0.01
    epochs = 100
    batch_size = 32

    # Entrenar modelo
    w, b, losses = train_svm_dinosaurios(X, y, C, eta, epochs, batch_size)

    # Evaluar y mostrar resultados
    accuracy = evaluate_svm(X, y, w, b)
    print(f"\nExactitud final: {accuracy:.4f}")

    # Graficar curva de entrenamiento
    plot_training_curve(losses)


def plot_decision_boundary(X, y, w, b):
    plt.figure(figsize=(8, 6))
    
    # Graficar los puntos
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Clase +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Clase -1')
    
    # Crear una malla de puntos para visualizar la frontera
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)
    
    # Frontera de decisión w^T x + b = 0
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    plt.contourf(xx, yy, Z, levels=[-float('inf'), 0, float('inf')], colors=['#FFAAAA', '#AAAAFF'], alpha=0.3)
    
    plt.title('Frontera de decisión SVM')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.savefig('decision_boundary.png')
    plt.close()
