import numpy as np
import pandas as pd

def hinge_loss(w, b, X, y, C):
    """
    Calcula la función de pérdida hinge para un clasificador lineal SVM.
    Parámetros:
    - w: vector de pesos (numpy array de forma (n_features,))
    - b: sesgo (float)
    - X: matriz de características (numpy array de forma (n_samples, n_features))
    - y: vector de etiquetas (numpy array de forma (n_samples,)), con valores en {-1, 1}
    - C: parámetro de regularización (float)
    Retorna:
    - L: valor de la función de pérdida hinge (float)
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    b = float(b)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    assert X.shape[0] == y.shape[0], "X e y deben tener el mismo número de muestras"
    assert X.shape[1] == w.shape[0], "Dimensiones incompatibles entre X y w"

    margins = y * (X @ w + b)
    hinge = np.maximum(0.0, 1.0 - margins)
    L = 0.5 * np.dot(w, w) + C * np.sum(hinge)
    return float(L)

# probamos la funcion 
data = pd.read_csv("./train_linear.csv")
w = np.array([0.5, -0.3]) 
b = 0.1 
C = 1.0 
X = data[['x1', 'x2']].values 
y = data['y'].values 


hinge_loss(w = w, b = b, X = X,y = y ,C = C )