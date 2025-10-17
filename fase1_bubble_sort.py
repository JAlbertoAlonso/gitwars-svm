import numpy as np

# TODO: Implementar la pérdida hinge y su gradiente
def hinge_loss(w, b, X, y, C):
    '''
    Calcula la pérdida hinge regularizada del SVM lineal.
    L(w,b) = 0.5 * ||w||^2 + C * sum_i max(0, 1 - y_i * (w^T x_i + b))
    
    Parámetros
    ----------
    w : array-like de forma (n_features,) o (n_features, 1)
    b : escalar (float)
    X : array-like de forma (n_samples, n_features) o (n_features,)
    y : array-like de forma (n_samples,) con etiquetas en {-1, 1} o {0, 1}
    C : escalar (float)

    Retorna
    -------
    L : float
        Valor escalar de la pérdida.
    '''
    # Conversión de tipos y formas (robustez a listas, 1D, etc.)
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    b = float(b)
    C = float(C)

    # Asegurar que X sea 2D y compatible
    if X.ndim == 1:
        # Caso borde: un único ejemplo
        X = X.reshape(1, -1)
    n_samples, n_features = X.shape

    if w.shape[0] != n_features:
        raise ValueError(f"w tiene {w.shape[0]} features, pero X tiene {n_features} columnas.")

    if y.shape[0] != n_samples:
        raise ValueError(f"y tiene {y.shape[0]} muestras, pero X tiene {n_samples} filas.")

    # Normalizar etiquetas: permitir {0,1} -> {-1,1}
    y_unique = np.unique(y)
    if set(y_unique).issubset({0.0, 1.0}):
        y = np.where(y == 0.0, -1.0, 1.0)
    elif not set(y_unique).issubset({-1.0, 1.0}):
        raise ValueError("y debe contener solo etiquetas en {-1, 1} o {0, 1}.")

    # Márgenes y pérdidas hinge (vectorizado y estable)
    margins = y * (X.dot(w) + b)                 # shape: (n_samples,)
    losses = np.maximum(0.0, 1.0 - margins)      # hinge por muestra

    # Regularización + suma de pérdidas (usar float64 para estabilidad en lotes grandes)
    reg = 0.5 * np.dot(w, w)
    data_term = C * np.sum(losses, dtype=np.float64)

    return float(reg + data_term)


import pandas as pd
df=pd.read_csv('./data/train_linear.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
w = np.array([0.5, -0.2])
b = 0.1 
C = 1.0
loss = hinge_loss(w, b, X, y, C)
print("Hinge Loss:", loss)