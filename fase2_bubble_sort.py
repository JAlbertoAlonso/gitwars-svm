import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


rng = np.random.RandomState(4)
if os.path.exists("train_linear.csv"):
    df = pd.read_csv("train_linear.csv")
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    # intenta x1, x2, y; si no, toma dos primeras numéricas como X y la última numérica como y
    if {'x1','x2','y'}.issubset(set(cols)):
        X = df[['x1','x2']].to_numpy(dtype=float)
        y = df['y'].to_numpy(dtype=float)
    else:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) < 3:
            raise ValueError("El CSV debe tener ≥2 columnas numéricas de features y 1 de etiqueta.")
        X = df[num_cols[:2]].to_numpy(dtype=float)
        y = df[num_cols[-1]].to_numpy(dtype=float)
    # mapear etiquetas a {-1, +1}
    yu = np.unique(y)
    if set(yu).issubset({0.0, 1.0}):
        y = np.where(y == 0.0, -1.0, 1.0)
    elif not set(yu).issubset({-1.0, 1.0}):
        # si hay dos valores, mapéalos a {-1,+1}
        if len(yu) == 2:
            y = np.where(y == yu[0], -1.0, 1.0)
        else:
            raise ValueError("La etiqueta y debe tener solo 2 clases.")
    fuente = "train_linear.csv"
else:
    # Fallback sintético 
    n = 200
    X_pos = rng.randn(n//2, 2) + np.array([2.0, 2.0])
    X_neg = rng.randn(n//2, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n//2), -np.ones(n//2)])
    fuente = "synthetic_fallback (sube train_linear.csv para usar tus datos)"


# Hiperparámetros
C = 1.0
eta = 0.1          # tasa de aprendizaje
epochs = 80
batch_size = 32
seed = 42


n, d = X.shape
w = np.zeros(d, dtype=float)
b = 0.0
hist_L = []


rng = np.random.RandomState(seed)
for epoch in range(1, epochs + 1):
    # shuffle data
    idx = rng.permutation(n)
    Xs, ys = X[idx], y[idx]

    # minibatches
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = Xs[start:end]
        yb = ys[start:end]

        # margin_i = y_i * (w·x_i + b)
        margins = yb * (Xb @ w + b)

        # if margin >= 1 => grad_w = w ; grad_b = 0
        # else           => grad_w = w - C * y_i * x_i ; grad_b = - C * y_i
        mask_bad = margins < 1.0

        grad_w = w.copy()
        if np.any(mask_bad):
            grad_w = grad_w - C * (Xb[mask_bad].T @ yb[mask_bad])

        grad_b = 0.0
        if np.any(mask_bad):
            grad_b = - C * np.sum(yb[mask_bad])

        # updates
        w = w - eta * grad_w
        b = b - eta * grad_b

   
    margins_all = y * (X @ w + b)
    hinge = np.maximum(0.0, 1.0 - margins_all)
    hinge_loss = np.mean(hinge)
    L = 0.5 * np.dot(w, w) + C * hinge_loss
    hist_L.append(L)

# -------------------------
# Métrica (accuracy final)
pred = np.sign(X @ w + b)
pred[pred == 0] = 1.0
acc = np.mean(pred == y)

print(f"Fuente de datos: {fuente}")
print(f"n={n}, d={d}")
print(f"L final: {hist_L[-1]:.6f}")
print(f"Accuracy final: {acc:.4f}")
print(f"w = {w}, b = {b:.6f}")

# -------------------------
# Gráfica de L por época
plt.figure()
plt.plot(np.arange(1, len(hist_L)+1), hist_L)
plt.xlabel("Época")
plt.ylabel("Costo total L")
plt.title("Curva de L por época (SVM hinge con SGD)")
plt.tight_layout()
plt.show()

# -------------------------
# Frontera de decisión (2D)
if d == 2 and abs(w[1]) > 1e-12:
    xmin, xmax = X[:,0].min()-1, X[:,0].max()+1
    ymin, ymax = X[:,1].min()-1, X[:,1].max()+1
    xx = np.linspace(xmin, xmax, 200)
    yy = -(w[0]/w[1]) * xx - b / w[1]
    plt.figure()
    plt.scatter(X[y==1,0], X[y==1,1], label="Clase +1")
    plt.scatter(X[y==-1,0], X[y==-1,1], label="Clase -1")
    plt.plot(xx, yy, label="w^T x + b = 0")
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title(f"Frontera de decisión (acc={acc:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()