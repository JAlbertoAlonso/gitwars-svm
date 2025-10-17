import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Carga de datos (simple)
# -------------------------
rng = np.random.RandomState(4)
if os.path.exists("train_linear.csv"):
    df = pd.read_csv("train_linear.csv")
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if {'x1','x2','y'}.issubset(set(cols)):
        X = df[['x1','x2']].to_numpy(dtype=float)
        y = df['y'].to_numpy(dtype=float)
    else:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) < 3:
            raise ValueError("El CSV debe tener ≥2 columnas numéricas de features y 1 de etiqueta.")
        X = df[num_cols[:2]].to_numpy(dtype=float)
        y = df[num_cols[-1]].to_numpy(dtype=float)
    yu = np.unique(y)
    if set(yu).issubset({0.0, 1.0}):
        y = np.where(y == 0.0, -1.0, 1.0)
    elif not set(yu).issubset({-1.0, 1.0}):
        if len(yu) == 2:
            y = np.where(y == yu[0], -1.0, 1.0)
        else:
            raise ValueError("La etiqueta y debe tener solo 2 clases.")
    fuente = "train_linear.csv"
else:
    n = 200
    X_pos = rng.randn(n//2, 2) + np.array([2.0, 2.0])
    X_neg = rng.randn(n//2, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n//2), -np.ones(n//2)])
    fuente = "synthetic_fallback"

# -------------------------
# Hiperparámetros
# -------------------------
C = 1.0
eta = 0.1
epochs = 80
batch_size = 32
seed = 42

# -------------------------
# Init
# -------------------------
n, d = X.shape
w = np.zeros(d, dtype=float)
b = 0.0
hist_L = []

# -------------------------
# Entrenamiento
# -------------------------
rng = np.random.RandomState(seed)
for epoch in range(1, epochs + 1):
    idx = rng.permutation(n)
    Xs, ys = X[idx], y[idx]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = Xs[start:end]
        yb = ys[start:end]

        margins = yb * (Xb @ w + b)
        mask_bad = margins < 1.0

        grad_w = w.copy()
        if np.any(mask_bad):
            grad_w = grad_w - C * (Xb[mask_bad].T @ yb[mask_bad])
        grad_b = 0.0
        if np.any(mask_bad):
            grad_b = - C * np.sum(yb[mask_bad])

        w = w - eta * grad_w
        b = b - eta * grad_b

    # costo total L por época
    margins_all = y * (X @ w + b)
    hinge = np.maximum(0.0, 1.0 - margins_all)
    hinge_loss = np.mean(hinge)
    L = 0.5 * np.dot(w, w) + C * hinge_loss
    hist_L.append(L)

# -------------------------
# Métrica (accuracy final)
# -------------------------
pred = np.sign(X @ w + b)
pred[pred == 0] = 1.0
acc = np.mean(pred == y)

print(f"Fuente de datos: {fuente}")
print(f"n={n}, d={d}")
print(f"L final: {hist_L[-1]:.6f}")
print(f"Accuracy final: {acc:.4f}")
print(f"w = {w}, b = {b:.6f}")

# -------------------------
# Curva de L por época
# -------------------------
plt.figure()
plt.plot(np.arange(1, len(hist_L)+1), hist_L)
plt.xlabel("Época")
plt.ylabel("Costo total L")
plt.title("Curva de L por época (SVM hinge con SGD)")
plt.tight_layout()
plt.show()

# -------------------------
# Gráfico: datos, recta w^T x + b = 0, márgenes ±1,
# y resaltado de candidatos a SV (y_i(w^T x_i + b) <= 1)
# -------------------------
if d == 2:
    xmin, xmax = X[:,0].min()-1, X[:,0].max()+1
    ymin, ymax = X[:,1].min()-1, X[:,1].max()+1

    # Máscara de candidatos a SV
    margins_all = y * (X @ w + b)
    sv_mask = margins_all <= 1.0 + 1e-12

    plt.figure()
    # Datos base
    plt.scatter(X[y==1,0],  X[y==1,1],  label="Clase +1")
    plt.scatter(X[y==-1,0], X[y==-1,1], label="Clase -1")

    # Resaltar SV (círculos huecos grandes)
    plt.scatter(X[sv_mask,0], X[sv_mask,1],
                s=140, facecolors='none', edgecolors='k', linewidths=1.5,
                label="Candidatos a SV (y·margin ≤ 1)")

    # Líneas: decisión y márgenes
    if abs(w[1]) > 1e-12:
        xx = np.linspace(xmin, xmax, 400)
        # w0*x + w1*y + b = c  => y = -(w0/w1)x - (b-c)/w1
        yy0  = -(w[0]/w[1])*xx -  b/ w[1]        # c = 0
        yy_p = -(w[0]/w[1])*xx - (b-1)/w[1]      # c = +1
        yy_m = -(w[0]/w[1])*xx - (b+1)/w[1]      # c = -1
        plt.plot(xx, yy0,  label="w^T x + b = 0")
        plt.plot(xx, yy_p, linestyle="--", label="w^T x + b = +1")
        plt.plot(xx, yy_m, linestyle="--", label="w^T x + b = -1")
    else:
        # Rectas verticales: x = const
        # w0*x + b = c  => x = (c - b)/w0
        x0  = (0.0 - b) / w[0]
        xp1 = (1.0 - b) / w[0]
        xm1 = (-1.0 - b) / w[0]
        plt.axvline(x0,  label="w^T x + b = 0")
        plt.axvline(xp1, linestyle="--", label="w^T x + b = +1")
        plt.axvline(xm1, linestyle="--", label="w^T x + b = -1")

    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title(f"Decisión y márgenes (acc={acc:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()