# svm_plot_demo.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def hinge_loss(w, b, X, y, C):
    """
    L(w,b) = 0.5||w||^2 + C * sum_i max(0, 1 - y_i (w^T x_i + b))
    Devuelve: L (escalar), dW (como w), dB (escalar)
    """
    X = np.asarray(X, float)           # (N, D)
    y = np.asarray(y, float).reshape(-1)  # (N,)
    w = np.asarray(w, float).reshape(-1)  # (D,)
    b = float(b)

    scores  = X @ w + b                # (N,)
    margins = 1.0 - y * scores         # (N,)
    hinge   = np.maximum(0.0, margins) # (N,)

    L  = 0.5 * w.dot(w) + C * hinge.sum()

    active = (margins > 0).astype(float)   # 1 si contribuye
    coeff  = -y * active                   # d(hinge)/d(score)

    dW = w + C * (X.T @ coeff)
    dB = C * coeff.sum()
    return L, dW, dB


if __name__ == "__main__":
    # === Carga (misma lógica que pasaste) ===
    df = pd.read_csv("train_linear.csv")

    try:
        data = np.genfromtxt("train_linear.csv", delimiter=",", dtype=float, skip_header=1)
        if np.isnan(data).any():
            raise ValueError
    except Exception:
        data = np.genfromtxt("train_linear.csv", delimiter=",", dtype=float)

    X = data[:, :-1]               # (N, 2) esperado
    y = data[:, -1]                # {-1, +1}

    # ====== Inicialización y un paso de GD (como tu main) ======
    w = np.zeros(X.shape[1])
    b = 0.0
    C = 1.0

    L, dW, dB = hinge_loss(w, b, X, y, C)
    print(f"N={len(y)}, D={X.shape[1]}")
    print(f"Pérdida inicial: {L:.6f}")
    print(f"||dW||: {np.linalg.norm(dW):.6f}, dB: {dB:.6f}")

    # Un paso de GD (puedes subir epochs si quieres una frontera más estable)
    lr = 1e-2
    w -= lr * dW
    b -= lr * dB
    L2, _, _ = hinge_loss(w, b, X, y, C)
    print(f"Pérdida tras 1 paso GD: {L2:.6f}")

    # ====== Subtareas de visualización ======
    if X.shape[1] != 2:
        raise ValueError("Para graficar la frontera, el dataset debe tener 2 features.")

    # 3) Candidatos a vectores soporte: y_i (w^T x_i + b) <= 1
    margins = y * (X @ w + b)
    sv_mask = margins <= 1 + 1e-12
    num_sv = int(sv_mask.sum())
    print(f"Candidatos a vectores soporte: {num_sv} puntos")

    # 1) Datos + recta de decisión w^T x + b = 0
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    xs = np.linspace(x_min, x_max, 300)

    def line_y(c):
        # w1*y = c - b - w0*x  ->  y = (c - b - w0*x)/w1
        if abs(w[1]) < 1e-12:
            return None  # vertical
        return (c - b - w[0]*xs) / (w[1] + 1e-12)

    y_dec = line_y(0)     # w^T x + b = 0
    y_m1 = line_y(1)      # margen +1
    y_p1 = line_y(-1)     # margen -1

    plt.figure(figsize=(6,5))

    # puntos por clase
    pos = y == 1
    plt.scatter(X[pos,0], X[pos,1], label="+1", alpha=0.8)
    plt.scatter(X[~pos,0], X[~pos,1], label="-1", alpha=0.8)

    # 3) resaltar candidatos a SV (borde negro y tamaño mayor)
    plt.scatter(X[sv_mask,0], X[sv_mask,1],
                facecolors='none', edgecolors='k', s=120, linewidths=1.5,
                label="y_i (w^T x_i + b) ≤ 1")

    # 1) y 2) rectas
    if y_dec is None:
        # frontera vertical en x = -(b)/w0
        x_vert = -(b) / (w[0] + 1e-12)
        plt.axvline(x_vert, linestyle='-', label="w^T x + b = 0")
        # márgenes ±1: x = (±1 - b)/w0
        plt.axvline((1 - b) / (w[0] + 1e-12), linestyle='--', label="+1")
        plt.axvline((-1 - b) / (w[0] + 1e-12), linestyle='--', label="-1")
    else:
        plt.plot(xs, y_dec, '-',  label="w^T x + b = 0")
        plt.plot(xs, y_m1, '--', label="w^T x + b = +1")
        plt.plot(xs, y_p1, '--', label="w^T x + b = -1")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Datos, frontera y márgenes; candidatos a vectores soporte")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()