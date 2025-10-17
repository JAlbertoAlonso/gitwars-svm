import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def rbf_kernel(X, Z, gamma):
    """
    K(x,z) = exp(-gamma * ||x - z||^2)
    X: (n, d), Z: (m, d)
    return: (n, m)
    """
    X2 = np.sum(X**2, axis=1)[:, None]
    Z2 = np.sum(Z**2, axis=1)[None, :]
    dists2 = X2 + Z2 - 2 * X @ Z.T
    return np.exp(-gamma * dists2)

def gram_matrix(X, gamma):
    return rbf_kernel(X, X, gamma)

def train_rbf_svm(X, y, C=1.0, gamma=1.0):
    """
    Entrena SVC(kernel='rbf') y devuelve:
    - w (no existe explícitamente en RBF; devolvemos None)
    - b (intercept_)
    - support_vectors (SVs)
    - support_idx (índices de SVs)
    - alphas (solo para los SVs)
    - dual_coef (vector con y_i * alpha_i para SVs)
    - clf (modelo entrenado)
    """
    clf = SVC(kernel='rbf', C=C, gamma=gamma)
    clf.fit(X, y)


    support_idx = clf.support_
    support_vectors = clf.support_vectors_
    dual_coef = clf.dual_coef_[0]            # y_sv * alpha_sv
    y_sv = y[support_idx]
    alphas = np.abs(dual_coef)               # alpha_sv >= 0
    b = float(clf.intercept_[0])

    eps = 1e-8
    margin_mask = (alphas > eps) & (alphas < C - eps)
    if np.any(margin_mask):
        K_sv = rbf_kernel(support_vectors, support_vectors[margin_mask], gamma)  
        b_est = y_sv[margin_mask] - (dual_coef @ K_sv)
        b = float(np.mean(b_est))

    return None, b, support_vectors, support_idx, alphas, dual_coef, clf

def decision_function(X_new, support_vectors, y_sv, dual_coef, b, gamma):
    K = rbf_kernel(X_new, support_vectors, gamma)  
    return K @ dual_coef + b

def plot_decision_boundary_rbf(X, y, support_vectors, y_sv, dual_coef, b, gamma,
                               title="SVM RBF: frontera, márgenes y SVs",
                               s=40, cmap_points=None):
    if cmap_points is None:
        # Colores simples: azul para +1, rojo para -1
        cmap_points = {1: 'blue', -1: 'red'}

    pad = 0.8
    x_min, x_max = X[:,0].min() - pad, X[:,0].max() + pad
    y_min, y_max = X[:,1].min() - pad, X[:,1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    Xgrid = np.c_[xx.ravel(), yy.ravel()]

    fvals = decision_function(Xgrid, support_vectors, y_sv, dual_coef, b, gamma)
    fvals = fvals.reshape(xx.shape)

    plt.figure(figsize=(7,7))
    plt.contour(xx, yy, fvals, levels=[0], linewidths=2, linestyles='-', colors='k')
    plt.contour(xx, yy, fvals, levels=[-1, 1], linewidths=1.5, linestyles='--', colors='k')

    plt.scatter(X[y==1,0], X[y==1,1], c=cmap_points[1], s=s, edgecolor='k', label='Clase +1')
    plt.scatter(X[y==-1,0], X[y==-1,1], c=cmap_points[-1], s=s, edgecolor='k', label='Clase -1')

    plt.scatter(support_vectors[:,0], support_vectors[:,1],
                s=120, facecolors='none', edgecolors='lime', linewidths=2, label='Vectores soporte')

    plt.title(title + f"\n(gamma={gamma}, b={b:.3f})")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)
    plt.show()

df = pd.read_csv("train_nonlinear.csv")
X = df[['x1', 'x2']].values
y = df['y'].values.astype(int)

C = 1.0
gamma = 1.0
_, b, SV, sv_idx, alphas_sv, dual_coef, clf = train_rbf_svm(X, y, C=C, gamma=gamma)

y_sv = y[sv_idx]

y_pred = clf.predict(X)
acc = (y_pred == y).mean()
print(f"Accuracy (train): {acc*100:.2f}%")
print(f"Número de vectores soporte: {SV.shape[0]}")

plot_decision_boundary_rbf(X, y, SV, y_sv, dual_coef, b, gamma,
                           title="SVM con kernel RBF (train_nonlinear.csv)")

configs = [
    (0.2, 0.5),
    (0.2, 10.0),
    (2.0, 0.5),
    (2.0, 10.0),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ax, (g, c) in zip(axes, configs):
    _, b_i, SV_i, sv_idx_i, alphas_i, dc_i, clf_i = train_rbf_svm(X, y, C=c, gamma=g)
    y_sv_i = y[sv_idx_i]
    # Malla
    pad = 0.8
    x_min, x_max = X[:,0].min() - pad, X[:,0].max() + pad
    y_min, y_max = X[:,1].min() - pad, X[:,1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Xgrid = np.c_[xx.ravel(), yy.ravel()]
    fvals = decision_function(Xgrid, SV_i, y_sv_i, dc_i, b_i, g).reshape(xx.shape)

    # Contornos
    cs0 = ax.contour(xx, yy, fvals, levels=[0], linewidths=2, linestyles='-', colors='k')
    cs1 = ax.contour(xx, yy, fvals, levels=[-1, 1], linewidths=1.5, linestyles='--', colors='k')

    # Puntos y SV
    ax.scatter(X[y==1,0], X[y==1,1], c='blue', s=25, edgecolor='k')
    ax.scatter(X[y==-1,0], X[y==-1,1], c='red',  s=25, edgecolor='k')
    ax.scatter(SV_i[:,0], SV_i[:,1], s=90, facecolors='none', edgecolors='lime', linewidths=2)

    acc_i = (clf_i.predict(X) == y).mean()
    ax.set_title(f"γ={g}, C={c} | SV={SV_i.shape[0]}, Acc={acc_i*100:.1f}%")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(True)

plt.tight_layout()
plt.show()