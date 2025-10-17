import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -utils
def hinge_loss(w, b, X, y):
    """
    Calcula el promedio de hinge loss sobre todas las muestras
    """
    scores = X @ w + b
    losses = np.maximum(0, 1 - y * scores)
    return np.mean(losses)

def plot_decision_boundary(w, b, X, y):
    """
    Grafica la frontera de decisión y los puntos
    """
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', alpha=0.7)
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    x_vals = np.linspace(x_min, x_max, 200)
    # Frontera w1*x1 + w2*x2 + b = 0 -> x2 = -(w1*x1 + b)/w2
    if w[1] != 0:
        y_vals = -(w[0]*x_vals + b)/w[1]
        plt.plot(x_vals, y_vals, 'k--', lw=2)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Frontera de decisión")
    plt.show()

# Entranamiento

def train_svm(X, y, C=1.0, eta=0.01, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    history = []

    for epoch in range(epochs):
        # Mezclar datos
        idx = np.random.permutation(n_samples)
        X_shuff = X[idx]
        y_shuff = y[idx]

        for i in range(n_samples):
            x_i = X_shuff[i]
            y_i = y_shuff[i]

            margin = y_i * (np.dot(w, x_i) + b)
            if margin >= 1:
                grad_w = w
                grad_b = 0
            else:
                grad_w = w - C * y_i * x_i
                grad_b = - C * y_i

            w -= eta * grad_w
            b -= eta * grad_b

        # Calcular pérdida total
        hinge = hinge_loss(w, b, X, y)
        L = 0.5 * np.dot(w, w) + C * hinge
        history.append(L)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {L:.4f}")

    return w, b, history

# Main

if __name__ == "__main__":
    # Cargar datos
    df = pd.read_csv("data/train_linear.csv")
    X = df[['x1','x2']].values
    y = df['y'].values
    #y = np.where(y==0, -1, 1)  # asegurar etiquetas -1, +1

    # Entrenamiento
    w, b, history = train_svm(X, y, C=1.0, eta=0.01, epochs=50)

    # Curva de pérdida
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Curva de pérdida hinge loss")
    plt.show()

    # Accuracy
    predictions = np.sign(X @ w + b)
    acc = np.mean(predictions == y)
    print(f"Accuracy final: {acc*100:.2f}%")

    # Frontera de decisión
    plot_decision_boundary(w, b, X, y)
