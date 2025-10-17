def train_svm(X, y, C=1.0, eta=0.01, epochs=100, batch_size=1, seed=42):
    np.random.seed(seed)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []

    for epoch in range(epochs):
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X[idx], y[idx]

        for start in range(0, n, batch_size):
            end = start + batch_size
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]

            grad_w = np.zeros_like(w)
            grad_b = 0.0

            for x_i, y_i in zip(X_batch, y_batch):
                margin = y_i * (np.dot(w, x_i) + b)
                if margin >= 1:
                    grad_w += w
                    grad_b += 0
                else:
                    grad_w += w - C * y_i * x_i
                    grad_b += -C * y_i

            grad_w /= batch_size
            grad_b /= batch_size

            w -= eta * grad_w
            b -= eta * grad_b

        # Calcular L al final de cada época
        L = hinge_loss(w, b, X, y, C)
        losses.append(L)
        print(f"Epoch {epoch:03d}: L = {L:.4f}")

    return w, b, losses

df = pd.read_csv("train_linear.csv")
X = df[['x1', 'x2']].values
y = df['y'].values
C = 1.0
eta = 0.001
epochs = 100
w, b, losses = train_svm(X, y, C=C, eta=eta, epochs=epochs, batch_size=10)

y_pred = np.sign(X @ w + b)
accuracy = np.mean(y_pred == y)
print(f"\nAccuracy final: {accuracy*100:.2f}%")

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 4))
plt.plot(losses, 'b-', linewidth=2)
plt.title("Curva de costo total L a lo largo de las épocas")
plt.xlabel("Época")
plt.ylabel("Costo total L")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(X[y==1,0], X[y==1,1], c='b', label='Clase +1', edgecolor='k')
plt.scatter(X[y==-1,0], X[y==-1,1], c='r', label='Clase -1', edgecolor='k')

# Generar frontera wTx+b=0
x1_range = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
x2_boundary = -(w[0]*x1_range + b) / w[1]
plt.plot(x1_range, x2_boundary, 'k--', label='wᵀx + b = 0')

plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Frontera de decisión del SVM lineal")
plt.legend()
plt.grid(True)
plt.show()