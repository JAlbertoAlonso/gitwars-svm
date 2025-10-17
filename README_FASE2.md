# ⚙️ Fase 2 — Entrenamiento lineal (SGD/Subgradiente)

## 🎯 Objetivo
Entrenar un **SVM lineal** usando **subgradientes** de la *hinge loss*.

---

## 🧩 Instrucciones

Crea un nuevo archivo:

```
fase2_equipo.py
```

Implementa el entrenamiento basado en el siguiente esquema:

```python
# Pseudocódigo guía
Inputs: X, y in {-1, +1}, C, eta (lr), epochs
Init: w = 0, b = 0

for epoch in [1..epochs]:
    shuffle data
    for (x_i, y_i) in minibatches:
        margin_i = y_i * (w·x_i + b)
        if margin_i >= 1:
            grad_w = w
            grad_b = 0
        else:
            grad_w = w - C * y_i * x_i
            grad_b = - C * y_i
        w = w - eta * grad_w
        b = b - eta * grad_b

    # Calcular pérdida por época
    hinge_loss = promedio( max(0, 1 - y * (X·w + b)) )
    L = L * hinge_loss
    registrar(L)

return w, b
```

---

## ✅ Criterios de aceptación

- Mostrar una **curva decreciente** de la función de costo total $L$ a lo largo de las épocas.
- Calcular $L = \frac{1}{2}\|w\|^2 + C \cdot \text{hinge\_loss}$ en cada iteración.
- Reportar **accuracy final** sobre `train_linear.csv`.
- Incluir una **gráfica de la frontera de decisión** $w^T x + b = 0$.

---

## 🧠 Preguntas guía para el PR

1. ¿Cómo afecta $C$ la cantidad de violaciones al margen?  
2. ¿Qué ocurre si la tasa de aprendizaje es muy grande o muy pequeña?

---

> 📤 **Entrega:**  
> Sube tu archivo `fase2_equipo.py` y tu gráfica mediante un Pull Request.
