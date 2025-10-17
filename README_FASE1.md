# 📘 Fase 1 — Implementación de `hinge_loss`

## 🎯 Objetivo
Implementar la **pérdida hinge regularizada** del SVM lineal.

---

## 🧩 Instrucciones

Crea un nuevo archivo llamado:

```
fase1_equipo.py
```

Dentro de él, implementa la función `hinge_loss()` de acuerdo con la siguiente plantilla:

```python
# TODO: Implementar la pérdida hinge y su gradiente
def hinge_loss(w, b, X, y, C):
    '''
    Calcula la pérdida hinge regularizada del SVM lineal.
    L = ???
    '''
    pass
```

---

## ✅ Criterios de aceptación mínimos

- La función debe devolver un escalar **L** coherente con la expresión:
  $
  L(w,b) = \tfrac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i (w^\top x_i + b))
  $
- Debe ser **estable** para lotes pequeños y grandes; no romper por `shape` ni tipos de datos.

---

## 🧠 Preguntas guía para el PR

1. ¿Qué penaliza el término $0.5\|w\|^2$ y qué controla $C$?  
2. ¿Cuándo la parte *hinge* contribuye con 0 al costo?

---

## 💾 Datos de prueba sugeridos

Usa el archivo `data/train_linear.csv`.  
Asegúrate de usar etiquetas $y \in \{-1, +1\}$.

---

> 📤 **Entrega:**  
> Sube tu archivo `fase1_equipo.py` mediante un Pull Request hacia este repositorio.
