# 🧮 Fase 4 — Idea dual + Kernel RBF

## 🎯 Objetivo
Reproducir un experimento guiado en datos **no lineales** usando la **formulación dual del SVM** con un **kernel RBF**, analizando el efecto de los hiperparámetros $γ$ y $C$.

---

## 🧩 Procedimiento

1. **Definir el kernel RBF**  
   Establecer la función de similitud entre dos ejemplos $x$ y $z$.
   
   $
   K(x, z) = \exp(-\gamma \|x - z\|^2)
   $  
   donde $\gamma > 0$ controla la flexibilidad del modelo.  

2. **Construir la matriz de Gram**  
   Cada elemento $K_{ij}$ representa la similitud entre los ejemplos $x_i$ y $x_j$.

3. **Resolver el problema dual:**  
   $
   \max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K_{ij}
   $
   sujeto a: $\sum_i \alpha_i y_i = 0,\ 0 ≤ \alpha_i ≤ C$.

4. **Calcular el sesgo del modelo:**  
   $
   b = \text{promedio}\left( y_k - \sum_i \alpha_i y_i K(x_i, x_k) \right)
   $

5. **Definir la función de decisión:**  
   $
   f(x) = \sum_i \alpha_i y_i K(x, x_i) + b
   $

6. **Visualizar los resultados:**  
   Graficar la frontera $f(x)=0$ y resaltar los vectores soporte.  
   Explorar diferentes valores de $γ$ y $C$.

---

## ✅ Criterios de aceptación

- Mostrar la **frontera de decisión** sobre un conjunto bidimensional no lineal (`train_nonlinear.csv`).  
- Identificar visualmente los **vectores soporte**.  
- Incluir un breve análisis del efecto de $γ$ y $C$.

---

## 🧠 Preguntas guía para el PR

1. ¿Qué sucede al aumentar $γ$ en la suavidad de la frontera?  
2. ¿Cómo interactúan $C$ y $γ$ en el sobreajuste o subajuste?

---

> 📤 **Entrega:**  
> Sube tu archivo `fase4_equipo.py` y tus visualizaciones mediante un Pull Request.
