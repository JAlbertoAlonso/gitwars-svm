import numpy as np
import matplotlib.pyplot as plt


# Vectores soporte
margins = y * (X @ w + b)
support_idx = np.where(margins <= 1 + 1e-6)[0] 
support_vectors = X[support_idx]

plt.figure(figsize=(7,7))

plt.scatter(X[y==1,0], X[y==1,1], c='blue', label='Clase +1', edgecolor='k')
plt.scatter(X[y==-1,0], X[y==-1,1], c='red', label='Clase -1', edgecolor='k')

# Frontera y márgenes
x1_range = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)
x2_decision = -(w[0]*x1_range + b)/w[1]           
x2_margin_pos = -(w[0]*x1_range + b - 1)/w[1]     
x2_margin_neg = -(w[0]*x1_range + b + 1)/w[1]     

plt.plot(x1_range, x2_decision, 'k-', linewidth=2, label='Frontera: wᵀx + b = 0')
plt.plot(x1_range, x2_margin_pos, 'k--', linewidth=1, label='Margen: wᵀx + b = +1')
plt.plot(x1_range, x2_margin_neg, 'k--', linewidth=1, label='Margen: wᵀx + b = -1')

# Vectores soporte
plt.scatter(support_vectors[:,0], support_vectors[:,1],
            s=100, facecolors='none', edgecolors='lime', linewidths=2, label='Vectores soporte')

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Frontera, márgenes y vectores soporte del SVM")
plt.legend()
plt.grid(True)
plt.show()


print("Número de vectores soporte:", len(support_vectors))

"""
Los vectores soporte son los puntos donde y_i (wᵀx_i + b) ≤ 1.
   Estos están sobre el margen o dentro del área de violación.

En el SVM lineal, solo estos puntos 'soportan' la solución
   porque definen el margen óptimo: si los movemos, la frontera cambia.
   Los puntos más alejados del margen no influyen en w ni en b.

El ancho del margen está dado por:
      margen = 2 / ||w||
   Por lo tanto, un ||w|| grande (pesos más grandes) implica un margen más angosto,
   mientras que un ||w|| pequeño produce un margen más ancho.
"""