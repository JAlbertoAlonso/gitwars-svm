# 🧠 Laboratorio 6: Git Wars — SVMs

**Duración:** 2 horas  
**Formato:** Competencia por Pull Requests  

---

## 🎯 Objetivo

Que los equipos integren teoría y práctica de las **Máquinas de Vectores de Soporte (SVM)** mediante una dinámica tipo **Git Wars**, utilizando buenas prácticas de **GitHub**: forks, ramas y Pull Requests.

El repositorio base incluye únicamente el **cuaderno template** y la **data** necesaria.  
Las instrucciones, pseudocódigo y preguntas teóricas se publicarán gradualmente en archivos `README.md` por fase.  
Cada equipo deberá implementar los componentes solicitados en su **fork**, y al final sincronizar con el repositorio principal para consolidar todo el material.

---

## 🕒 Estructura detallada de la sesión

| Tiempo | Actividad |
|:------:|:----------|
| 0:00–0:10 | Introducción a la dinámica *Git Wars*, flujo de trabajo con forks y PRs, y breve repaso teórico de SVM (margen, C, hinge loss, kernel). |
| 0:10–0:25 | **Fase 1 — Implementación de `hinge_loss`.** |
| 0:25–1:00 | **Fase 2 — Entrenamiento lineal (SGD/Subgradiente).** |
| 1:00–1:25 | **Fase 3 — Visualización y análisis del margen.** |
| 1:25–1:50 | **Fase 4 — Idea dual + Kernel RBF.** |
| 1:50–2:00 | Cierre general: sincronización final (`git pull`), consolidación y conclusiones. Anuncio de equipos con PRs validados. |

---

## ⚙️ Requisitos técnicos y datos

- **Stack sugerido:** `python 3.10+`, `numpy`, `matplotlib`.  
  (Opcional: `scikit-learn` para validación de métricas y visualizaciones.)
- **Datos incluidos:**  
  - `data/train_linear.csv`  
  - `data/train_nonlinear.csv`  
  - `data/hidden_test.csv`
- **Etiquetas:** usar \( y \in \{-1, +1\} \).

---

## ⚔️ Dinámica *Git Wars*

1. Cada equipo hace **fork** del repositorio base y crea una **rama por fase** (por ejemplo: `fase1/equipo`).
2. Implementan el contenido de la fase y abren un **Pull Request (PR)** hacia el repo principal.
3. El PR se **revisa y mergea** si:
   - pasa validaciones básicas,
   - el código es claro y funcional,
   - y cumple los criterios de aceptación.
4. El **orden de merge** alimenta el *ranking en vivo*.
5. Al final, todos realizan `git pull` del `main` consolidado y entregan su **cuaderno final** (Jupyter Notebook o PDF).

---

## 📦 Entregables por equipo

- **Pull Requests por fase:**  
  - Código funcional y validado en su fork.  
  - Evidencias breves: figuras, métricas y respuestas a preguntas guía en el cuerpo del PR.

- **Cuaderno final consolidado:**  
  - Resultados globales (fronteras, márgenes, análisis de C y γ).  
  - Conclusiones interpretativas y figura final comparativa.

---

## 🧩 Reglas para los Pull Requests

1. Un **PR por fase**, desde una rama con nombre claro (`fase2/equipo`).
2. En cada PR incluir:
   - Explicación corta y observaciones sobre hiperparámetros.  
   - Figuras o fragmentos representativos de los resultados.
3. El PR se mergea únicamente si pasa validaciones y cumple criterios.  
   El orden de merge determina la posición en el *ranking*.
4. Mantener el código **modular, legible y comentado**.  
   No subir datos externos ni archivos binarios pesados.

---

## 🧮 Evaluación

| Criterio | Ponderación |
|-----------|-------------|
| Cuaderno final consolidado (resultados, figuras y conclusiones integradas) | **65 %** |
| Bitácora de control de versiones y ramas (documentación de PRs por fase) | **15 %** |
| Respuestas teóricas y análisis (preguntas guía, interpretación de márgenes y kernels) | **20 %** |
| **Total** | **100 %** |

---

### 📓 Bitácora de control de versiones (15 %)

Cada equipo deberá incluir dentro de su cuaderno final una subsección Markdown titulada  
**“Bitácora de control de versiones — Fase X”**, que contenga:

- **Rama utilizada:** nombre exacto (`fase2/equipo`).  
- **Commit representativo:** mensaje más relevante.  
- **Pull Request:** enlace al PR o captura de pantalla.  
- **Resumen de cambios:** 2–3 líneas sobre qué se implementó y qué dificultades surgieron.  
- *(Opcional)* **Retroalimentación y ajustes:** qué se modificó tras revisión.

**Criterios de evaluación:**
- Integridad (todas las fases documentadas).  
- Claridad (mensajes comprensibles).  
- Evidencia (PR o captura).  
- Coherencia entre lo descrito y el contenido del cuaderno.

---

### 🧱 Estructura esperada del cuaderno por fase

Cada fase dentro del cuaderno debe contener tres secciones:

1. **Código:** celdas de ejecución (implementación, entrenamiento o visualización).  
2. **Análisis:** celdas Markdown con interpretaciones y respuestas teóricas.  
3. **Bitácora de control de versiones:** subsección Markdown con rama, commit y PR.

---

## 🏆 Rúbrica de la competencia (Ranking en vivo)

| Fase | 1er lugar | 2do | 3ro | 4to o más |
|------|------------|------|------|-----------|
| Fase 1 — `hinge_loss` (coherencia teórica) | 15 | 10 | 5 | 0 |
| Fase 2 — Entrenamiento lineal (SGD/Subgradiente) | 25 | 20 | 15 | 0 |
| Fase 3 — Visualización del margen y vectores soporte | 25 | 20 | 15 | 0 |
| Fase 4 — Kernel RBF (dual guiado) | 25 | 20 | 15 | 0 |

**Interpretación:**
- La nota base proviene de los criterios técnicos y teóricos.  
- El ranking otorga puntos extra según el orden de aceptación de los PRs.  
- En caso de empate, se prioriza la calidad del código y documentación.  
- Los puntos del ranking pueden servir como bonificación o desempate académico.

---

## 📘 Apéndice A — Recordatorio teórico mínimo

- **Función objetivo (forma primal):**  
  L(w,b)=½‖w‖² + CΣᵢmax(0,1−yᵢ(wᵗxᵢ+b))

- **Subgradiente de la pérdida hinge:**  
  Si yᵢ(wᵗxᵢ+b)≥1 → ∇w=w, ∇b=0  
  Si yᵢ(wᵗxᵢ+b)<1 → ∇w=w−Cyᵢxᵢ, ∇b=−Cyᵢ

- **Frontera y márgenes lineales:**  
  Frontera: wᵗx+b=0  
  Márgenes: wᵗx+b=±1  
  Puntos con yᵢ(wᵗxᵢ+b)≤1 son **vectores soporte**.

- **Formulación dual (resumen):**  
  max_α Σᵢαᵢ−½ΣᵢⱼαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)  
  sujeto a Σᵢαᵢyᵢ=0, 0≤αᵢ≤C.

- **Kernel RBF:**  
  K(x,z)=exp(−γ‖x−z‖²)

- **Función de decisión dual:**  
  f(x)=ΣᵢαᵢyᵢK(xᵢ,x)+b

---

## 🧭 Apéndice B — Checklist de Git (forks & PRs)

### 1. Hacer fork del repositorio base
- Ingresar al repositorio original del laboratorio en GitHub y pulsar el botón **Fork**.  
- Verificar que el nuevo repositorio aparezca bajo su cuenta o la de su equipo.  
- Confirmar que el nombre sea claro, por ejemplo: `<usuario>/gitwars-svm`.

### 2. Clonar el fork en su máquina local
```bash
git clone https://github.com/<usuario>/gitwars-svm.git
cd gitwars-svm
git remote -v
```
Esto asegura que el *remote origin* apunte correctamente a su fork.

### 3. Crear una rama dedicada para cada fase
```bash
git checkout -b faseX/equipo
```
- Evitar trabajar directamente sobre `main`.  
- Una rama por fase permite PRs independientes y trazabilidad limpia.

### 4. Implementar los cambios correspondientes a la fase
- Añadir código nuevo, visualizaciones o documentación según las instrucciones de la fase.  
- Comentar el código brevemente y verificar que todo ejecute sin errores.  
- Antes de confirmar cambios, revisar:
```bash
git status
```

### 5. Guardar y subir cambios al fork
```bash
git add [nombredelequipo_faseX].py  <- archivo a mergear
git commit -m "feat(fase2): entrenamiento lineal con SGD"
git push origin fase2/equipo
```
- Mantener mensajes de commit **breves pero descriptivos**.  
  Prefijos sugeridos: `feat:`, `fix:`, `refactor:`, `docs:`.

### 6. Abrir un Pull Request (PR) hacia el repo base
- En GitHub, pulsar **Compare & Pull Request**.  
- Base: `main` del repositorio original (`upstream`).  
- Compare: la rama de su fork.  
- En la descripción del PR incluir:
  - Resumen de lo implementado.  
  - Figuras o resultados clave.  
  - Respuestas a las preguntas guía.  
- Etiquetar el PR con el nombre de la fase (por ejemplo, `fase3-equipo`).

### 7. Atender retroalimentación y esperar el merge
- El docente revisará que:
  1. El PR cumpla los criterios técnicos.  
  2. El código sea legible, ejecutable y consistente.  
  3. Incluya las figuras y respuestas requeridas.  
- Si hay comentarios, corregir localmente y hacer `git push` nuevamente sobre la misma rama (el PR se actualiza automáticamente).  
- Una vez aprobado, el PR se integrará al `main` del repo base y se actualizará el ranking de equipos.

### 8. Sincronizar el fork con los avances del repositorio base
```bash
git checkout main
git pull upstream main
git push origin main
```
- Esto mantiene el fork actualizado con el contenido consolidado.  
- Antes de iniciar la siguiente fase, crear la nueva rama desde el `main` actualizado.

### 9. Buenas prácticas y errores comunes a evitar
- No subir archivos grandes (`.zip`, `.ipynb_checkpoints`, etc.) ni datasets externos.  
- No usar commits genéricos como "update" o "final".  
- No abrir PRs desde `main`; siempre desde una rama de fase.  
- Resolver conflictos de merge antes de subir.  
- Usar nombres de archivo y carpetas sin espacios ni acentos.

### 10. (Opcional) Trabajar con interfaz gráfica (VS Code o web)
- En **VS Code**, usar la pestaña *Source Control*:  
  - “+” para *Stage changes*  
  - Escribir el mensaje de commit.  
  - Botón “Sync Changes” para subir a GitHub.  
- También es válido usar la interfaz web siempre que el flujo de commits esté documentado.

### 11. Cierre general de la práctica
- Confirmar que todas las ramas de fase tengan su PR mergeado.  
- Ejecutar un último:
```bash
git pull upstream main
```
- Generar el **cuaderno final consolidado** con los resultados y subirlo al repositorio.


---

# 🔄 Guía de Sincronización de Fases — Git Wars SVMs

Cada vez que el docente publique una nueva fase (por ejemplo, `README_fase2.md`), debes **sincronizar tu fork** con el repositorio base para obtener los nuevos archivos antes de comenzar.

---

## 🧩 1. Cambiar a la rama `main`

Antes de actualizar tu repositorio, asegúrate de estar en `main`:

```bash
git checkout main
```

---

## 📨 2. Descargar los cambios del repositorio base (upstream)

Esto trae las actualizaciones del docente (nuevos READMEs o archivos).

```bash
git pull upstream main
```

Si es la primera vez que configuras el remoto `upstream`, usa:

```bash
git remote add upstream https://github.com/<docente>/gitwars-svm.git
```

---

## ⬆️ 3. Subir la actualización también a tu fork

Después de jalar los cambios del docente, súbelos a tu propio repositorio en GitHub:

```bash
git push origin main
```

---

## 🌱 4. Crear una nueva rama para la siguiente fase

Crea una rama específica para trabajar la nueva fase (nunca trabajes en `main`).

```bash
git checkout -b faseX/equipo
```

Ejemplo:

```bash
git checkout -b fase2/team-alpha
```

---

## ✅ 5. Flujo completo de actualización

| Paso | Comando | Descripción |
|------|----------|-------------|
| 1 | `git checkout main` | Cambiar a la rama principal |
| 2 | `git pull upstream main` | Descargar actualizaciones del docente |
| 3 | `git push origin main` | Subir los cambios al fork |
| 4 | `git checkout -b faseX/equipo` | Crear una nueva rama para trabajar |
| 5 | *Implementar y abrir PR* | Enviar tus avances al repo base |

---

## ⚠️ Notas importantes

- Nunca edites el `main` directamente.  
- Siempre sincroniza tu fork **antes de crear la rama de la siguiente fase**.  
- Si tienes conflictos al hacer `pull upstream main`, revisa qué archivos cambiaste y resuélvelos antes de continuar.  
- Verifica con `git branch` en qué rama estás antes de trabajar.  

---

© Facultad de Ingeniería — Laboratorio de Aprendizaje Automático · 2025

