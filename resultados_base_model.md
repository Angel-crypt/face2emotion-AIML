# Resultados del Modelo Base (`face2emotion_baseModel.ipynb`)

Este documento resume los resultados obtenidos con el modelo CNN entrenado desde cero para clasificar imágenes de rostros en 3 emociones: **Angry** (Enojado), **Happy** (Feliz) y **Sad** (Triste).

---

## Resumen de Métricas Generales

| Conjunto                | Pérdida (Loss) | Precisión (Accuracy) | F1-Score (Macro) |
| :---------------------- | :-------------: | :-------------------: | :--------------: |
| **Validación**   |     0.3977     |        87.87%        |      0.8793      |
| **Prueba (Test)** |     0.4086     |        86.93%        |      0.8699      |

---

## Reporte de Clasificación (Conjunto de Prueba)

El modelo muestra un excelente desempeño general, siendo la clase **Happy** la que tiene mejores métricas de precisión y recall.

```text
                   precision      recall    f1-score    support

       Angry        0.86         0.82      0.84         1522
       Happy       0.95         0.91      0.93         1523
         Sad          0.80         0.87      0.84         1522

    accuracy                                       0.87      4567
   macro avg            0.87      0.87      0.87      4567
weighted avg          0.87      0.87      0.87      4567
```

### Análisis por Clase:

* **Happy (Feliz):** Obtiene la precisión más alta (**95%**), lo que significa que cuando el modelo predice "Feliz", casi nunca se equivoca. El recall es de **91%**.
* **Sad (Triste):** Es la clase que presenta el segundo mayor recall (**87%**), lo que indica que el modelo es muy bueno detectando caras tristes, aunque tiene una precisión del **80%** (tiende a confundir otras emociones con tristeza).
* **Angry (Enojado):** Mantiene un comportamiento equilibrado con un F1-score de **0.84**.

---

## Estructura del Modelo CNN Base

El modelo base es una arquitectura secuencial liviana adaptada para imágenes de **48x48x1** (escala de grises):

1. **Bloque 1:** Conv2D (32 filtros, 3x3) + BatchNormalization + MaxPooling2D (2x2)
2. **Bloque 2:** Conv2D (64 filtros, 3x3) + BatchNormalization + MaxPooling2D (2x2)
3. **Bloque 3:** Conv2D (128 filtros, 3x3) + BatchNormalization + MaxPooling2D (2x2) + Dropout (0.3)
4. **Clasificador:** GlobalAveragePooling2D + Dense (128, ReLU) + Dropout (0.6) + Dense (3, Softmax)
