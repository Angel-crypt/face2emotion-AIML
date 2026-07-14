# Resultados del Modelo de Fine-Tuning (`face2emotion_finetune.ipynb`)

Este documento resume los resultados obtenidos aplicando transferencia de aprendizaje (Transfer Learning) y ajuste fino (Fine-Tuning) sobre una red **ResNet50** pre-entrenada con ImageNet para clasificar imágenes de rostros en 3 emociones: **Angry** (Enojado), **Happy** (Feliz) y **Sad** (Triste).

---

## Resumen de Métricas Generales

| Conjunto                | Pérdida (Loss) | Precisión (Accuracy) | F1-Score (Macro) |
| :---------------------- | :-------------: | :-------------------: | :--------------: |
| **Validación**   |     1.3091     |        92.36%        |      0.9236      |
| **Prueba (Test)** |     1.2414     |        92.47%        |      0.9246      |

---

## Reporte de Clasificación (Conjunto de Prueba)

El modelo de Fine-Tuning muestra un desempeño sumamente robusto y equilibrado en todas las clases, reduciendo significativamente los errores de predicción.

```text
              precision    recall  f1-score   support

       Angry       0.91      0.91      0.91      1522
       Happy       0.95      0.95      0.95      1523
         Sad       0.92      0.91      0.91      1522

    accuracy                           0.92      4567
   macro avg       0.92      0.92      0.92      4567
weighted avg       0.92      0.92      0.92      4567
```

### Análisis por Clase:

* **Happy (Feliz):** Sigue siendo la clase más fuerte, con un F1-score de **0.95**. La precisión y el recall están perfectamente balanceados en **95%**.
* **Sad (Triste):** Mejoró drásticamente en comparación con el modelo base, subiendo su precisión al **92%** (antes 80%) y manteniendo un recall de **91%**, lo que demuestra que la transferencia de características de ResNet50 ayudó a disipar las falsas alarmas de tristeza.
* **Angry (Enojado):** Alcanzó un F1-score de **0.91** (un avance sustancial respecto al 0.84 del modelo base).

---

## Estructura del Modelo con Transfer Learning

El modelo utiliza un enfoque híbrido con procesamiento dinámico:

1. **Entrada del Dataset:** Imágenes cargadas en RAM en su resolución nativa de $48 \times 48 \times 3$ en escala `[0.0, 1.0]`.
2. **Capa de Resizing:** Redimensión al vuelo en GPU a $224 \times 224 \times 3$.
3. **Capa de Preprocesamiento:** Ajuste automático a la escala e intensidades de ImageNet.
4. **Backbone (ResNet50):** Extracción de características avanzadas de ImageNet (descongelado de manera parcial hasta `conv5_block`).
5. **Clasificador Customizado:** GlobalAveragePooling2D + Dropout + Dense (128, ReLU) + Dropout + Dense (3, Softmax).
