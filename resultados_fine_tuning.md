# Resultados del Modelo de Fine-Tuning (`face2emotion_finetune.ipynb`)

Este documento resume los resultados obtenidos aplicando transferencia de aprendizaje (Transfer Learning) y ajuste fino (Fine-Tuning) sobre una red **ResNet50** pre-entrenada con ImageNet para clasificar imágenes de rostros en 3 emociones: **Angry** (Enojado), **Happy** (Feliz) y **Sad** (Triste).

---

## 📊 Resumen de Métricas Generales (Modelo Optimizado)

| Conjunto | Pérdida (Loss) | Precisión (Accuracy) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: |
| **Validación** | **0.4565** | **92.75%** | **92.76%** |
| **Prueba (Test)** | **0.4799** | **91.61%** | **91.65%** |

> [!TIP]  
> Gracias a la incorporación de **Label Smoothing (0.1)** en las dos etapas de compilación y la adición del regularizador **Dropout(0.4)** en el head del modelo, logramos desplomar la pérdida de validación de **1.3091 a 0.4565** (una reducción de casi 3x) y la de test de **1.2414 a 0.4799**, sin perder poder de predicción.

---

## 📈 Reporte de Clasificación (Conjunto de Prueba)

El modelo de Fine-Tuning muestra un desempeño sumamente robusto y equilibrado en todas las clases, con un comportamiento muy estable.

```text
              precision    recall  f1-score   support

       Angry       0.90      0.91      0.90      1522
       Happy       0.97      0.93      0.95      1523
         Sad       0.88      0.92      0.90      1522

    accuracy                           0.92      4567
   macro avg       0.92      0.92      0.92      4567
weighted avg       0.92      0.92      0.92      4567
```

### 🔍 Análisis por Clase:
*   **Happy (Feliz):** Excelente precisión del **97%** y recall del **93%**, dando un F1-score consolidado de **0.95**.
*   **Sad (Triste):** Gran balance con un recall del **92%** y precisión del **88%** (F1-score de **0.90**). 
*   **Angry (Enojado):** Alcanzó un F1-score estable de **0.90**, mostrando gran consistencia en la detección de ira en comparación con el modelo base.

---

## 🧠 Estructura y Optimización del Pipeline

1.  **Entrada y Escalado Dinámico:** Las imágenes se cargan a $48 \times 48 \times 3$ en rango `[0.0, 1.0]` y se escalan en GPU al vuelo a $224 \times 224 \times 3$ con la capa `Resizing`, ahorrando memoria RAM crítica en Colab T4.
2.  **Regularización del Head (Dropout):** Se integró `Dropout(0.4)` después de `GlobalAveragePooling2D()` para mitigar que el clasificador memorice las características profundas de ResNet50.
3.  **Suavizado de Etiquetas (Label Smoothing):** Se configuró `label_smoothing=0.1` en la pérdida de la Etapa 1 y se mantuvo coherente en la compilación de la Etapa 2. Esto limitó la sobreconfianza en las probabilidades de salida, lo que controló directamente el comportamiento del `loss` de validación.
4.  **Bucle de Entrenamiento en 2 Etapas:**
    *   **Etapa 1:** Entrenamiento exclusivo de la cabeza densa con backbone congelado (15 épocas).
    *   **Etapa 2:** Descongelamiento de `conv5_block` con tasa de aprendizaje baja ($5 \times 10^{-5}$). La Etapa 3 fue descartada por redundancia en el ajuste de parámetros.
