# Face2Emotion (Clasificación de Emociones Faciales)

Este proyecto fue desarrollado en el marco de la materia de AI y Machine Learning para la clasificación de expresiones faciales en tres emociones básicas: Angry (Enojado), Happy (Feliz) y Sad (Triste).

Se experimentó con dos enfoques arquitectónicos: una red convolucional (CNN) entrenada desde cero como modelo base, y un modelo optimizado mediante transferencia de aprendizaje y ajuste fino (Fine-Tuning) sobre una red ResNet50 pre-entrenada.

---

## Enlaces de Descarga y Modelo Final

El modelo final entrenado con los mejores pesos se encuentra disponible para su descarga en la sección de Releases:

*   Modelo Final: [face2emotion_resnet50_final.keras](https://github.com/Angel-crypt/face2emotion-AIML/releases/download/v1.0/face2emotion_resnet50_final.keras) (207.73 MB)
*   Release Oficial: [Release v1.0](https://github.com/Angel-crypt/face2emotion-AIML/releases/tag/v1.0)
*   SHA-256 Checksum: `3a52d9a348cf91e79e03dedf137bebafdcc547acb3afc5d9f4b7b0df6666e2d5`

---

## Comparación de Resultados

A continuación se presenta un resumen comparativo del desempeño de ambos modelos en el conjunto de prueba (Test Set):

| Métrica | Modelo Base CNN (Desde Cero) | Modelo Fine-Tuning (ResNet50) | Impacto / Diferencia |
| :--- | :---: | :---: | :---: |
| **Exactitud (Accuracy)** | 86.93% | **91.61%** | **+4.68%** (Mejora significativa) |
| **F1-Score (Macro)** | 86.99% | **91.65%** | **+4.66%** (Clases más estables) |
| **Pérdida (Loss)** | **0.4086** | 0.4799 | Pérdida calibrada mediante regularización |

Para ver análisis detallados de cada modelo, puedes revisar los siguientes informes:
*   [Resultados del Modelo Base](file:///d:/PERSONAL/UNI/AI&ML/face2emotion-AIML/resultados_base_model.md)
*   [Resultados del Modelo Fine-Tuning](file:///d:/PERSONAL/UNI/AI&ML/face2emotion-AIML/resultados_fine_tuning.md)

---

## Estructura del Repositorio

El proyecto se divide en tres fases secuenciales:

1.  **[face2emotion_AIML-1st.ipynb](file:///d:/PERSONAL/UNI/AI&ML/face2emotion-AIML/face2emotion_AIML-1st.ipynb) (Fase 1: EDA y Preprocesamiento):**
    *   Análisis Exploratorio de Datos (EDA) del dataset original.
    *   Detección y corrección de desbalances de clases mediante submuestreo de la clase mayoritaria.
    *   Estandarización del formato de imágenes (48x48 píxeles en escala de grises).
2.  **[face2emotion_baseModel.ipynb](file:///d:/PERSONAL/UNI/AI&ML/face2emotion-AIML/face2emotion_baseModel.ipynb) (Fase 2: Modelo CNN desde Cero):**
    *   Entrenamiento de un modelo base convolucional propio de 3 bloques Conv2D + MaxPooling2D.
    *   Aplicación de técnicas de Data Augmentation directas.
    *   Métricas sólidas y matriz de confusión base.
3.  **[face2emotion_finetune.ipynb](file:///d:/PERSONAL/UNI/AI&ML/face2emotion-AIML/face2emotion_finetune.ipynb) (Fase 3: Fine-Tuning y Optimización):**
    *   Uso del backbone de ResNet50 pre-entrenado en ImageNet.
    *   Implementación de escalamiento dinámico ("on-the-fly") de 48x48x3 a 224x224x3 en GPU para evitar problemas de memoria (OOM) en Colab T4.
    *   Entrenamiento en 2 etapas: (1) Clasificador congelado, (2) Descongelamiento de conv5_block con LR bajo (5e-5).
    *   Optimización de la función de pérdida mediante Label Smoothing (0.1) y adición de Dropout (0.4) intermedio para controlar el sobreajuste y estabilizar el val_loss.

---

## Inicio Rápido (Quick Path)

Para ejecutar este proyecto en tu entorno local o en Google Colab:

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/Angel-crypt/face2emotion-AIML.git
    ```
2.  **Descargar el Dataset**:
    *   Descarga el dataset [samithsachidanandan/human-face-emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions) de Kaggle.
    *   Los notebooks implementan la descarga automática utilizando kagglehub, por lo que puedes ejecutarlos directamente si tienes tus credenciales configuradas.
3.  **Ejecutar los Notebooks**:
    *   Ejecútalos en orden secuencial para comprender todo el flujo de trabajo (EDA -> Modelo Base -> Fine-Tuning).
