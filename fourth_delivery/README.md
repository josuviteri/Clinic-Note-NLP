Objetivos de la cuarta entrega

Elegir, aplicar, evaluar y comparar modelos basados en transformers, procedentes de la plataforma hugging face, a nuestras tareas de clasificación y resumen de textos clínicos, basandonos en nuestas limitaciones de hardware (6/12 GB VRAM).

Modelo de clasificación basdado en transformers:
distilbert-base-uncased (DistilBERT)

Modelo de resumen basado en transformers:

t5-small (text-to-text)

Estrategia:

Volver a utilizar el preprocessing de la tercera entrega para poder comparar los modelos de forma justa

Para ambas tareas:

Comprobar la diferencia entre el full fine-tuning y el fine-tuning parcial con pesos congelados de los modelos basados en transformers para cada una de las dos tareas.

Metodos de evaluación: 
Clasificación: accuracy, precision/recall, F1 score y confusion matrix. 

Resumen: ROUGE y BERTScore. Además de comparar con los baselines de la anterior entrega TF-IDF + TextRank, resultados del LSTM results y Clinical BERT.