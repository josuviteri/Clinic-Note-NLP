3rd Delivery - README

CONCLUSIONES

Este documento ranna la reflexión de los aspectos que hemos tratado una vez terminado el proceso de experimentación, no entra en profundidad en cada tema, simplemente explica los temas tratados en cada documento.
El documento "explicación_entrega_3.ipynb", es una declaración de objetivos previos al inicio a abordar el proyecto.

Esta entrega ("third_delivery") cubre varios aspectos de nuetro proyecto:
Correción de la entrega 2
Clasificación (Shallow ML + CNN)
Resumen (Shallow ML + CNN)

CORRECCIÓN ENTREGA 2:
En primer lugar, hemos modificado la entrega anterior para que funcione acorde a 
las sugerencias recibidas. 
Hemos modificado la forma en la que creamos los embeddings contextuales y no contextuales.
En general, hemos entendido que el hecho de que tengamos la cantidad de tokens únicos y tokens totales que tenemos, no es un problema como habíamos pensado en un principio. Y aunque nos hayamos obsesionado un poco con este aspecto, nos ha ayudado a entender que no es un defecto de nuestra tokenización, sino un aspecto característico de nuestro dataset.

CONTENIDO ESPECÍFICO DE LA ENTREGA 3:

"./classification/"

Respecto a la primera tarea a abordar, entre todas las propuestas para nuestro dataset, hemos optado por la clasificación de las conversaciónes en distintas clases definidas previamente en el dataset.

"./classification/classif_shallow_ml.ipynb"
Primero, hemos usado TF-IDF y Count Vectorizer con distintas técnicas de clasificación pertenecientes al machine learning clásico y las hemos comparado.
El objetivo era el de clasificar una conversación entre un paciente y su doctor (además de otras posibles personas que pudiese haber en la conversación) en una clase.

"./classification/classif_cnn.ipynb"
Por otro lado, hemos fine-tuneado la classification head de clinical-bert con el objetivo de aplicar técnicas de redes neuronales convolucionales a nuestra tarea de clasificación.
El modelo no está en el repositorio ya que es demasiado voluminoso, pero hemos adjuntado un ejemplo del funcionamiento del modelo en inferencia con la primera conversación del dataset de validación.


"./summarisation/"

Respecto a la segunda tarea, hemos propuesto seguir con nuestro objetivo inicial, el resumen de las conversaciónes.

"./summarisation/sum_shallow_ml.ipynb"
En este aspecto también, hemos aplicado técnicas de machine learning clásico. TF-IDF, Text Rank y evaluación y comparación haciendo uso de métricas ROUGE.

"./summarisation/sum_cnn.ipynb"
La implementación de arquitecturas pertenecientes al deep learning se ha aplicado haciendo uso de redes neuronales convolucionales de arquitectura LSTM + encoder-decoder, experimentando con los embeddings congelados y fine-tuneados, con el objetivo de descubrir y entender que es lo que mejor funciona.


En todas las propuestas hemos aplicado la evaluación con distintas métricas de accuracy y las comparamos, ya que no todas las técnicas funcionan igual para cada tarea, y aunque usar el modelo en inferencia da una visión real de los resultados, los números también son relevantes para una comparación objetiva.

Próximos pasos:
Aplicaremos técnicas basadas en transformers y herramientas en las que estamos especialmente interesados (ya que el profesor indicó que para la última entrega podriamos experimentar y aprovechar para aprender y explorar aspectos en los que estamos especialmente interesados en profundizar).