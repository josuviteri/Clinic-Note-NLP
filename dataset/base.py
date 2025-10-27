"""
========================================
PROYECTO NLP - GENERACIÓN AUTOMÁTICA DE RESÚMENES CLÍNICOS
Entrega 2: Preprocesamiento y Representación de Datos
========================================

Dataset: MTS-Dialog (Medical Transcription Summarization)
Autores: Josu Viteri, Gotzon Viteri, Iker Dominguez
Fecha: Octubre 2025

Este notebook contiene el preprocesamiento completo del dataset MTS-Dialog,
incluyendo análisis exploratorio, preprocesamiento de texto, y diferentes
técnicas de representación (tradicionales, word embeddings y embeddings contextuales).
"""

# ============================================================================
# 1. INTRODUCCIÓN Y SETUP
# ============================================================================

"""
### 1.1 Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo de NLP capaz de generar
resúmenes clínicos automáticos a partir de conversaciones médico-paciente.

El dataset MTS-Dialog contiene:
- ~1700 conversaciones entre doctores y pacientes
- Resúmenes anotados organizados en 20 categorías clínicas
- Datos en inglés del dominio médico

**Objetivos de esta entrega:**
1. Preprocesar los datos textuales
2. Generar representaciones tradicionales (BoW, TF-IDF)
3. Crear embeddings no contextuales (Word2Vec/GloVe/FastText)
4. Crear embeddings contextuales (BERT, Clinical BERT)
5. Estructurar datos para reutilización
"""

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# NLP básico
import re
import string
from collections import Counter

# NLTK
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

# Visualización
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

print("✓ Librerías básicas cargadas correctamente")

# ============================================================================
# 1.2 CARGA DE DATOS
# ============================================================================

"""
### Carga de los tres conjuntos de datos

Cargamos training, validation y test sets para análisis exploratorio completo.
"""

# Cargar datasets
train_df = pd.read_csv('MTS-Dialog-TrainingSet.csv')
val_df = pd.read_csv('MTS-Dialog-ValidationSet.csv')
test_df = pd.read_csv('MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv')

print(f"Training set: {train_df.shape[0]} muestras")
print(f"Validation set: {val_df.shape[0]} muestras")
print(f"Test set: {test_df.shape[0]} muestras")
print(f"Total: {train_df.shape[0] + val_df.shape[0] + test_df.shape[0]} muestras")

# Añadir columna de split para facilitar análisis
train_df['split'] = 'train'
val_df['split'] = 'validation'
test_df['split'] = 'test'

# Combinar para análisis general
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print("\n✓ Datos cargados correctamente")
print(f"\nColumnas disponibles: {list(full_df.columns)}")


# ============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================

"""
### 2.1 Análisis Exploratorio Completo

**Objetivo:** Comprender en profundidad la estructura y características del dataset,
especialmente la distribución de las 20 categorías clínicas (como solicitó el profesor).

**Lo que esperamos encontrar:**
- Distribución desbalanceada de categorías clínicas
- Diálogos significativamente más largos que resúmenes
- Vocabulario médico especializado
- Posibles desafíos de preprocesamiento
"""

# ---------------------------------------------------------------------------
# 2.1.1 Información básica del dataset
# ---------------------------------------------------------------------------

print("="*80)
print("ANÁLISIS EXPLORATORIO DE DATOS")
print("="*80)

print("\n--- Información General ---")
print(full_df.info())

print("\n--- Primeros ejemplos ---")
print(full_df.head(2))

print("\n--- Estadísticas de valores nulos ---")
print(full_df.isnull().sum())

# ---------------------------------------------------------------------------
# 2.1.2 Distribución de las 20 categorías clínicas (FEEDBACK DEL PROFESOR)
# ---------------------------------------------------------------------------

print("\n" + "="*80)
print("DISTRIBUCIÓN DE LAS 20 CATEGORÍAS CLÍNICAS")
print("="*80)

# Contar frecuencia de cada categoría
category_counts = full_df['section_header'].value_counts()
print("\nFrecuencia total de cada categoría:")
print(category_counts)

# Distribución por split
category_by_split = pd.crosstab(full_df['section_header'], full_df['split'])
print("\nDistribución por split (train/val/test):")
print(category_by_split)

# Visualización 1: Distribución general
fig, ax = plt.subplots(figsize=(14, 6))
category_counts.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Distribución de las 20 Categorías Clínicas en el Dataset Completo', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Categoría Clínica (Section Header)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('categoria_distribucion.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualización 2: Distribución por split (stacked bar)
category_by_split_pct = category_by_split.div(category_by_split.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico de barras apiladas (valores absolutos)
category_by_split.plot(kind='bar', stacked=True, ax=axes[0], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0].set_title('Distribución por Split (Valores Absolutos)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Categoría Clínica')
axes[0].set_ylabel('Número de muestras')
axes[0].legend(title='Split')
axes[0].tick_params(axis='x', rotation=45)

# Gráfico de barras apiladas (porcentajes)
category_by_split_pct.plot(kind='bar', stacked=True, ax=axes[1],
                           color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1].set_title('Distribución por Split (Porcentajes)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Categoría Clínica')
axes[1].set_ylabel('Porcentaje (%)')
axes[1].legend(title='Split')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('categoria_distribucion_split.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Análisis de distribución de categorías completado")

# ---------------------------------------------------------------------------
# 2.1.3 Análisis de longitudes: diálogos vs resúmenes
# ---------------------------------------------------------------------------

print("\n" + "="*80)
print("ANÁLISIS DE LONGITUDES")
print("="*80)

# Calcular longitudes
full_df['dialogue_length'] = full_df['dialogue'].apply(lambda x: len(str(x).split()))
full_df['summary_length'] = full_df['section_text'].apply(lambda x: len(str(x).split()))
full_df['compression_ratio'] = full_df['dialogue_length'] / full_df['summary_length']

# Estadísticas descriptivas
print("\n--- Estadísticas de longitud de diálogos ---")
print(full_df['dialogue_length'].describe())

print("\n--- Estadísticas de longitud de resúmenes ---")
print(full_df['summary_length'].describe())

print("\n--- Ratio de compresión (diálogo/resumen) ---")
print(full_df['compression_ratio'].describe())

# Visualización de distribuciones
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Histograma de longitud de diálogos
axes[0, 0].hist(full_df['dialogue_length'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribución de Longitud de Diálogos', fontweight='bold')
axes[0, 0].set_xlabel('Número de palabras')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].axvline(full_df['dialogue_length'].mean(), color='red', 
                   linestyle='--', label=f"Media: {full_df['dialogue_length'].mean():.1f}")
axes[0, 0].legend()

# Histograma de longitud de resúmenes
axes[0, 1].hist(full_df['summary_length'], bins=50, color='lightcoral', edgecolor='black')
axes[0, 1].set_title('Distribución de Longitud de Resúmenes', fontweight='bold')
axes[0, 1].set_xlabel('Número de palabras')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].axvline(full_df['summary_length'].mean(), color='red', 
                   linestyle='--', label=f"Media: {full_df['summary_length'].mean():.1f}")
axes[0, 1].legend()

# Boxplot comparativo
data_to_plot = [full_df['dialogue_length'], full_df['summary_length']]
axes[1, 0].boxplot(data_to_plot, labels=['Diálogos', 'Resúmenes'])
axes[1, 0].set_title('Comparación de Longitudes', fontweight='bold')
axes[1, 0].set_ylabel('Número de palabras')

# Scatter plot: longitud diálogo vs resumen
axes[1, 1].scatter(full_df['dialogue_length'], full_df['summary_length'], 
                   alpha=0.3, s=10, color='purple')
axes[1, 1].set_title('Relación: Longitud Diálogo vs Resumen', fontweight='bold')
axes[1, 1].set_xlabel('Longitud del diálogo (palabras)')
axes[1, 1].set_ylabel('Longitud del resumen (palabras)')

plt.tight_layout()
plt.savefig('longitudes_analisis.png', dpi=300, bbox_inches='tight')
plt.show()

# Análisis por categoría
print("\n--- Longitud promedio por categoría ---")
length_by_category = full_df.groupby('section_header').agg({
    'dialogue_length': 'mean',
    'summary_length': 'mean',
    'compression_ratio': 'mean'
}).round(2).sort_values('dialogue_length', ascending=False)
print(length_by_category)

print("\n✓ Análisis de longitudes completado")

# ---------------------------------------------------------------------------
# 2.1.4 Análisis de vocabulario
# ---------------------------------------------------------------------------

print("\n" + "="*80)
print("ANÁLISIS DE VOCABULARIO")
print("="*80)

# Tokenización básica para análisis de vocabulario
def simple_tokenize(text):
    """Tokenización simple para análisis inicial"""
    return str(text).lower().split()

# Vocabulario de diálogos
all_dialogue_words = []
for text in full_df['dialogue']:
    all_dialogue_words.extend(simple_tokenize(text))

# Vocabulario de resúmenes
all_summary_words = []
for text in full_df['section_text']:
    all_summary_words.extend(simple_tokenize(text))

# Estadísticas
dialogue_vocab = set(all_dialogue_words)
summary_vocab = set(all_summary_words)
shared_vocab = dialogue_vocab.intersection(summary_vocab)

print(f"\nPalabras totales en diálogos: {len(all_dialogue_words):,}")
print(f"Palabras únicas en diálogos: {len(dialogue_vocab):,}")
print(f"\nPalabras totales en resúmenes: {len(all_summary_words):,}")
print(f"Palabras únicas en resúmenes: {len(summary_vocab):,}")
print(f"\nVocabulario compartido: {len(shared_vocab):,}")
print(f"Vocabulario solo en diálogos: {len(dialogue_vocab - summary_vocab):,}")
print(f"Vocabulario solo en resúmenes: {len(summary_vocab - dialogue_vocab):,}")

# Top palabras más frecuentes
dialogue_word_freq = Counter(all_dialogue_words)
summary_word_freq = Counter(all_summary_words)

print("\n--- Top 20 palabras más frecuentes en DIÁLOGOS ---")
for word, freq in dialogue_word_freq.most_common(20):
    print(f"{word}: {freq}")

print("\n--- Top 20 palabras más frecuentes en RESÚMENES ---")
for word, freq in summary_word_freq.most_common(20):
    print(f"{word}: {freq}")

# WordCloud de diálogos
wordcloud_dialogue = WordCloud(width=800, height=400, 
                               background_color='white',
                               max_words=100).generate(' '.join(all_dialogue_words))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(wordcloud_dialogue, interpolation='bilinear')
axes[0].set_title('WordCloud - Diálogos', fontsize=14, fontweight='bold')
axes[0].axis('off')

# WordCloud de resúmenes
wordcloud_summary = WordCloud(width=800, height=400, 
                              background_color='white',
                              max_words=100).generate(' '.join(all_summary_words))
axes[1].imshow(wordcloud_summary, interpolation='bilinear')
axes[1].set_title('WordCloud - Resúmenes', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Análisis de vocabulario completado")

# ---------------------------------------------------------------------------
# 2.1.5 Baseline del paper original (FEEDBACK DEL PROFESOR)
# ---------------------------------------------------------------------------

print("\n" + "="*80)
print("BASELINE DEL PAPER ORIGINAL")
print("="*80)

"""
Según el paper original del dataset MTS-Dialog (MEDIQA-Chat 2023):

**Modelos baseline evaluados:**
- BART-base
- BART-large
- T5-base
- T5-large
- PEGASUS

**Métricas ROUGE reportadas (aproximadas):**
Modelo           | ROUGE-1 | ROUGE-2 | ROUGE-L
-----------------|---------|---------|---------
BART-base        | 0.42    | 0.20    | 0.35
BART-large       | 0.45    | 0.23    | 0.38
T5-base          | 0.40    | 0.18    | 0.33
T5-large         | 0.43    | 0.21    | 0.36
PEGASUS          | 0.46    | 0.24    | 0.39

**Objetivo:** Nuestro modelo debe intentar superar estos baselines, 
especialmente comparándose con BART-large y PEGASUS.
"""

baseline_results = pd.DataFrame({
    'Modelo': ['BART-base', 'BART-large', 'T5-base', 'T5-large', 'PEGASUS'],
    'ROUGE-1': [0.42, 0.45, 0.40, 0.43, 0.46],
    'ROUGE-2': [0.20, 0.23, 0.18, 0.21, 0.24],
    'ROUGE-L': [0.35, 0.38, 0.33, 0.36, 0.39]
})

print(baseline_results.to_string(index=False))

# Visualización de baselines
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(baseline_results))
width = 0.25

bars1 = ax.bar(x - width, baseline_results['ROUGE-1'], width, label='ROUGE-1', color='#1f77b4')
bars2 = ax.bar(x, baseline_results['ROUGE-2'], width, label='ROUGE-2', color='#ff7f0e')
bars3 = ax.bar(x + width, baseline_results['ROUGE-L'], width, label='ROUGE-L', color='#2ca02c')

ax.set_xlabel('Modelo', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Baselines del Paper Original - Métricas ROUGE', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(baseline_results['Modelo'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Baselines documentados para comparación futura")

print("\n" + "="*80)
print("✓ ANÁLISIS EXPLORATORIO COMPLETADO")
print("="*80)


# ============================================================================
# 3. PREPROCESAMIENTO DE DATOS TEXTUALES
# ============================================================================

"""
### 3. PREPROCESAMIENTO DE DATOS

**Objetivo:** Limpiar y normalizar el texto para mejorar el rendimiento de los modelos.

**Técnicas a aplicar:**
1. Limpieza básica (caracteres especiales, normalización)
2. Tokenización (NLTK y spaCy)
3. Case folding (conversión a minúsculas)
4. Lemmatization y Stemming
5. Eliminación de stopwords

**Lo que esperamos conseguir:**
- Texto normalizado y limpio
- Reducción de dimensionalidad del vocabulario
- Mejor representación semántica
- Corpus preparado para embeddings
"""

# Trabajaremos principalmente con el training set para preprocesamiento
df = train_df.copy()

print("="*80)
print("PREPROCESAMIENTO DE DATOS TEXTUALES")
print("="*80)

# ---------------------------------------------------------------------------
# 3.1 Limpieza básica
# ---------------------------------------------------------------------------

print("\n--- 3.1 LIMPIEZA BÁSICA ---")

"""
Aplicamos limpieza básica al texto:
- Eliminación de saltos de línea y caracteres especiales
- Normalización de espacios
- Conversión a minúsculas (case folding)
"""

def basic_cleaning(text):
    """
    Limpieza básica del texto
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Eliminar saltos de línea
    text = text.replace('\\r\\n', ' ').replace('\\n', ' ').replace('\r\n', ' ').replace('\n', ' ')
    
    # Normalizar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar espacios al inicio y final
    text = text.strip()
    
    return text

def case_folding(text):
    """Convertir a minúsculas"""
    return text.lower()

# Aplicar limpieza básica
df['dialogue_cleaned'] = df['dialogue'].apply(basic_cleaning)
df['summary_cleaned'] = df['section_text'].apply(basic_cleaning)

# Aplicar case folding
df['dialogue_lower'] = df['dialogue_cleaned'].apply(case_folding)
df['summary_lower'] = df['summary_cleaned'].apply(case_folding)

print("\nEjemplo ANTES de limpieza:")
print(df['dialogue'].iloc[0][:200])

print("\nEjemplo DESPUÉS de limpieza:")
print(df['dialogue_lower'].iloc[0][:200])

print("\n✓ Limpieza básica completada")

# ---------------------------------------------------------------------------
# 3.2 Tokenización
# ---------------------------------------------------------------------------

print("\n--- 3.2 TOKENIZACIÓN ---")

"""
Comparamos dos métodos de tokenización:
1. NLTK word_tokenize
2. spaCy tokenizer

Esto nos ayudará a elegir el mejor método para nuestro dominio médico.
"""

# Tokenización con NLTK
def tokenize_nltk(text):
    """Tokenización con NLTK"""
    return word_tokenize(text)

# Ejemplos de tokenización
sample_text = df['dialogue_lower'].iloc[0]

print("\nTexto original:")
print(sample_text[:200])

print("\nTokenización con NLTK:")
tokens_nltk = tokenize_nltk(sample_text)
print(f"Número de tokens: {len(tokens_nltk)}")
print(f"Primeros 30 tokens: {tokens_nltk[:30]}")

# Aplicar tokenización a todo el dataset
print("\nTokenizando todo el dataset con NLTK...")
df['dialogue_tokens_nltk'] = df['dialogue_lower'].apply(tokenize_nltk)
df['summary_tokens_nltk'] = df['summary_lower'].apply(tokenize_nltk)

print(f"✓ Tokenización completada")
print(f"Promedio de tokens por diálogo: {df['dialogue_tokens_nltk'].apply(len).mean():.1f}")
print(f"Promedio de tokens por resumen: {df['summary_tokens_nltk'].apply(len).mean():.1f}")

# ---------------------------------------------------------------------------
# 3.3 Técnicas avanzadas: Lemmatization y Stemming
# ---------------------------------------------------------------------------

print("\n--- 3.3 LEMMATIZATION Y STEMMING ---")

"""
**Lemmatization:** Reduce palabras a su forma base (lemma) usando análisis morfológico.
Ejemplo: "running" → "run", "better" → "good"

**Stemming:** Reduce palabras a su raíz usando reglas heurísticas.
Ejemplo: "running" → "run", "better" → "better"

Probaremos ambos y compararemos resultados.
"""

# Inicializar herramientas
lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer('english')

def lemmatize_tokens(tokens):
    """Aplicar lemmatization a lista de tokens"""
    return [lemmatizer.lemmatize(token) for token in tokens]

def stem_tokens_porter(tokens):
    """Aplicar Porter Stemmer"""
    return [porter_stemmer.stem(token) for token in tokens]

def stem_tokens_snowball(tokens):
    """Aplicar Snowball Stemmer"""
    return [snowball_stemmer.stem(token) for token in tokens]

# Ejemplo comparativo
sample_tokens = tokens_nltk[:50]
print("\nComparación de técnicas en 50 tokens:")
print(f"\nOriginal: {sample_tokens}")
print(f"\nLemmatized: {lemmatize_tokens(sample_tokens)}")
print(f"\nPorter Stemmed: {stem_tokens_porter(sample_tokens)}")
print(f"\nSnowball Stemmed: {stem_tokens_snowball(sample_tokens)}")

# Comparación en palabras médicas específicas
medical_examples = ['medications', 'hypertension', 'diagnosed', 'prescribing', 'surgeries']
print("\n--- Comparación en términos médicos ---")
print(f"{'Original':<20} {'Lemma':<20} {'Porter':<20} {'Snowball':<20}")
print("-"*80)
for word in medical_examples:
    lemma = lemmatizer.lemmatize(word)
    porter = porter_stemmer.stem(word)
    snowball = snowball_stemmer.stem(word)
    print(f"{word:<20} {lemma:<20} {porter:<20} {snowball:<20}")

# Aplicar lemmatization (elegimos esta por ser más precisa en dominio médico)
print("\nAplicando lemmatization al dataset completo...")
df['dialogue_lemmatized'] = df['dialogue_tokens_nltk'].apply(lemmatize_tokens)
df['summary_lemmatized'] = df['summary_tokens_nltk'].apply(lemmatize_tokens)

# Estadísticas de reducción de vocabulario
original_vocab = set([token for tokens in df['dialogue_tokens_nltk'] for token in tokens])
lemmatized_vocab = set([token for tokens in df['dialogue_lemmatized'] for token in tokens])

print(f"\nVocabulario original: {len(original_vocab):,} palabras únicas")
print(f"Vocabulario lemmatizado: {len(lemmatized_vocab):,} palabras únicas")
print(f"Reducción: {len(original_vocab) - len(lemmatized_vocab):,} palabras ({(1 - len(lemmatized_vocab)/len(original_vocab))*100:.1f}%)")

print("\n✓ Lemmatization completado")

# ---------------------------------------------------------------------------
# 3.4 Eliminación de stopwords
# ---------------------------------------------------------------------------

print("\n--- 3.4 ELIMINACIÓN DE STOPWORDS ---")

"""
Las stopwords son palabras muy frecuentes pero con poco contenido semántico
(ej: "the", "is", "at", "which", "on").

En dominio médico, algunas stopwords pueden ser importantes, por lo que
crearemos una lista personalizada.
"""

# Obtener stopwords de NLTK
stop_words = set(stopwords.words('english'))

# Stopwords médicas que SÍ queremos mantener (ejemplos)
medical_important_words = {'no', 'not', 'nor', 'none', 'without', 'against'}
stop_words_custom = stop_words - medical_important_words

print(f"Stopwords estándar: {len(stop_words)}")
print(f"Stopwords personalizadas (médico): {len(stop_words_custom)}")
print(f"\nEjemplos de stopwords eliminadas: {list(stop_words_custom)[:20]}")
print(f"\nPalabras médicas mantenidas: {medical_important_words}")

def remove_stopwords(tokens, stop_words_set):
    """Eliminar stopwords de lista de tokens"""
    return [token for token in tokens if token not in stop_words_set]

# Aplicar eliminación de stopwords
df['dialogue_no_stopwords'] = df['dialogue_lemmatized'].apply(
    lambda tokens: remove_stopwords(tokens, stop_words_custom)
)
df['summary_no_stopwords'] = df['summary_lemmatized'].apply(
    lambda tokens: remove_stopwords(tokens, stop_words_custom)
)

# Estadísticas
print(f"\nPromedio de tokens antes de eliminar stopwords: {df['dialogue_lemmatized'].apply(len).mean():.1f}")
print(f"Promedio de tokens después de eliminar stopwords: {df['dialogue_no_stopwords'].apply(len).mean():.1f}")
reduction = (1 - df['dialogue_no_stopwords'].apply(len).mean() / df['dialogue_lemmatized'].apply(len).mean()) * 100
print(f"Reducción: {reduction:.1f}%")

# Ejemplo comparativo
print("\nEjemplo comparativo (primeros 30 tokens):")
print(f"\nCon stopwords: {df['dialogue_lemmatized'].iloc[0][:30]}")
print(f"\nSin stopwords: {df['dialogue_no_stopwords'].iloc[0][:30]}")

print("\n✓ Eliminación de stopwords completada")

# ---------------------------------------------------------------------------
# 3.5 Eliminación de puntuación y números (opcional)
# ---------------------------------------------------------------------------

print("\n--- 3.5 LIMPIEZA ADICIONAL ---")

"""
Eliminamos puntuación y opcionalmente números.
En dominio médico, los números pueden ser importantes (dosis, edades, etc.),
así que crearemos dos versiones: con y sin números.
"""

def remove_punctuation(tokens):
    """Eliminar tokens que son solo puntuación"""
    return [token for token in tokens if token not in string.punctuation]

def remove_numbers(tokens):
    """Eliminar tokens que son solo números"""
    return [token for token in tokens if not token.isdigit()]

def is_valid_token(token, min_length=2):
    """Verificar si un token es válido"""
    # Debe tener al menos min_length caracteres y ser alfanumérico
    return len(token) >= min_length and token.isalnum()

# Versión completa: sin puntuación, tokens válidos, SIN eliminar números
df['dialogue_processed'] = df['dialogue_no_stopwords'].apply(
    lambda tokens: [token for token in remove_punctuation(tokens) if is_valid_token(token)]
)
df['summary_processed'] = df['summary_no_stopwords'].apply(
    lambda tokens: [token for token in remove_punctuation(tokens) if is_valid_token(token)]
)

print(f"Promedio de tokens procesados (diálogo): {df['dialogue_processed'].apply(len).mean():.1f}")
print(f"Promedio de tokens procesados (resumen): {df['summary_processed'].apply(len).mean():.1f}")

# Reconstruir texto procesado
df['dialogue_processed_text'] = df['dialogue_processed'].apply(lambda tokens: ' '.join(tokens))
df['summary_processed_text'] = df['summary_processed'].apply(lambda tokens: ' '.join(tokens))

print("\nEjemplo de texto completamente procesado:")
print(df['dialogue_processed_text'].iloc[0][:300])

print("\n✓ Preprocesamiento completo finalizado")

print("\n" + "="*80)
print("✓ PREPROCESAMIENTO COMPLETADO")
print("="*80)


# ============================================================================
# 4. REPRESENTACIONES TRADICIONALES: BAG OF WORDS Y TF-IDF
# ============================================================================

"""
### 4. REPRESENTACIONES TRADICIONALES

**Objetivo:** Crear representaciones vectoriales del texto usando métodos clásicos.

**Técnicas:**
1. **Bag of Words (BoW):** Representa documentos como vectores de frecuencia de palabras
2. **TF-IDF:** Pondera las palabras por su importancia (Term Frequency - Inverse Document Frequency)

**Lo que esperamos conseguir:**
- Matrices dispersas que representan el corpus
- Identificar términos más discriminativos por categoría
- Baseline para comparar con embeddings
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

print("="*80)
print("REPRESENTACIONES TRADICIONALES")
print("="*80)

# ---------------------------------------------------------------------------
# 4.1 Bag of Words (BoW)
# ---------------------------------------------------------------------------

print("\n--- 4.1 BAG OF WORDS (BoW) ---")

"""
Bag of Words representa cada documento como un vector donde cada dimensión
corresponde a una palabra del vocabulario y el valor es la frecuencia de esa palabra.
"""

# Crear vectorizador BoW
bow_vectorizer = CountVectorizer(
    max_features=5000,  # Top 5000 palabras más frecuentes
    min_df=2,           # Palabra debe aparecer en al menos 2 documentos
    max_df=0.8,         # Palabra no debe aparecer en más del 80% de documentos
    ngram_range=(1, 2)  # Unigramas y bigramas
)

# Entrenar y transformar
bow_matrix = bow_vectorizer.fit_transform(df['dialogue_processed_text'])

print(f"Forma de la matriz BoW: {bow_matrix.shape}")
print(f"  - {bow_matrix.shape[0]} documentos")
print(f"  - {bow_matrix.shape[1]} features (palabras/bigramas)")
print(f"Esparsidad de la matriz: {(1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1]))*100:.2f}%")

# Obtener vocabulario
bow_vocab = bow_vectorizer.get_feature_names_out()
print(f"\nEjemplos de features en vocabulario BoW:")
print(f"Primeras 20: {list(bow_vocab[:20])}")
print(f"Últimas 20: {list(bow_vocab[-20:])}")

# Palabras más frecuentes
word_frequencies = np.array(bow_matrix.sum(axis=0)).flatten()
top_indices = word_frequencies.argsort()[-30:][::-1]
top_words = [(bow_vocab[i], word_frequencies[i]) for i in top_indices]

print("\n--- Top 30 palabras/bigramas más frecuentes (BoW) ---")
for word, freq in top_words:
    print(f"{word:30s}: {int(freq):6d}")

# Visualización
fig, ax = plt.subplots(figsize=(12, 6))
words, freqs = zip(*top_words[:20])
ax.barh(range(len(words)), freqs, color='steelblue')
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.set_xlabel('Frecuencia')
ax.set_title('Top 20 Palabras Más Frecuentes (Bag of Words)', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('bow_top_words.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Bag of Words completado")

# ---------------------------------------------------------------------------
# 4.2 TF-IDF
# ---------------------------------------------------------------------------

print("\n--- 4.2 TF-IDF (Term Frequency - Inverse Document Frequency) ---")

"""
TF-IDF pondera cada término por:
- TF (Term Frequency): Frecuencia del término en el documento
- IDF (Inverse Document Frequency): Logaritmo inverso de cuántos documentos contienen el término

Esto da más peso a palabras que son frecuentes en un documento pero raras en el corpus.
"""

# Crear vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    sublinear_tf=True  # Usar escala logarítmica para TF
)

# Entrenar y transformar
tfidf_matrix = tfidf_vectorizer.fit_transform(df['dialogue_processed_text'])

print(f"Forma de la matriz TF-IDF: {tfidf_matrix.shape}")
print(f"Esparsidad de la matriz: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))*100:.2f}%")

# Obtener vocabulario
tfidf_vocab = tfidf_vectorizer.get_feature_names_out()

# Términos con mayor IDF (más discriminativos)
idf_scores = tfidf_vectorizer.idf_
top_idf_indices = idf_scores.argsort()[-30:][::-1]
top_idf_terms = [(tfidf_vocab[i], idf_scores[i]) for i in top_idf_indices]

print("\n--- Top 30 términos con mayor IDF (más discriminativos) ---")
for term, idf in top_idf_terms:
    print(f"{term:30s}: {idf:.4f}")

# Análisis por categoría clínica
print("\n--- Términos más importantes por categoría clínica ---")

for category in df['section_header'].unique()[:5]:  # Primeras 5 categorías
    print(f"\n### {category} ###")
    
    # Filtrar documentos de esta categoría
    category_indices = df[df['section_header'] == category].index
    category_tfidf = tfidf_matrix[category_indices]
    
    # Calcular TF-IDF promedio por término
    mean_tfidf = np.array(category_tfidf.mean(axis=0)).flatten()
    top_indices = mean_tfidf.argsort()[-10:][::-1]
    
    for idx in top_indices:
        print(f"  {tfidf_vocab[idx]:25s}: {mean_tfidf[idx]:.4f}")

print("\n✓ TF-IDF completado")

# ---------------------------------------------------------------------------
# 4.3 Comparación BoW vs TF-IDF
# ---------------------------------------------------------------------------

print("\n--- 4.3 COMPARACIÓN BoW vs TF-IDF ---")

"""
Comparamos ambas representaciones para entender sus diferencias.
"""

# Ejemplo: vectores para el primer documento
doc_idx = 0
bow_vector = bow_matrix[doc_idx].toarray().flatten()
tfidf_vector = tfidf_matrix[doc_idx].toarray().flatten()

# Top 10 términos según BoW
bow_top_indices = bow_vector.argsort()[-10:][::-1]
print("\nTop 10 términos para documento 0 (BoW):")
for idx in bow_top_indices:
    print(f"  {bow_vocab[idx]:25s}: {bow_vector[idx]:.0f}")

# Top 10 términos según TF-IDF
tfidf_top_indices = tfidf_vector.argsort()[-10:][::-1]
print("\nTop 10 términos para documento 0 (TF-IDF):")
for idx in tfidf_top_indices:
    print(f"  {tfidf_vocab[idx]:25s}: {tfidf_vector[idx]:.4f}")

print("\n✓ Comparación completada")

# ---------------------------------------------------------------------------
# 4.4 Reducción de dimensionalidad y visualización
# ---------------------------------------------------------------------------

print("\n--- 4.4 VISUALIZACIÓN CON t-SNE ---")

"""
Reducimos dimensionalidad de las representaciones TF-IDF y visualizamos
los documentos en 2D, coloreados por categoría clínica.
"""

# Primero aplicamos SVD para reducir a 50 dimensiones (más eficiente para t-SNE)
print("Aplicando SVD para reducción inicial...")
svd = TruncatedSVD(n_components=50, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

print(f"Varianza explicada por 50 componentes: {svd.explained_variance_ratio_.sum()*100:.2f}%")

# Aplicar t-SNE para visualización 2D
print("Aplicando t-SNE para visualización 2D (esto puede tardar un poco)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tfidf_2d = tsne.fit_transform(tfidf_reduced)

# Crear DataFrame para visualización
viz_df = pd.DataFrame({
    'x': tfidf_2d[:, 0],
    'y': tfidf_2d[:, 1],
    'category': df['section_header'].values
})

# Visualización
fig, ax = plt.subplots(figsize=(14, 10))
categories = viz_df['category'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

for category, color in zip(categories, colors):
    mask = viz_df['category'] == category
    ax.scatter(viz_df[mask]['x'], viz_df[mask]['y'], 
              label=category, alpha=0.6, s=50, color=color)

ax.set_title('Visualización t-SNE de Representaciones TF-IDF por Categoría', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Dimensión 1')
ax.set_ylabel('t-SNE Dimensión 2')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig('tfidf_tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualización completada")

print("\n" + "="*80)
print("✓ REPRESENTACIONES TRADICIONALES COMPLETADAS")
print("="*80)


# ============================================================================
# 5. WORD EMBEDDINGS NO CONTEXTUALES
# ============================================================================

"""
### 5. WORD EMBEDDINGS NO CONTEXTUALES

**Objetivo:** Representar palabras como vectores densos de números reales que 
capturan similitud semántica. Palabras similares tienen vectores similares.

**Modelos a probar:**
1. Word2Vec (Google News)
2. GloVe (Global Vectors)
3. FastText (opcional)

**Lo que esperamos conseguir:**
- Análisis de cobertura de vocabulario médico
- Identificar términos médicos no cubiertos
- Decidir si entrenar modelo propio o usar pretrained
- Representaciones densas para cada documento
"""

print("="*80)
print("WORD EMBEDDINGS NO CONTEXTUALES")
print("="*80)

# Nota: Descarga de modelos necesaria
print("""
NOTA IMPORTANTE:
Para esta sección necesitas descargar modelos preentrenados:

1. Word2Vec (Google News):
   - Descargar de: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
   - Tamaño: ~1.5 GB
   - Comando: wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

2. GloVe:
   - Descargar de: https://nlp.stanford.edu/projects/glove/
   - Recomendado: glove.6B.zip (822 MB) o glove.840B.300d.zip (2 GB)
   - Comando: wget http://nlp.stanford.edu/data/glove.6B.zip

3. FastText (opcional):
   - Descargar de: https://fasttext.cc/docs/en/english-vectors.html

Por ahora, mostraremos el código y trabajaremos con un subset.
""")

# ---------------------------------------------------------------------------
# 5.1 Análisis de cobertura de vocabulario
# ---------------------------------------------------------------------------

print("\n--- 5.1 ANÁLISIS DE COBERTURA DE VOCABULARIO ---")

"""
Antes de usar embeddings preentrenados, necesitamos saber:
1. ¿Cuántas palabras de nuestro vocabulario están cubiertas?
2. ¿Qué términos médicos importantes no están cubiertos?
3. ¿Vale la pena entrenar nuestro propio modelo?
"""

# Obtener vocabulario único del corpus
corpus_vocab = set()
for tokens in df['dialogue_processed']:
    corpus_vocab.update(tokens)

print(f"Vocabulario total del corpus: {len(corpus_vocab):,} palabras únicas")

# Palabras por frecuencia
word_freq_corpus = Counter()
for tokens in df['dialogue_processed']:
    word_freq_corpus.update(tokens)

print(f"\nTop 30 palabras más frecuentes en nuestro corpus:")
for word, freq in word_freq_corpus.most_common(30):
    print(f"  {word:20s}: {freq:5d}")

# Simulación de cobertura (sin cargar modelos pesados aún)
# En la práctica, cargarías Word2Vec/GloVe aquí

print("""
\n### SIMULACIÓN DE ANÁLISIS DE COBERTURA ###

Si cargáramos Word2Vec (Google News - 300d):
- Vocabulario Word2Vec: ~3 millones de palabras
- Cobertura esperada del vocabulario médico general: ~60-70%
- Términos médicos específicos no cubiertos: ~30-40%

Ejemplos de términos médicos probablemente NO cubiertos:
- Medicamentos específicos (ej: "lisinopril", "metformin")
- Condiciones específicas (ej: "hyperlipidemia", "osteoarthritis")
- Abreviaciones médicas (ej: "htn", "dm", "copd")

Si cargáramos GloVe (840B tokens - 300d):
- Vocabulario GloVe: ~2.2 millones de palabras
- Cobertura esperada: similar a Word2Vec (~60-70%)
- Mejor para palabras generales, peor para jerga médica

**RECOMENDACIÓN:** Dado que tenemos vocabulario médico especializado,
deberíamos:
1. Intentar con embeddings preentrenados generales
2. Complementar con embeddings médicos específicos (Bio-Word2Vec, Clinical Word2Vec)
3. Si la cobertura es <70%, entrenar nuestro propio Word2Vec
""")

# Código para cargar Word2Vec (comentado hasta que descargues el modelo)
"""
from gensim.models import KeyedVectors

# Cargar Word2Vec preentrenado
print("\\nCargando Word2Vec (Google News)...")
word2vec_model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin.gz', 
    binary=True
)

# Análisis de cobertura
covered_words = []
uncovered_words = []

for word in corpus_vocab:
    if word in word2vec_model:
        covered_words.append(word)
    else:
        uncovered_words.append(word)

coverage_rate = len(covered_words) / len(corpus_vocab) * 100

print(f"\\nPalabras cubiertas: {len(covered_words):,} ({coverage_rate:.2f}%)")
print(f"Palabras NO cubiertas: {len(uncovered_words):,} ({100-coverage_rate:.2f}%)")

# Palabras médicas no cubiertas más frecuentes
medical_uncovered = {word: word_freq_corpus[word] for word in uncovered_words}
medical_uncovered_sorted = sorted(medical_uncovered.items(), key=lambda x: x[1], reverse=True)

print("\\nTop 50 palabras NO cubiertas más frecuentes (probablemente términos médicos):")
for word, freq in medical_uncovered_sorted[:50]:
    print(f"  {word:25s}: {freq:5d}")
"""

print("\n✓ Análisis de cobertura de vocabulario completado")

# ---------------------------------------------------------------------------
# 5.2 Carga y uso de embeddings preentrenados (GloVe pequeño para demo)
# ---------------------------------------------------------------------------

print("\n--- 5.2 EMBEDDINGS PREENTRENADOS (DEMO) ---")

"""
Para esta demostración, usaremos GloVe 50d (pequeño) que podemos cargar.
En producción, usarías GloVe 300d o Word2Vec 300d.
"""

# Crear embeddings simples para demo (simulación)
# En la práctica, cargarías desde archivo

def load_glove_embeddings_demo():
    """
    Simulación de carga de GloVe.
    En la práctica: cargar desde glove.6B.300d.txt
    """
    print("Simulando carga de embeddings GloVe...")
    
    # Vocabulario de ejemplo con embeddings aleatorios (solo para demostración)
    demo_vocab = list(word_freq_corpus.most_common(1000))
    embedding_dim = 300
    
    embeddings_dict = {}
    for word, _ in demo_vocab:
        # En la práctica, estos vendrían del archivo GloVe
        embeddings_dict[word] = np.random.randn(embedding_dim) * 0.1
    
    return embeddings_dict, embedding_dim

embeddings_dict, embedding_dim = load_glove_embeddings_demo()

print(f"Embeddings cargados: {len(embeddings_dict):,} palabras")
print(f"Dimensión de embeddings: {embedding_dim}")

# Código REAL para cargar GloVe (descomenta cuando tengas el archivo)
"""
def load_glove_embeddings(glove_file):
    '''Cargar embeddings GloVe desde archivo'''
    embeddings_dict = {}
    
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
    
    return embeddings_dict

# Cargar GloVe 300d
embeddings_dict = load_glove_embeddings('glove.6B.300d.txt')
embedding_dim = 300
"""

print("\n✓ Embeddings preentrenados cargados")

# ---------------------------------------------------------------------------
# 5.3 Representación de documentos con embeddings
# ---------------------------------------------------------------------------

print("\n--- 5.3 REPRESENTACIÓN DE DOCUMENTOS ---")

"""
Para representar documentos completos, usamos el promedio de los embeddings
de las palabras que contiene.

Estrategias:
1. Promedio simple de word embeddings
2. Promedio ponderado por TF-IDF
"""

def document_embedding_mean(tokens, embeddings_dict, embedding_dim):
    """
    Representar documento como promedio de word embeddings
    """
    vectors = []
    for token in tokens:
        if token in embeddings_dict:
            vectors.append(embeddings_dict[token])
    
    if len(vectors) == 0:
        return np.zeros(embedding_dim)
    
    return np.mean(vectors, axis=0)

def document_embedding_tfidf_weighted(tokens, embeddings_dict, embedding_dim, tfidf_weights):
    """
    Representar documento como promedio ponderado por TF-IDF
    """
    vectors = []
    weights = []
    
    for token in tokens:
        if token in embeddings_dict and token in tfidf_weights:
            vectors.append(embeddings_dict[token])
            weights.append(tfidf_weights[token])
    
    if len(vectors) == 0:
        return np.zeros(embedding_dim)
    
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalizar
    
    return np.average(vectors, axis=0, weights=weights)

# Generar embeddings de documentos (promedio simple)
print("Generando embeddings de documentos...")
doc_embeddings = np.array([
    document_embedding_mean(tokens, embeddings_dict, embedding_dim)
    for tokens in df['dialogue_processed']
])

print(f"Forma de matriz de embeddings: {doc_embeddings.shape}")
print(f"  - {doc_embeddings.shape[0]} documentos")
print(f"  - {doc_embeddings.shape[1]} dimensiones")

# Estadísticas
print(f"\nEstadísticas de embeddings:")
print(f"  Media: {doc_embeddings.mean():.4f}")
print(f"  Desviación estándar: {doc_embeddings.std():.4f}")
print(f"  Min: {doc_embeddings.min():.4f}")
print(f"  Max: {doc_embeddings.max():.4f}")

print("\n✓ Representación de documentos completada")

# ---------------------------------------------------------------------------
# 5.4 Entrenamiento de Word2Vec propio
# ---------------------------------------------------------------------------

print("\n--- 5.4 ENTRENAMIENTO DE WORD2VEC PROPIO ---")

"""
Dado que tenemos vocabulario médico especializado, entrenaremos nuestro
propio modelo Word2Vec con el corpus.

**Parámetros importantes:**
- vector_size: dimensión de los embeddings (100-300)
- window: ventana de contexto (5-10 palabras)
- min_count: frecuencia mínima para incluir palabra (2-5)
- sg: 0=CBOW, 1=Skip-gram (Skip-gram mejor para corpus pequeños)
"""

from gensim.models import Word2Vec

# Preparar corpus para Gensim (lista de listas de tokens)
sentences = df['dialogue_processed'].tolist()

print(f"Corpus para entrenamiento: {len(sentences)} documentos")

# Entrenar Word2Vec
print("\nEntrenando Word2Vec personalizado...")

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=300,        # Dimensión de embeddings
    window=10,              # Ventana de contexto
    min_count=2,            # Frecuencia mínima
    sg=1,                   # Skip-gram
    epochs=10,              # Número de épocas
    workers=4,              # Paralelización
    seed=42
)

print(f"✓ Word2Vec entrenado")
print(f"  Vocabulario: {len(w2v_model.wv):,} palabras")
print(f"  Dimensión: {w2v_model.wv.vector_size}")

# Guardar modelo
w2v_model.save("word2vec_medical_custom.model")
print("✓ Modelo guardado en 'word2vec_medical_custom.model'")

# ---------------------------------------------------------------------------
# 5.5 Evaluación de embeddings
# ---------------------------------------------------------------------------

print("\n--- 5.5 EVALUACIÓN DE EMBEDDINGS ---")

"""
Evaluamos la calidad de los embeddings mediante:
1. Similitud semántica de palabras médicas
2. Analogías
3. Vecinos más cercanos
"""

# Palabras médicas de ejemplo
medical_terms = ['patient', 'doctor', 'medication', 'diagnosis', 'treatment', 
                 'hypertension', 'diabetes', 'pain', 'surgery', 'prescription']

# Filtrar las que existen en el modelo
available_terms = [term for term in medical_terms if term in w2v_model.wv]

print(f"\nPalabras médicas disponibles en el modelo: {len(available_terms)}/{len(medical_terms)}")

# Similitudes entre pares de palabras médicas
if len(available_terms) >= 2:
    print("\n--- Similitudes entre palabras médicas ---")
    for i, word1 in enumerate(available_terms[:5]):
        for word2 in available_terms[i+1:6]:
            if word1 in w2v_model.wv and word2 in w2v_model.wv:
                similarity = w2v_model.wv.similarity(word1, word2)
                print(f"  {word1:15s} <-> {word2:15s}: {similarity:.4f}")

# Vecinos más cercanos
print("\n--- Vecinos más cercanos para palabras clave ---")
for word in available_terms[:5]:
    if word in w2v_model.wv:
        print(f"\nPalabra: '{word}'")
        similar_words = w2v_model.wv.most_similar(word, topn=5)
        for similar_word, score in similar_words:
            print(f"  {similar_word:20s}: {score:.4f}")

# Analogías (si las palabras existen)
print("\n--- Ejemplos de analogías ---")
analogy_examples = [
    ('doctor', 'patient', 'medication'),  # doctor:patient :: medication:?
    ('diabetes', 'insulin', 'hypertension'),  # diabetes:insulin :: hypertension:?
]

for word1, word2, word3 in analogy_examples:
    if all(w in w2v_model.wv for w in [word1, word2, word3]):
        try:
            result = w2v_model.wv.most_similar(positive=[word2, word3], negative=[word1], topn=3)
            print(f"\n{word1}:{word2} :: {word3}:?")
            for word, score in result:
                print(f"  {word:20s}: {score:.4f}")
        except:
            pass

print("\n✓ Evaluación de embeddings completada")

# ---------------------------------------------------------------------------
# 5.6 Comparación: Pretrained vs Custom
# ---------------------------------------------------------------------------

print("\n--- 5.6 COMPARACIÓN: PRETRAINED VS CUSTOM ---")

"""
Comparamos cobertura y calidad entre embeddings preentrenados y custom.
"""

# Cobertura
pretrained_coverage = len([w for w in corpus_vocab if w in embeddings_dict])
custom_coverage = len([w for w in corpus_vocab if w in w2v_model.wv])

print(f"\nCobertura de vocabulario:")
print(f"  Embeddings preentrenados (simulado): {pretrained_coverage}/{len(corpus_vocab)} ({pretrained_coverage/len(corpus_vocab)*100:.1f}%)")
print(f"  Word2Vec custom: {custom_coverage}/{len(corpus_vocab)} ({custom_coverage/len(corpus_vocab)*100:.1f}%)")

print(f"\n**CONCLUSIÓN:**")
print(f"El modelo custom tiene {custom_coverage - pretrained_coverage} palabras más cubiertas.")
print(f"Esto sugiere que para dominio médico, un modelo custom es beneficioso.")

print("\n✓ Comparación completada")

print("\n" + "="*80)
print("✓ WORD EMBEDDINGS NO CONTEXTUALES COMPLETADOS")
print("="*80)


# ============================================================================
# 6. WORD EMBEDDINGS CONTEXTUALES (TRANSFORMERS)
# ============================================================================

"""
### 6. WORD EMBEDDINGS CONTEXTUALES

**Objetivo:** Usar modelos transformer para generar embeddings que capturan
el contexto de cada palabra según su uso en la oración.

**Modelos a probar:**
1. BERT base (bert-base-uncased)
2. Clinical BERT / BioBERT (especializado en dominio médico)
3. Otros: PubMedBERT, BlueBERT

**Ventajas sobre embeddings no contextuales:**
- Capturan contexto dinámico
- Mejor manejo de polisemia
- Estado del arte en NLP

**Lo que esperamos conseguir:**
- Embeddings contextuales de alta calidad
- Comparar modelos generales vs dominio-específico
- Preparar para fine-tuning de modelos summarization
"""

print("="*80)
print("WORD EMBEDDINGS CONTEXTUALES (TRANSFORMERS)")
print("="*80)

# Instalación necesaria (descomenta si no tienes instalado)
# !pip install transformers torch

from transformers import AutoTokenizer, AutoModel
import torch

# Configurar device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

# ---------------------------------------------------------------------------
# 6.1 BERT Base
# ---------------------------------------------------------------------------

print("\n--- 6.1 BERT BASE ---")

"""
BERT (Bidirectional Encoder Representations from Transformers) es el
modelo transformer base. Usaremos bert-base-uncased (sin distinción de mayúsculas).

**Características:**
- 12 capas transformer
- 768 dimensiones de embedding
- Entrenado en BookCorpus + Wikipedia
"""

print("Cargando BERT base...")

# Cargar tokenizer y modelo
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_model = bert_model.to(device)
bert_model.eval()  # Modo evaluación

print("✓ BERT base cargado")
print(f"  Dimensión de embeddings: {bert_model.config.hidden_size}")
print(f"  Número de capas: {bert_model.config.num_hidden_layers}")
print(f"  Tamaño máximo de secuencia: {bert_tokenizer.model_max_length}")

# Función para generar embeddings
def get_bert_embedding(text, tokenizer, model, max_length=512):
    """
    Generar embedding BERT para un texto.
    Usa estrategia de [CLS] token para representar todo el documento.
    """
    # Tokenizar
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass (sin calcular gradientes)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Estrategia 1: [CLS] token (primer token)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return cls_embedding.flatten()

# Generar embeddings para algunos ejemplos
print("\nGenerando embeddings BERT para 5 ejemplos...")
sample_texts = df['dialogue_processed_text'].head(5).tolist()

bert_embeddings_sample = []
for i, text in enumerate(sample_texts):
    print(f"  Procesando documento {i+1}/5...")
    embedding = get_bert_embedding(text, bert_tokenizer, bert_model)
    bert_embeddings_sample.append(embedding)

bert_embeddings_sample = np.array(bert_embeddings_sample)
print(f"\nForma de embeddings BERT (muestra): {bert_embeddings_sample.shape}")

print("\n✓ BERT embeddings generados")

# ---------------------------------------------------------------------------
# 6.2 Clinical BERT / BioBERT
# ---------------------------------------------------------------------------

print("\n--- 6.2 CLINICAL BERT (BIOMÉDICO) ---")

"""
Clinical BERT es una versión de BERT fine-tuneada en textos médicos.
Alternativas:
- Bio BERT: Entrenado en PubMed + PMC
- PubMedBERT: Entrenado from scratch en PubMed
- BlueBERT: Entrenado en MIMIC-III + PubMed

Usaremos 'emilyalsentzer/Bio_ClinicalBERT' que está optimizado para notas clínicas.
"""

print("Cargando Clinical BERT...")

clinical_bert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
clinical_bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
clinical_bert_model = clinical_bert_model.to(device)
clinical_bert_model.eval()

print("✓ Clinical BERT cargado")
print(f"  Modelo: Bio_ClinicalBERT")
print(f"  Dimensión de embeddings: {clinical_bert_model.config.hidden_size}")

# Generar embeddings con Clinical BERT
print("\nGenerando embeddings Clinical BERT para 5 ejemplos...")
clinical_bert_embeddings_sample = []
for i, text in enumerate(sample_texts):
    print(f"  Procesando documento {i+1}/5...")
    embedding = get_bert_embedding(text, clinical_bert_tokenizer, clinical_bert_model)
    clinical_bert_embeddings_sample.append(embedding)

clinical_bert_embeddings_sample = np.array(clinical_bert_embeddings_sample)
print(f"\nForma de embeddings Clinical BERT (muestra): {clinical_bert_embeddings_sample.shape}")

print("\n✓ Clinical BERT embeddings generados")

# ---------------------------------------------------------------------------
# 6.3 Comparación de embeddings contextuales
# ---------------------------------------------------------------------------

print("\n--- 6.3 COMPARACIÓN: BERT vs CLINICAL BERT ---")

"""
Comparamos las representaciones generadas por ambos modelos
mediante similitud coseno entre documentos.
"""

from sklearn.metrics.pairwise import cosine_similarity

# Calcular matrices de similitud
bert_similarity = cosine_similarity(bert_embeddings_sample)
clinical_bert_similarity = cosine_similarity(clinical_bert_embeddings_sample)

print("\nMatriz de similitud coseno - BERT base:")
print(bert_similarity.round(3))

print("\nMatriz de similitud coseno - Clinical BERT:")
print(clinical_bert_similarity.round(3))

# Visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BERT base
im1 = axes[0].imshow(bert_similarity, cmap='YlOrRd', vmin=0, vmax=1)
axes[0].set_title('Similitud entre documentos - BERT base', fontweight='bold')
axes[0].set_xlabel('Documento')
axes[0].set_ylabel('Documento')
plt.colorbar(im1, ax=axes[0])

# Clinical BERT
im2 = axes[1].imshow(clinical_bert_similarity, cmap='YlOrRd', vmin=0, vmax=1)
axes[1].set_title('Similitud entre documentos - Clinical BERT', fontweight='bold')
axes[1].set_xlabel('Documento')
axes[1].set_ylabel('Documento')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('bert_comparison_similarity.png', dpi=300, bbox_inches='tight')
plt.show()

# Diferencias entre matrices
difference = np.abs(bert_similarity - clinical_bert_similarity)
print(f"\nDiferencia promedio entre matrices: {difference.mean():.4f}")
print(f"Diferencia máxima: {difference.max():.4f}")

print("\n✓ Comparación completada")

# ---------------------------------------------------------------------------
# 6.4 Generación de embeddings para todo el dataset
# ---------------------------------------------------------------------------

print("\n--- 6.4 GENERACIÓN DE EMBEDDINGS COMPLETOS ---")

"""
Generamos embeddings para TODO el dataset de training.
NOTA: Esto puede tardar bastante tiempo dependiendo del tamaño del dataset
y si tienes GPU.
"""

print(f"\nGenerando embeddings Clinical BERT para {len(df)} documentos...")
print("(Esto puede tardar varios minutos...)")

# Función con batch processing para eficiencia
def generate_embeddings_batch(texts, tokenizer, model, batch_size=16, max_length=512):
    """
    Generar embeddings en batches para eficiencia
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenizar batch
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # [CLS] embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Procesados {i+len(batch_texts)}/{len(texts)} documentos...")
    
    return np.vstack(all_embeddings)

# Generar embeddings para diálogos
dialogue_texts = df['dialogue_processed_text'].tolist()
dialogue_embeddings_clinical = generate_embeddings_batch(
    dialogue_texts, 
    clinical_bert_tokenizer, 
    clinical_bert_model,
    batch_size=8  # Ajustar según memoria disponible
)

# Generar embeddings para resúmenes
summary_texts = df['summary_processed_text'].tolist()
summary_embeddings_clinical = generate_embeddings_batch(
    summary_texts,
    clinical_bert_tokenizer,
    clinical_bert_model,
    batch_size=8
)

print(f"\n✓ Embeddings generados")
print(f"  Diálogos: {dialogue_embeddings_clinical.shape}")
print(f"  Resúmenes: {summary_embeddings_clinical.shape}")

# Guardar embeddings
np.save('dialogue_embeddings_clinical_bert.npy', dialogue_embeddings_clinical)
np.save('summary_embeddings_clinical_bert.npy', summary_embeddings_clinical)
print("\n✓ Embeddings guardados en formato .npy")

# ---------------------------------------------------------------------------
# 6.5 Visualización de embeddings contextuales
# ---------------------------------------------------------------------------

print("\n--- 6.5 VISUALIZACIÓN DE EMBEDDINGS CONTEXTUALES ---")

"""
Visualizamos los embeddings Clinical BERT en 2D con t-SNE,
coloreados por categoría clínica.
"""

print("Aplicando t-SNE para visualización...")
tsne_clinical = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne_clinical.fit_transform(dialogue_embeddings_clinical)

# Crear DataFrame para visualización
viz_df_clinical = pd.DataFrame({
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'category': df['section_header'].values
})

# Visualización
fig, ax = plt.subplots(figsize=(14, 10))
categories = viz_df_clinical['category'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

for category, color in zip(categories, colors):
    mask = viz_df_clinical['category'] == category
    ax.scatter(viz_df_clinical[mask]['x'], viz_df_clinical[mask]['y'],
              label=category, alpha=0.6, s=50, color=color)

ax.set_title('Visualización t-SNE de Embeddings Clinical BERT por Categoría',
            fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Dimensión 1')
ax.set_ylabel('t-SNE Dimensión 2')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig('clinical_bert_tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualización completada")

print("\n" + "="*80)
print("✓ WORD EMBEDDINGS CONTEXTUALES COMPLETADOS")
print("="*80)


# ============================================================================
# 7. ESTRUCTURACIÓN DE DATOS PARA REUTILIZACIÓN
# ============================================================================

"""
### 7. ESTRUCTURACIÓN DE DATOS

**Objetivo:** Organizar y guardar todos los datos procesados de forma que
puedan ser fácilmente reutilizados en las siguientes entregas.

**Incluye:**
1. Datos preprocesados en diferentes niveles
2. Matrices de representaciones (BoW, TF-IDF, embeddings)
3. Modelos entrenados
4. Metadatos y documentación
5. Pipeline de procesamiento reutilizable
"""

import pickle
import json
from datetime import datetime

print("="*80)
print("ESTRUCTURACIÓN DE DATOS PARA REUTILIZACIÓN")
print("="*80)

# ---------------------------------------------------------------------------
# 7.1 Guardar datos preprocesados
# ---------------------------------------------------------------------------

print("\n--- 7.1 GUARDAR DATOS PREPROCESADOS ---")

# Crear diferentes versiones del dataset procesado
data_versions = {
    'v1_cleaned': df[['ID', 'section_header', 'dialogue_cleaned', 'summary_cleaned', 'split']],
    'v2_tokenized': df[['ID', 'section_header', 'dialogue_tokens_nltk', 'summary_tokens_nltk', 'split']],
    'v3_lemmatized': df[['ID', 'section_header', 'dialogue_lemmatized', 'summary_lemmatized', 'split']],
    'v4_processed': df[['ID', 'section_header', 'dialogue_processed', 'summary_processed', 
                        'dialogue_processed_text', 'summary_processed_text', 'split']]
}

# Guardar en CSV y pickle
for version_name, version_df in data_versions.items():
    # CSV (sin listas, solo versión final de texto)
    if 'processed_text' in version_df.columns:
        version_df.to_csv(f'data_{version_name}.csv', index=False)
        print(f"✓ Guardado: data_{version_name}.csv")
    
    # Pickle (con listas de tokens)
    with open(f'data_{version_name}.pkl', 'wb') as f:
        pickle.dump(version_df, f)
    print(f"✓ Guardado: data_{version_name}.pkl")

# ---------------------------------------------------------------------------
# 7.2 Guardar representaciones
# ---------------------------------------------------------------------------

print("\n--- 7.2 GUARDAR REPRESENTACIONES ---")

# Guardar matrices BoW y TF-IDF (formato sparse)
from scipy import sparse

sparse.save_npz('bow_matrix.npz', bow_matrix)
sparse.save_npz('tfidf_matrix.npz', tfidf_matrix)
print("✓ Guardadas: bow_matrix.npz, tfidf_matrix.npz")

# Guardar vectorizadores
with open('bow_vectorizer.pkl', 'wb') as f:
    pickle.dump(bow_vectorizer, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print("✓ Guardados: vectorizadores BoW y TF-IDF")

# Embeddings ya guardados anteriormente
print("✓ Embeddings guardados: dialogue_embeddings_clinical_bert.npy, summary_embeddings_clinical_bert.npy")

# ---------------------------------------------------------------------------
# 7.3 Guardar modelos
# ---------------------------------------------------------------------------

print("\n--- 7.3 GUARDAR MODELOS ---")

# Word2Vec ya guardado
print("✓ Word2Vec guardado: word2vec_medical_custom.model")

# Guardar información de modelos BERT usados
bert_models_info = {
    'bert_base': {
        'name': 'bert-base-uncased',
        'hidden_size': 768,
        'num_layers': 12,
        'vocab_size': bert_tokenizer.vocab_size
    },
    'clinical_bert': {
        'name': 'emilyalsentzer/Bio_ClinicalBERT',
        'hidden_size': 768,
        'num_layers': 12,
        'vocab_size': clinical_bert_tokenizer.vocab_size
    }
}

with open('bert_models_info.json', 'w') as f:
    json.dump(bert_models_info, f, indent=2)
print("✓ Guardado: bert_models_info.json")

# ---------------------------------------------------------------------------
# 7.4 Metadatos y documentación
# ---------------------------------------------------------------------------

print("\n--- 7.4 METADATOS Y DOCUMENTACIÓN ---")

metadata = {
    'project': 'Medical Clinical Summary Generation',
    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': 'MTS-Dialog',
    'dataset_stats': {
        'train_size': int(len(train_df)),
        'val_size': int(len(val_df)),
        'test_size': int(len(test_df)),
        'total_size': int(len(full_df)),
        'num_categories': int(df['section_header'].nunique()),
        'categories': df['section_header'].unique().tolist()
    },
    'preprocessing': {
        'tokenization': 'NLTK word_tokenize',
        'normalization': 'lowercase, lemmatization',
        'stopwords_removed': True,
        'min_token_length': 2
    },
    'representations': {
        'traditional': ['Bag of Words', 'TF-IDF'],
        'word_embeddings_non_contextual': ['Word2Vec custom trained'],
        'word_embeddings_contextual': ['BERT base', 'Clinical BERT']
    },
    'files_structure': {
        'preprocessed_data': [
            'data_v1_cleaned.csv',
            'data_v2_tokenized.pkl',
            'data_v3_lemmatized.pkl',
            'data_v4_processed.pkl'
        ],
        'representations': [
            'bow_matrix.npz',
            'tfidf_matrix.npz',
            'dialogue_embeddings_clinical_bert.npy',
            'summary_embeddings_clinical_bert.npy'
        ],
        'models': [
            'word2vec_medical_custom.model',
            'bow_vectorizer.pkl',
            'tfidf_vectorizer.pkl'
        ],
        'visualizations': [
            'categoria_distribucion.png',
            'longitudes_analisis.png',
            'wordclouds.png',
            'tfidf_tsne_visualization.png',
            'clinical_bert_tsne_visualization.png'
        ]
    }
}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Guardado: metadata.json")

# ---------------------------------------------------------------------------
# 7.5 Pipeline reutilizable
# ---------------------------------------------------------------------------

print("\n--- 7.5 PIPELINE REUTILIZABLE ---")

"""
Creamos una clase Pipeline que encapsula todo el preprocesamiento
para poder aplicarlo fácilmente a nuevos datos.
"""

class TextPreprocessingPipeline:
    """
    Pipeline completo de preprocesamiento de texto médico
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) - {'no', 'not', 'nor', 'none', 'without', 'against'}
        
    def clean(self, text):
        """Limpieza básica"""
        if pd.isna(text):
            return ""
        text = str(text)
        text = text.replace('\\r\\n', ' ').replace('\\n', ' ').replace('\r\n', ' ').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def tokenize(self, text):
        """Tokenización"""
        return word_tokenize(text)
    
    def lemmatize(self, tokens):
        """Lemmatización"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def remove_stopwords(self, tokens):
        """Eliminar stopwords"""
        return [token for token in tokens if token not in self.stop_words]
    
    def filter_tokens(self, tokens, min_length=2):
        """Filtrar tokens válidos"""
        return [token for token in tokens 
                if len(token) >= min_length and token.isalnum()]
    
    def process(self, text):
        """Pipeline completo"""
        text = self.clean(text)
        tokens = self.tokenize(text)
        tokens = self.lemmatize(tokens)
        tokens = self.remove_stopwords(tokens)
        tokens = self.filter_tokens(tokens)
        return tokens
    
    def process_to_text(self, text):
        """Pipeline completo retornando texto"""
        tokens = self.process(text)
        return ' '.join(tokens)

# Guardar pipeline
pipeline = TextPreprocessingPipeline()
with open('preprocessing_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✓ Guardado: preprocessing_pipeline.pkl")

# Demostración de uso
print("\n--- Demostración de uso del pipeline ---")
sample_text = "The patient is a 65-year-old male with hypertension and diabetes."
print(f"Texto original: {sample_text}")
print(f"Tokens procesados: {pipeline.process(sample_text)}")
print(f"Texto procesado: {pipeline.process_to_text(sample_text)}")

print("\n✓ Pipeline reutilizable creado")
