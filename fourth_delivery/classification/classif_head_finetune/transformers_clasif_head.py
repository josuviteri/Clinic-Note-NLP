# this file is runned because of latency in jupyter notebook kernel selection

import pandas as pd
import re
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import DistilBertTokenizer, DistilBertModel

# carga del dataset
df = pd.read_csv("../dataset/MTS-Dialog-TrainingSet.csv")

# preprocesamiento para BERT
def normalize_for_bert(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r'\b(Doctor|Doctor_2|Patient|Guest_family(_\d)?|Guest_clinician)[:\-]\s*', '', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

df['text_for_bert'] = df['dialogue'].apply(normalize_for_bert)


X = df['text_for_bert']
y = df['section_header']

# Encode 
le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# carga del tokenizer y modelo
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_name = "distilbert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(le.classes_),
    problem_type="single_label_classification"
)

# FREEZE BASE MODEL ENCODER WEIGHTS
for param in model.distilbert.parameters(): 
    param.requires_grad = False

# Tokenización
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

# datasets
train_dataset = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
test_dataset = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# métricas de evaluación
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro')
    }

# configuración del entrenamiento
training_args = TrainingArguments(
    output_dir='./results_distilbert',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    logging_dir='./logs',
    logging_steps=10,
    seed=42,
    fp16=True,
    gradient_accumulation_steps=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Fine-tune
trainer.train()

# Evaluación
results = trainer.evaluate()
print(results)

# Guardar modelo
model.save_pretrained('./finetuned_distilbert')
tokenizer.save_pretrained('./finetuned_distilbert')

# Guardar encoder de labels
import pickle
with open('./finetuned_distilbert/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)