## Objectives for the Fourth Deliverable

Select, apply, evaluate, and compare transformer-based models from the Hugging Face platform for our clinical text classification and summarization tasks, taking into account our hardware constraints (6/12 GB VRAM).

Transformer-based Classification Model: `distilbert-base-uncased (DistilBERT)`

Transformer-based Summarization Model: `t5-small (text-to-text)`

### Strategy:

Reuse the preprocessing pipeline from the third deliverable and clean the memory between trainings to ensure a fair comparison between models.

For both tasks: Compare the performance differences between full fine-tuning and partial fine-tuning (frozen weights).

### Evaluation Methods:

Classification: Accuracy, Precision/Recall, F1 Score, Confusion Matrix.

Summarization: ROUGE and BERTScore. Additionally, will compare these against the baselines from the previous deliverable (TF-IDF + TextRank), as well as the LSTM and Clinical BERT results.

For all the produced models, inference tests will be runned using validation sets, mostly useful in the classification task.