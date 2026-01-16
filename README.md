# Clinic Note NLP Project

A comprehensive Natural Language Processing project focused on clinical dialogue analysis, featuring medical text classification and summarization using traditional machine learning, deep learning, and transformer-based approaches.

## 📋 Project Overview

This project explores multiple NLP techniques to process clinical dialogues between doctors and patients, converting conversational text into structured medical notes. The work is based on the MEDIQA-Chat and MEDIQA-Sum 2023 datasets and implements various classification and summarization models.

### Key Tasks

1. **Medical Dialogue Classification**: Categorizing clinical conversations into predefined section headers (e.g., GENHX, MEDICATIONS, CC, PASTMEDICALHX)
2. **Clinical Conversation Summarization**: Generating concise medical summaries from doctor-patient dialogues

## 📊 Dataset

The project uses the MTS-Dialog dataset from MEDIQA 2023:

- **Training Set**: `MTS-Dialog-TrainingSet.csv`
- **Validation Set**: `MTS-Dialog-ValidationSet.csv`
- **Test Sets**: 
  - `MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv`
  - `MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv`

Each dataset contains clinical dialogues with corresponding section headers and structured section texts.

## 🏗️ Project Structure

```
.
├── dataset/                          # MEDIQA-Chat and MEDIQA-Sum 2023 datasets
├── embedding_projector/              # Embedding visualizations
│   ├── clinical_bert_embeddings_tsv.tsv
│   ├── elmo_embeddings_tsv.tsv
│   └── projector_config.pbtxt
├── processed/                        # Processed models and artifacts
│   └── custom_clinical_word2vec.model
├── second_delivery/                  # Initial embeddings and preprocessing
│   └── second_delivery.ipynb
├── third_delivery/                   # Classical ML and custom DL approaches
│   ├── classification/
│   │   ├── classif_shallow_ml.ipynb  # TF-IDF + classical ML classifiers
│   │   └── classif_cnn.ipynb         # CNN with Clinical-BERT
│   └── summarisation/
│       ├── sum_shallow_ml.ipynb      # TF-IDF + TextRank summarization
│       └── sum_cnn.ipynb             # LSTM encoder-decoder
└── fourth_delivery/                  # Transformer-based approaches
    ├── cnn_classification/
    │   └── BI-LSTM.ipynb             # Bidirectional LSTM implementation
    ├── transformers_classification/
    │   └── transformers_classification.ipynb  # DistilBERT classification
    ├── transformers_sumarisation/
    │   └── transformers_summarisation.ipynb   # T5-small summarization
    └── raspberry_llm_lab/
        └── distilGPT2.ipynb           # Distil GPT 2 model experiments
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Virtual environment recommended
- CUDA-capable GPU (6-12 GB VRAM recommended for transformer models)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Clinic-Note-NLP
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔬 Methodology

### Phase 1: Embeddings and Preprocessing (Second Delivery)

- Exploration of contextual and non-contextual embeddings
- Custom Clinical Word2Vec model training
- Analysis of vocabulary size and tokenization characteristics
- Embedding visualization using TensorBoard Projector

### Phase 2: Classical ML and CNN (Third Delivery)

#### Classification
- **Shallow ML Approach**: TF-IDF and Count Vectorizer with classical ML classifiers (Naive Bayes, SVM, Random Forest, etc.)
- **Deep Learning Approach**: Fine-tuned Clinical-BERT classification head with CNN architecture

#### Summarization
- **Shallow ML Approach**: TF-IDF and TextRank algorithms
- **Deep Learning Approach**: LSTM encoder-decoder with frozen and fine-tuned embeddings

### Phase 3: Transformer Models (Fourth Delivery)

#### Classification
- **Model**: DistilBERT (distilbert-base-uncased)
- **Techniques**: Full fine-tuning vs. partial fine-tuning (frozen weights)
- **Evaluation**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

#### Summarization
- **Model**: T5-small (text-to-text)
- **Evaluation**: ROUGE scores, BERTScore
- **Baseline Comparison**: Against TF-IDF, TextRank, LSTM, and Clinical-BERT

#### Additional Experiments
- Bidirectional LSTM architectures
- Distil GPT2 model exploration

## 📈 Evaluation Metrics

### Classification
- Accuracy
- Precision, Recall, F1 Score
- Confusion Matrix
- Inference testing on validation sets

### Summarization
- ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- BERTScore
- Qualitative analysis of generated summaries

## 🛠️ Technologies Used

### Core Libraries
- **Deep Learning**: PyTorch, TensorFlow
- **Transformers**: Hugging Face Transformers, Accelerate
- **NLP**: NLTK, spaCy
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Evaluation**: datasets, evaluate

### Pre-trained Models
- Clinical-BERT
- DistilBERT
- T5-small
- ELMo
- distil-GPT2
- Custom Clinical Word2Vec

## 💻 Usage

### Running Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to the desired delivery folder and open the notebook of interest

3. Run cells sequentially to reproduce experiments



## 📝 Key Findings

- **Embeddings**: Custom clinical embeddings capture domain-specific semantics better than general-purpose embeddings
- **Classification**: Transformer models (DistilBERT) outperform classical ML, with proper fine-tuning strategies being crucial
- **Summarization**: T5-small provides competitive results while remaining computationally feasible on limited hardware
- **Trade-offs**: Balance between model size, performance, and computational resources is critical for clinical NLP applications

## 🔧 Hardware Considerations

The project is designed to work with limited GPU resources (6-12 GB VRAM). Strategies employed:

- Use of smaller transformer variants (DistilBERT, T5-small)
- Gradient accumulation for larger effective batch sizes
- Mixed precision training (FP16)
- Memory cleanup between training sessions

## 🤝 Contributing

This is an academic project. For questions or suggestions, please open an issue.

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MEDIQA 2023 Challenge organizers for providing the datasets
- Hugging Face for their transformers library and model hub
- Clinical-BERT, T5 and distil-GPT2 authors for domain-specific pre-trained models

## 📧 Reference

For more specific information about this project, please refer to the individual delivery README files:
- [Third Delivery](third_delivery/README.md)
- [Fourth Delivery](fourth_delivery/README.md)

---

**Note**: Due to size constraints, trained model weights are not included in this repository. The notebooks contain code to reproduce all models from scratch.
