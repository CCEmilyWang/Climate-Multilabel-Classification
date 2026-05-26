# Climate Multi-Label Classification

A research-oriented NLP project for multi-label classification of climate-related documents using BERT-based models and LLM-assisted summarization pipelines.

---

# Overview

Climate-related documents often contain:

- Dense semantic overlap
- Long-tail label distributions
- Multi-topic associations
- Severe class imbalance

This project explores a cascaded NLP framework combining:

1. Filtering
2. LLM-based summarization
3. Multi-label classification

to improve classification performance on challenging climate-related text datasets.

---

# Research Motivation

Traditional single-stage classifiers often struggle with:

- Minority labels
- Long documents
- Ambiguous semantic boundaries
- Imbalanced label distributions

This project investigates whether LLM-generated summaries can improve downstream classification quality while maintaining efficient inference performance.

---

# Methodology

The framework consists of three stages:

## 1. Filtering Stage
A lightweight filtering module removes irrelevant or low-confidence samples before downstream processing.

## 2. Summarization Stage
A fine-tuned Llama-3.1 model generates concise summaries of climate-related documents to reduce semantic redundancy and improve label separability.

## 3. Multi-Label Classification
A BERT-based classifier predicts multiple topic labels using weighted BCEWithLogitsLoss to address long-tail label imbalance.

---

# Experimental Features

- BERT fine-tuning
- LLM-assisted summarization
- Weighted BCEWithLogitsLoss
- Long-tail label optimization
- Confidence-aware prediction
- vLLM inference acceleration

---

# Results

Key observations from experiments:

- Improved Macro-F1 performance on minority labels by approximately 8%
- Improved robustness for dense semantic documents
- Achieved approximately 2.5× faster inference throughput using vLLM

---

# Repository Structure

```text
climate-multilabel-classification/
│
├── README.md
├── requirements.txt
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   └── utils.py
│
├── configs/
│   └── config.yaml
│
├── experiments/
│   └── results.md
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

# Technologies

- Python
- PyTorch
- Hugging Face Transformers
- vLLM
- Llama-3.1
- Scikit-learn

---

# Future Work

Potential future directions include:

- Hierarchical label modeling
- Retrieval-augmented classification
- Calibration-aware confidence estimation
- Explainable multi-label prediction
- Efficient long-context LLM inference

---

# Notes

The dataset is not publicly included due to privacy and storage constraints.

This repository is intended for research and educational purposes.
