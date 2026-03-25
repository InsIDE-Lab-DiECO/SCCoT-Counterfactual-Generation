# SCCoT: Self-Supervised Counterfactual Text Generation via Control Token Conditioning

This repository contains the official implementation of **SCCoT**, a unified framework for classification and counterfactual text generation using a single masked language model (MLM). [cite_start]This work was presented at the 30th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2026)[cite: 5, 6].

## 📖 Overview

[cite_start]Counterfactual text explanations traditionally rely on separate models for classification and generation[cite: 14]. [cite_start]SCCoT unifies these tasks within a single RoBERTa-based architecture using **control token conditioning**[cite: 15, 16]. 

By prepending a learned control token (e.g., `[POSITIVE]` or `[NEGATIVE]`), the model learns to:
1. [cite_start]Predict the sentiment class at the control token position[cite: 49].
2. [cite_start]Generate label-conditioned counterfactuals through masked infilling[cite: 49].

[cite_start]This bidirectional relationship is optimized via a novel composite loss function combining masked language modeling, auxiliary control token prediction, and contrastive embedding separation[cite: 50].

## 📂 Repository Contents

* **`src/cf_text_utils.py`**: The core engine of the framework. Contains the `CustomTrainer` with the composite loss function, feature attribution methods (SHAP and Integrated Gradients) for token masking, and custom Beam Search algorithms (Standard, Optimized, and Contrastive) for counterfactual decoding.
* **`src/cf_metrics.py`**: Evaluation suite for counterfactual quality. Includes automated metrics (Flip Rate, Probability Change, Token Distance, Perplexity/Diversity via GPT-2) and an OpenAI API integration for qualitative assessment (Grammar, Cohesiveness, Fluency).
* **`src/env_setup.py`**: Centralized environment configuration and library imports.
* **`scripts/cf_eval_loop.py`**: The main evaluation script. It iterates through the test set, applies SHAP-guided masking across varying thresholds, generates counterfactuals via beam search, and plots the resulting Flip Rate vs. Masking Percentage curve.
* **`notebooks/Loss_Ablations.ipynb`**: Jupyter notebook demonstrating the data preparation, model fine-tuning with the custom data collator (`DataCollatorWithForcedLabelMasking`), and ablation studies on the control token embeddings.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed. Install the required dependencies:

```bash
pip install torch transformers datasets shap captum scikit-learn pandas numpy matplotlib seaborn nltk spacy openai python-Levenshtein
python -m spacy download en_core_web_sm
