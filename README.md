# SCCoT: Self-Supervised Counterfactual Text Generation via Control Token Conditioning

This repository contains the official implementation of **SCCoT**, a unified framework for classification and counterfactual text generation using a single masked language model (MLM). 


## 📂 Repository Contents

* **`src/cf_text_utils.py`**: Contains the `CustomTrainer` with the composite loss function, feature attribution methods (SHAP and Integrated Gradients) for token masking, and custom Beam Search algorithms (Standard, Optimized, and Contrastive) for counterfactual decoding.
* **`src/cf_metrics.py`**: Evaluation suite for counterfactual quality. Includes automated metrics (Flip Rate, Probability Change, Token Distance, Perplexity/Diversity) and an OpenAI API integration for qualitative assessment (Grammar, Cohesiveness, Fluency).
* **`src/env_setup.py`**: Environment configuration and library imports.
* **`scripts/cf_eval_loop.py`**: Main evaluation script. It iterates through the test set, applies SHAP-guided masking across varying thresholds, generates counterfactuals via beam search, and plots the resulting Flip Rate vs. Masking Percentage curve.
* **`notebooks/Loss_Ablations.ipynb`**: Notebook of ablation studies.

## 🚀 Getting Started

### Prerequisites

- Ensure you have **Python 3.9+** installed.
- Install the required dependencies from the `requirements.txt` file:


