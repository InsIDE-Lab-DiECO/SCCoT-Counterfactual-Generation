# %%
import torch
print(torch.__version__)
print(torch.cuda.is_available())
torch.set_printoptions(sci_mode=False, precision=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available(): 
    device = torch.device("mps")
print(device)

# %%
from transformers import RobertaForSequenceClassification, RobertaForMaskedLM, RobertaTokenizer, BertTokenizer, DistilBertTokenizer, BertForSequenceClassification, BertForMaskedLM, DistilBertForSequenceClassification, DistilBertForMaskedLM, Trainer, TrainingArguments, DistilBertTokenizerFast
from transformers import pipeline,TextClassificationPipeline, DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import ParameterGrid
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from itertools import chain, product
import random
import time
import pickle
import pandas as pd
import numpy as np
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DataCollatorWithPadding
from captum.attr import LayerIntegratedGradients, visualization
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
import string
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML, display
import shap
import math
import json
#import openai
from transformers import EarlyStoppingCallback
from datasets import load_dataset

from cf_metrics import Metrics
from cf_metrics import evaluate_text_quality
from cf_metrics import compute_perplexity
metrics = Metrics()

# %%
from cf_text_utils import (
    DataCollatorWithForcedLabelMasking,
    CustomTrainer,
    get_prediction,
    get_mlm_prediction,
    fill_all_masks_rnd_sampling,
    mask_tokens_with_shap,
    get_gradient_saliency,
    mask_tokens_with_gradients,
    create_shap_explainer,
    fill_all_masks_beam,
    apply_masks_from_scores,
    get_shap_scores
)

# %%
# fill-mask is responsible for predicting replacements for the masked tokens in our test sentences
tokenizer = RobertaTokenizer.from_pretrained("tokenizer_IMDB")

# Load fine-tuned models
model = RobertaForSequenceClassification.from_pretrained("IMDB_model_roberta_classifier").to(device)
model_m = RobertaForMaskedLM.from_pretrained("IMDB_model_mlm_roberta_re-finetuned").to(device)

model.eval();
model_m.eval();
fill_mask_pipeline = pipeline("fill-mask", model=model_m, tokenizer=tokenizer, device=device)

# %%
df = pd.read_csv('IMDB Dataset.csv')
df = df.rename(columns={'sentiment': 'label'})
df['label'] = df['label'].replace({'positive': 1, 'negative': 0})

train_df = pd.read_csv("data/train.csv", index_col=0)
val_df = pd.read_csv("data/val.csv", index_col=0)
test_df = pd.read_csv("data/test.csv", index_col=0)
test_df_ceval = pd.read_csv("data/test_df_ceval.csv", index_col=0)

# %%
# subset of indices used in CEval present in my test set
ceval_test = test_df[test_df.index.isin(test_df_ceval.index)]
test_sentences = ceval_test['review']

# %%
# 2. Create the explainer using the helper from the refactored file
explainer = create_shap_explainer(model_m, tokenizer, device=device)

# Filter for reviews with fewer than 50 words
test_sentences = test_df[test_df['review'].str.split().str.len() < 100]['review'].tolist()

print(f"Selected {len(test_sentences)} sentences for processing.")


# %%
masking_perc = np.round(np.arange(0.05, 1.0, 0.05), 2)


# 1. Initialize storage for per-percentage results
results_per_perc = {perc: {"y_orig": [], "y_cf": [], "cf_counts": [], "texts_cf": []} for perc in masking_perc}
pred_probas = {perc: {"probas_o": [], "probas_c": []} for perc in masking_perc}

for test_sentence in tqdm(test_sentences, desc="Processing sentences"):
    
    y_o, probas_o = get_mlm_prediction(test_sentence, model_m, tokenizer, return_probas=True)
    important_tokens, predicted_class = get_shap_scores(test_sentence, model_m, tokenizer, explainer, device=device)
    
    num_tokens = len(test_sentence.split())

    for i in masking_perc:

        actual_mask_count = max(1, int(num_tokens * i)) 
        masked_sentence = apply_masks_from_scores(test_sentence, important_tokens, predicted_class, actual_mask_count)

        found_counterfactuals, found_prototypes = fill_all_masks_beam(
            test_sentence, masked_sentence, fill_mask_pipeline, 
            model_m, tokenizer, device, beam_size=30
        )
        
        if found_counterfactuals:

            results_per_perc[i]["cf_counts"].append(len(found_counterfactuals))
            results_per_perc[i]["texts_cf"].append(found_counterfactuals)

            # Strip control tokens and predict
            clean_cf = found_counterfactuals[0].replace('[POSITIVE]', '').replace('[NEGATIVE]', '')
            y_c, probas_c = get_mlm_prediction(clean_cf, model_m, tokenizer, return_probas=True)
        else:
            results_per_perc[i]["cf_counts"].append(0)
            results_per_perc[i]["texts_cf"].append([])
            y_c = y_o  # No counterfactual found, so no flip occurred
            probas_c = probas_o

            
        # Store in our dictionary
        results_per_perc[i]["y_orig"].append(y_o)
        results_per_perc[i]["y_cf"].append(y_c)

        pred_probas[i]["probas_o"].append(probas_o)
        pred_probas[i]["probas_c"].append(probas_c)

# %%
# 2. Calculate Flip Rates
flip_rates = []
for i in masking_perc:
    fr = metrics.flip_rate(results_per_perc[i]["y_orig"], results_per_perc[i]["y_cf"])
    flip_rates.append(fr)
    print(f"Masking {i*100:.0f}%: Flip Rate = {fr:.4f}")


# Save raw result
with open('results_per_perc.pkl', 'wb') as f:
    pickle.dump(results_per_perc, f)


# Convert masking_perc to percentages for cleaner X-axis labels (e.g., 5, 10, 15...)
x_values = [round(i * 100, 2) for i in masking_perc]

plt.figure(figsize=(8, 5))

# Plotting the line
plt.plot(x_values, flip_rates, marker='o', linestyle='-', color='#2c3e50', linewidth=2, markersize=8)

# Adding labels and title
plt.title("Flip Rate vs. Masking Percentage (SHAP)", fontsize=14, fontweight='bold')
plt.xlabel("Tokens Masked (%)", fontsize=12)
plt.ylabel("Flip Rate (Label Flips / Total Samples)", fontsize=12)

# Set the Y-axis to always show 0 to 1 range (since it's a rate)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)


for x, y in zip(x_values, flip_rates):
    plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('flip_rate-tokens_masked_test_set_900.png')
plt.show()
