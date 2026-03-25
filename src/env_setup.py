from transformers import RobertaForSequenceClassification, RobertaForMaskedLM, RobertaTokenizer # ==4.30.0 must be
from transformers import pipeline,TextClassificationPipeline, DataCollatorForLanguageModeling, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer
from sklearn.model_selection import ParameterGrid
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.notebook import tqdm
from itertools import chain, product
import random
import time
import pandas as pd
import numpy as np
import re
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DataCollatorWithPadding
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
import openai
from transformers import EarlyStoppingCallback
from datasets import load_dataset
from cf_metrics import Metrics
from cf_metrics import evaluate_text_quality
from cf_metrics import compute_perplexity
metrics = Metrics()

from cf_text_utils import (
    DataCollatorWithForcedLabelMasking,
    CustomTrainer,
    get_prediction,
    get_mlm_prediction,
    classify_with_restriction,
    get_topk_predictions,
    fill_all_masks_rnd_sampling,
    mask_tokens_with_shap,
    get_gradient_saliency,
    get_integrated_gradients,
    mask_tokens_with_gradients,
    create_shap_explainer,
    fill_all_masks_beam,
    fill_all_masks_beam_optimized,
    fill_all_masks_beam_contrastive,
    get_shap_scores,
    get_gradient_scores_aligned,
    apply_masks_from_scores
)