import torch
import torch.nn.functional as F
import math
import numpy as np
import re
import shap
from transformers import DataCollatorForLanguageModeling, Trainer

# ---------------------------------------------------------
# 1. Training Classes & Functions
# ---------------------------------------------------------

class DataCollatorWithForcedLabelMasking(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)

    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask=None):
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[:, 0] = 1.0  # Always mask position 0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[:, 0] = True

        labels[~masked_indices] = -100

        # 80% -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        indices_replaced[:, 0] = False
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% -> Random
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        indices_random[:, 0] = False
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # Force position 0 to be [MASK]
        inputs[:, 0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        #labels[:, 0] = -100  # ADD THIS LINE TO ABLATE ONLY FOR TESTING NO CONTROL TOKEN LOSS AT ALL   <------------- !!!
        return inputs, labels

def compute_auxiliary_loss(output_logits, input_ids):
    control_code_token_ids = input_ids[:, 0]
    aux_loss_fn = torch.nn.CrossEntropyLoss()
    logits_for_control_code = output_logits[:, 0, :]
    aux_loss = aux_loss_fn(logits_for_control_code, control_code_token_ids)
    return aux_loss

class CustomTrainer(Trainer):
    def __init__(self, *args, aux_loss_weight=0.2, contrastive_weight=0.5, use_contrastive=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = aux_loss_weight
        self.contrastive_weight = contrastive_weight
        self.use_contrastive = use_contrastive
        # Tokenizer must be passed in args or exist in self.tokenizer from Trainer init
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        mlm_loss = outputs.loss
        aux_loss = compute_auxiliary_loss(outputs.logits, inputs["input_ids"])
        
        total_loss = mlm_loss + self.aux_loss_weight * aux_loss

        contrastive_loss = torch.tensor(0.0)
        cos_sim = torch.tensor(0.0)

        if self.use_contrastive:
            emb = model.roberta.embeddings.word_embeddings.weight
            pos_id = self.tokenizer.convert_tokens_to_ids("[POSITIVE]")
            neg_id = self.tokenizer.convert_tokens_to_ids("[NEGATIVE]")

            cos_sim = torch.cosine_similarity(emb[pos_id], emb[neg_id], dim=0)
            contrastive_loss = (1 + cos_sim) / 2
            total_loss = total_loss + self.contrastive_weight * contrastive_loss

        # Logging logic (omitted slightly for brevity, but kept structure)
        if self.state.global_step % 100 == 0 or not self.model.training:
             pass

        return (total_loss, outputs) if return_outputs else total_loss

# ---------------------------------------------------------
# 2. Prediction Utilities
# ---------------------------------------------------------

def get_prediction(text, model, tokenizer, device, return_probas=False):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True, return_token_type_ids=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.max(dim=1).indices.item()
    
    if return_probas:
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()
        return predicted_class, probabilities
    
    return predicted_class

def get_mlm_prediction(text, model_m, tokenizer, device=None, return_probas=False):
    """
    Get classification prediction from MLM model using control tokens.
    """
    if device is None:
        device = model_m.device

    masked_input = f"{tokenizer.mask_token}{text}"
    inputs = tokenizer(
        masked_input,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
        return_token_type_ids=False
    ).to(device)
    
    with torch.no_grad():
        logits = model_m(**inputs).logits[0, 0]
    
    positive_id = tokenizer.convert_tokens_to_ids("[POSITIVE]")
    negative_id = tokenizer.convert_tokens_to_ids("[NEGATIVE]")
    
    control_logits = torch.tensor([logits[negative_id], logits[positive_id]])
    probabilities = F.softmax(control_logits, dim=-1).tolist()
    
    predicted_class = int(probabilities[1] > 0.5)
    
    return (predicted_class, probabilities) if return_probas else predicted_class


# Test the classification capabilities of the mlm model
def classify_with_restriction(text, model, tokenizer):
    device = next(model.parameters()).device
    # Step 1: Mask the control token position
    inputs = tokenizer(f"{tokenizer.mask_token}{text}", 
                      return_tensors="pt", 
                      truncation=True,
                      max_length=512).to(device)
    
    # Step 2: Get logits for [MASK] at position 0
    with torch.no_grad():
        logits = model(**inputs).logits[0, 0]  # Get logits for first token
    
    # Step 3: Restrict to control tokens
    positive_id = tokenizer.convert_tokens_to_ids("[POSITIVE]")
    negative_id = tokenizer.convert_tokens_to_ids("[NEGATIVE]")
    
    # Create mask and suppress all other tokens
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[[positive_id, negative_id]] = False
    restricted_logits = logits.clone()
    restricted_logits[mask] = -float('inf')
    
    # Step 4: Get probabilities
    probs = F.softmax(restricted_logits, dim=-1)
    pred_token_id = restricted_logits.argmax().item()
    
    return {
        'prediction': 1 if pred_token_id == positive_id else 0,
        'confidence': probs[positive_id].item() if pred_token_id == positive_id else probs[negative_id].item(),
        'positive_prob': probs[positive_id].item(),
        'negative_prob': probs[negative_id].item()
    }


# ---------------------------------------------------------
# 2.5. Replicate Pipeline in Pytorch
# ---------------------------------------------------------


def get_topk_predictions(sentences, model, tokenizer, device, top_k=30):
    # ensure sentences is a list
    if isinstance(sentences, str):
        sentences = [sentences]

    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    results = []

    for i in range(len(sentences)):
        # find all mask positions in sentence
        mask_index = (inputs["input_ids"][i] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        mask_logits = logits[i, mask_index, :]
        probs = F.softmax(mask_logits, dim=-1)

        top_probs, top_ids = probs.topk(top_k)

        preds = []
        for prob, tok in zip(top_probs[0], top_ids[0]):
            preds.append({
                "token_str": tokenizer.decode(tok).strip(),
                "score": prob.item()
            })

        results.append(preds)

    return results




# ---------------------------------------------------------
# 3. SHAP & Explanation Utilities
# ---------------------------------------------------------

# Optional for loop efficiency, separating shap values calculation from masking
def get_shap_scores(test_sentence, model_m, tokenizer, explainer, device='cpu'):
    """Calculate SHAP values once for a sentence."""
    predicted_class = get_mlm_prediction(test_sentence, model_m, tokenizer, device=device)
    shap_values = explainer([test_sentence], silent=True)
    
    shap_tokens = shap_values.data[0]
    # Pick scores for the predicted class
    shap_scores = shap_values.values[0][:, predicted_class] 
    
    # Filter out empty/mask tokens and sort by importance
    token_score_pairs = [(t, s) for t, s in zip(shap_tokens, shap_scores) if t.strip() not in {'', '<mask>'}]
    token_score_pairs.sort(key=lambda x: (-x[1], x[0]))
    
    return [t for t, _ in token_score_pairs], predicted_class



def create_shap_explainer(model_m, tokenizer, device, max_evals=50):
    
    pos_id = tokenizer.convert_tokens_to_ids("[POSITIVE]")
    neg_id = tokenizer.convert_tokens_to_ids("[NEGATIVE]")

    def shap_wrapper(texts):
        if isinstance(texts, str):
            texts = [texts]
        
        masked_texts = [f"{tokenizer.mask_token}{text}" for text in texts]
        inputs = tokenizer(masked_texts, return_tensors="pt", padding=True, truncation=True, 
                           max_length=512).to(device)
        
        with torch.no_grad():
            logits = model_m(**inputs).logits[:, 0]
        
        probs = F.softmax(logits[:, [neg_id, pos_id]], dim=-1)
        return probs.cpu().numpy()

    explainer = shap.Explainer(
        model=shap_wrapper,
        masker=shap.maskers.Text(tokenizer),
        algorithm="partition",
        max_evals=max_evals,
        output_names=["[NEGATIVE]", "[POSITIVE]"]
    )
    return explainer

def get_gradient_saliency(text, model_m, tokenizer, device=None):
    """Returns tokens ranked by their influence on control token prediction"""
    if device is None:
        device = next(model_m.parameters()).device
    # 1. Prepare inputs
    inputs = tokenizer(
        f"{tokenizer.mask_token}{text}", 
        return_tensors="pt",
        truncation=True, 
        max_length=512
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get current prediction
    pred_class, _ = get_mlm_prediction(text, model_m, tokenizer, return_probas=True)
    target_class = "[POSITIVE]" if pred_class == 1 else "[NEGATIVE]"

    # 2. Get embeddings and mark as leaf variable
    with torch.no_grad():
        embeddings = model_m.roberta.embeddings.word_embeddings(input_ids)
    embeddings = embeddings.clone().requires_grad_(True) 
    
    # 3. Custom forward pass
    model_m.zero_grad()
    outputs = model_m(
        inputs_embeds=embeddings,
        attention_mask=attention_mask
    )
    
    # 4. Target control token logit
    target_id = tokenizer.convert_tokens_to_ids(target_class)
    logit = outputs.logits[0, 0, target_id]
    logit.backward()
    
    # 5. Process gradients (skip [MASK] at position 0)
    grads = embeddings.grad.abs().sum(dim=-1).squeeze()[1:]
    
    # 6. Pair tokens with importance scores
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][1:])

    filtered = []
    for token, grad in zip(tokens, grads):
        if token not in [tokenizer.mask_token]:
            filtered.append((token.strip(), grad.item()))
    
    return sorted(filtered, key=lambda x: -x[1]) #[:10]
    #return filtered # <---   optionally unordered for heatmap plot creation



def get_integrated_gradients(text, model_m, tokenizer, device=None, steps=50):
    """Returns tokens ranked by Integrated Gradients on control token prediction"""
    if device is None:
        device = next(model_m.parameters()).device
        
    # 1. Prepare inputs
    inputs = tokenizer(
        f"{tokenizer.mask_token}{text}",
        return_tensors="pt",
        truncation=True,    # <--- Fixed
        max_length=512      # <--- Fixed
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get current prediction to find the target class
    pred_class, _ = get_mlm_prediction(text, model_m, tokenizer, return_probas=True)
    target_class = "[POSITIVE]" if pred_class == 1 else "[NEGATIVE]"
    target_id = tokenizer.convert_tokens_to_ids(target_class)

    # 2. Get actual input embeddings
    with torch.no_grad():
        embeddings = model_m.roberta.embeddings.word_embeddings(input_ids)
    
    # Define the baseline (a zero tensor of the same shape)
    baseline_embeddings = torch.zeros_like(embeddings)
    
    # Initialize a tensor to accumulate the gradients
    accumulated_grads = torch.zeros_like(embeddings)
    
    # 3. Interpolation Loop
    for i in range(1, steps + 1):
        alpha = i / steps
        # Create interpolated embeddings
        interpolated = baseline_embeddings + alpha * (embeddings - baseline_embeddings)
        interpolated = interpolated.clone().requires_grad_(True)
        
        # Forward pass with interpolated embeddings
        model_m.zero_grad()
        outputs = model_m(
            inputs_embeds=interpolated,
            attention_mask=attention_mask
        )
        
        # Target control token logit
        logit = outputs.logits[0, 0, target_id]
        logit.backward()
        
        # Accumulate the gradients
        accumulated_grads += interpolated.grad
        
    # 4. Compute Integrated Gradients
    # Average the gradients and multiply by (input - baseline)
    avg_grads = accumulated_grads / steps
    integrated_grads = (embeddings - baseline_embeddings) * avg_grads
    
    # 5. Process scores
    # Sum across the embedding dimension and take the absolute value for magnitude
    ig_scores = integrated_grads.sum(dim=-1).abs().squeeze()[1:] # Skip <mask> at position 0
    
    # 6. Pair tokens with importance scores
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][1:])

    filtered = []
    for token, score in zip(tokens, ig_scores):
        if token not in [tokenizer.mask_token]:
            filtered.append((token.strip(), score.item()))
    
    return sorted(filtered, key=lambda x: -x[1]) 
    #return filtered # <-------  for running the corr heatmap and other corr plots



# Optional for loop efficiency, getting same output as get_shap_scores
def get_gradient_scores_aligned(test_sentence, model_m, tokenizer, device=None):
    """
    Wraps the existing get_gradient_saliency to return the same 
    format as get_shap_scores. 
    """
    # 1. Get the list of (token, score) tuples from your provided function
    # Your function already sorts them by descending importance
    #sorted_pairs = get_gradient_saliency(test_sentence, model_m, tokenizer, device=device)
    sorted_pairs = get_integrated_gradients(test_sentence, model_m, tokenizer, device=device) 
    # 2. Extract just the token strings for the masking function
    important_tokens = [token for token, score in sorted_pairs] 
    
    # 3. Get the predicted class to determine control token direction
    predicted_class = get_mlm_prediction(test_sentence, model_m, tokenizer, device=device) 
    
    return important_tokens, predicted_class 



def mask_tokens_with_shap(test_sentence, model_m, tokenizer, explainer, n_to_mask=None, device='cpu'):
    # Get SHAP values
    predicted_class = get_mlm_prediction(test_sentence, model_m, tokenizer, device=device)
    shap_values = explainer([test_sentence], silent=True)

    
    shap_tokens = shap_values.data[0]
    # Depending on prediction, pick the column (0 for Neg, 1 for Pos)
    shap_scores = shap_values.values[0][:, predicted_class] 
    
    #token_score_pairs = [(t, s) for t, s in zip(shap_tokens, shap_scores) if s > 0 and t.strip() not in {'', '<mask>'}]
    token_score_pairs = [(t, s) for t, s in zip(shap_tokens, shap_scores) if t.strip() not in {'', '<mask>'}]
    
    token_score_pairs.sort(key=lambda x: (-x[1], x[0]))
    important_tokens = [t for t, _ in token_score_pairs]
    #print('important_tokens: ', important_tokens)


    masked_text = test_sentence
    masks_applied = 0
    
    for token in important_tokens:
        if n_to_mask is not None and masks_applied >= n_to_mask:
            break
            
        search_token = token.replace('Ġ', ' ').strip()
        if not search_token:
            continue
            
        pattern = r'(?<!\w)' + re.escape(search_token) + r'(?!\w)'
        if re.search(pattern, masked_text, flags=re.IGNORECASE):
            masked_text = re.sub(pattern, '<mask>', masked_text, count=1, flags=re.IGNORECASE)
            masks_applied += 1

    label = '[POSITIVE]' if predicted_class == 0 else '[NEGATIVE]'
    masked_text = f"{label}{masked_text}{label}"

    print('masked sentence: ', masked_text)
    return masked_text



# Optional for loop efficiency, separating shap values calculation from masking
def apply_masks_from_scores(test_sentence, important_tokens, predicted_class, n_to_mask):
    """Apply N masks using pre-calculated important tokens."""
    masked_text = test_sentence
    masks_applied = 0
    
    for token in important_tokens:
        if masks_applied >= n_to_mask:
            break
            
        search_token = token.replace('Ġ', ' ').strip()
        if not search_token: continue
            
        pattern = r'(?<!\w)' + re.escape(search_token) + r'(?!\w)'
        if re.search(pattern, masked_text, flags=re.IGNORECASE):
            masked_text = re.sub(pattern, '<mask>', masked_text, count=1, flags=re.IGNORECASE)
            masks_applied += 1

    # Flip label for the control token (target the opposite class)
    label = '[POSITIVE]' if predicted_class == 0 else '[NEGATIVE]'
    return f"{label}{masked_text}{label}"




def mask_tokens_with_gradients(test_sentence, model_m, tokenizer, n_to_mask=None):
   
    # 1. Get prediction and determine control token
    predicted_class = get_mlm_prediction(test_sentence, model_m, tokenizer)

    # 2. Get gradient-based important tokens 
    #salient_tokens = get_gradient_saliency(test_sentence, model_m, tokenizer)
    salient_tokens = get_integrated_gradients(test_sentence, model_m, tokenizer)
    important_tokens = [f' {t}' for t, _ in salient_tokens] # this is formatting is done to match SHAP list format in original mask_tokens()
    #print('important_tokens: ', important_tokens)

    # Mask most important tokens first
    masked_text = test_sentence
    masks_applied = 0
    
    for token in important_tokens:
        if masks_applied >= n_to_mask:
            break
            
        # Handle subword tokens
        search_token = token.replace('Ġ', ' ').strip()
        if not search_token:
            continue
            
        # Create regex pattern that matches whole words only
        pattern = r'(?<!\w)' + re.escape(search_token) + r'(?!\w)'
        
        # Mask first occurrence only
        if re.search(pattern, masked_text, flags=re.IGNORECASE):
            masked_text = re.sub(
                pattern,
                '<mask>',
                masked_text,
                count=1,  # Only replace first occurrence
                flags=re.IGNORECASE
            )
            masks_applied += 1
            #print(f"Masked '{search_token}' (SHAP value: {next(v for t,v in token_score_pairs if t == token):.4f})")
    
    # Prepend control token
    label = '[POSITIVE]' if predicted_class == 0 else '[NEGATIVE]'       #  <------------------------
    masked_text = f"{label}{masked_text}{label}"

    print('masked sentence: ', masked_text)
    return masked_text




# ---------------------------------------------------------
# 4. Generation & Beam Search
# ---------------------------------------------------------

def fill_all_masks_beam(test_sentence, masked_sentence, pipeline, model_m, tokenizer, device, beam_size=15):
    pipeline.device = device
    original_prediction = get_mlm_prediction(test_sentence, model_m, tokenizer, device=device)
    
    filled_sentences = [(masked_sentence, 0)]

    for _ in range(masked_sentence.count("<mask>")):
        new_filled = []
        for sent, score in filled_sentences:
            tokens = tokenizer.encode(sent, truncation=True, max_length=512, add_special_tokens=False) # new
            sent = tokenizer.decode(tokens)  # new

            predictions = pipeline(sent, top_k=beam_size)
            if not isinstance(predictions, list): predictions = [predictions]
            if isinstance(predictions[0], dict): predictions = [predictions]

            top_tokens = predictions[0][:beam_size]
            for pred in top_tokens:
                token = pred['token_str']
                token_score = pred['score']
                new_sent = sent.replace("<mask>", token, 1)
                new_filled.append((new_sent, score + token_score))

        filled_sentences = sorted(new_filled, key=lambda x: -x[1])[:beam_size]

    found_counterfactuals = []
    found_prototypes = []

    for sent, _ in filled_sentences:
        found_prototypes.append(sent)
        cf_candidate = sent.replace('[POSITIVE]', '').replace('[NEGATIVE]', '')
        new_prediction = get_mlm_prediction(cf_candidate, model_m, tokenizer, device=device)
        if new_prediction != original_prediction:
            found_counterfactuals.append(sent)

    return found_counterfactuals, found_prototypes




def fill_all_masks_beam_optimized(test_sentence, masked_sentence, pipeline, model_m, tokenizer, device, original_prediction, beam_size=15):
    pipeline.device = device
    original_prediction = get_mlm_prediction(test_sentence, model_m, tokenizer, device=device)
    
    filled_sentences = [(masked_sentence, 0.0)] # base score of 0
    
    for _ in range(masked_sentence.count("<mask>")):
        new_filled = []
        for sent, current_log_prob in filled_sentences:
            
            predictions = pipeline(sent, top_k=beam_size)
            
            if not isinstance(predictions, list): predictions = [predictions]
            if isinstance(predictions[0], dict): predictions = [predictions]

            # Grab predictions for the FIRST mask in the current sentence
            top_tokens = predictions[0][:beam_size]
            
            for pred in top_tokens:
                token = pred['token_str']
                
                # Convert to log probability (adding a small epsilon to prevent log(0))
                token_log_prob = math.log(pred['score'] + 1e-10)
                
                # Replace only the first mask
                new_sent = sent.replace("<mask>", token, 1)
                
                # ADD log probabilities
                new_filled.append((new_sent, current_log_prob + token_log_prob))

        # Sort by highest log-prob (closest to 0) and prune
        filled_sentences = sorted(new_filled, key=lambda x: x[1], reverse=True)[:beam_size]

    found_counterfactuals = []
    found_prototypes = []

    for sent, _ in filled_sentences:
        found_prototypes.append(sent)
        
        cf_candidate = sent.replace('[POSITIVE]', '').replace('[NEGATIVE]', '').strip()
        
        new_prediction = get_mlm_prediction(cf_candidate, model_m, tokenizer, device=device)
        
        if new_prediction != original_prediction:
            found_counterfactuals.append(sent)

    return found_counterfactuals, found_prototypes



def fill_all_masks_beam_contrastive(test_sentence, masked_sentence, pipeline, model_m, tokenizer, device, original_prediction, beam_size=15, alpha=5.0, pool_size = 30): 
    pipeline.device = device 
    original_prediction = get_mlm_prediction(test_sentence, model_m, tokenizer, device=device) 
    filled_sentences = [(masked_sentence, 0.0)] # initial base score of 0 

    # our target and opposite labels 
    target_label = '[POSITIVE]' if '[POSITIVE]' in masked_sentence else '[NEGATIVE]' 
    opposite_label = '[NEGATIVE]' if target_label == '[POSITIVE]' else '[POSITIVE]' 


    for _ in range(masked_sentence.count("<mask>")): 
        new_filled = [] 

        # BOTH BATCHES 
        current_sents = [sent for sent, _ in filled_sentences] 
        opposite_sents = [sent.replace(target_label, opposite_label) for sent in current_sents] 

        # INFERENCE FOR BOTH CONTEXTS 
        batch_preds_target = pipeline(current_sents, top_k=pool_size, batch_size=len(current_sents)) 
        batch_preds_opp = pipeline(opposite_sents, top_k=pool_size, batch_size=len(opposite_sents)) 

        #batch_preds_target = get_topk_predictions(current_sents, model_m, tokenizer, device, pool_size) 
        #batch_preds_opp = get_topk_predictions(opposite_sents, model_m, tokenizer, device, pool_size) 

        # NORMALIZE OUTPUTS 
        def normalize_preds(batch_predictions, num_sents): 
            norm = [] 
            if num_sents == 1: batch_predictions = [batch_predictions] 
            for preds in batch_predictions: 
                if isinstance(preds[0], list): norm.append(preds[0]) 
                else: norm.append(preds) 
            return norm 

        norm_target = normalize_preds(batch_preds_target, len(current_sents)) 
        norm_opp = normalize_preds(batch_preds_opp, len(opposite_sents)) 

        # CALCULATE CONTRASTIVE SCORES 
        for i, (sent, current_log_prob) in enumerate(filled_sentences): 
            target_preds = norm_target[i] 
            opp_preds = norm_opp[i] 

            opp_dict = {p['token_str']: p['score'] for p in opp_preds} 

            scored_tokens = [] 
            for pred in target_preds: 
                tok = pred['token_str'] 
                prob_t = pred['score'] 

                # If token isn't in opposite's top_k, put a tiny proba 
                prob_o = opp_dict.get(tok, 1e-8) 

                log_t = math.log(prob_t + 1e-10) # log_proba token 
                log_o = math.log(prob_o + 1e-10) # log_proba token for opposite label 

                # Correlation score 
                correlation = log_t - log_o 

                # Mix proba token with correlation that constraint 
                final_token_score = log_t + (alpha * correlation) 

                scored_tokens.append((tok, final_token_score)) 

            # Sort the tokens for THIS branch by the new contrastive score 
            best_tokens = sorted(scored_tokens, key=lambda x: x[1], reverse=True)[:beam_size] 

            # EXPAND THE BEAM 
            for tok, final_score in best_tokens: 
                new_sent = sent.replace("<mask>", tok, 1) 
                new_filled.append((new_sent, current_log_prob + final_score)) 

        # PRUNE THE OVERALL BEAM 
        filled_sentences = sorted(new_filled, key=lambda x: x[1], reverse=True)[:beam_size] 

    found_counterfactuals, found_prototypes = [], [] 
    for sent, _ in filled_sentences: 
        found_prototypes.append(sent) 
        cf_candidate = sent.replace('[POSITIVE]', '').replace('[NEGATIVE]', '').strip() 
        new_prediction = get_mlm_prediction(cf_candidate, model_m, tokenizer, device=device) 
        if new_prediction != original_prediction: 
            found_counterfactuals.append(sent) 

    return found_counterfactuals, found_prototypes



def fill_all_masks_rnd_sampling(test_sentence, masked_sentence, pipeline, model_m, tokenizer, device, num_samples=15, temperature=1.0, top_k=5):
    pipeline.device = device
    original_prediction = get_mlm_prediction(test_sentence, model_m, tokenizer, device=device)
    filled_sentences = []
    
    for _ in range(num_samples):
        current_sent = masked_sentence
        total_score = 0
        
        for mask_num in range(masked_sentence.count("<mask>")):
            predictions = pipeline(current_sent, top_k=top_k)
            if not isinstance(predictions, list): predictions = [predictions]
            if isinstance(predictions[0], dict): predictions = [predictions]
            
            top_predictions = predictions[0][:top_k]
            tokens = [pred['token_str'] for pred in top_predictions]
            scores = [pred['score'] for pred in top_predictions]
            
            logits = np.array(scores) / temperature
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            if temperature == 0:
                sampled_idx = np.argmax(probs)
            else:
                sampled_idx = np.random.choice(len(probs), p=probs)
            
            token = tokens[sampled_idx]
            token_score = scores[sampled_idx]
            
            current_sent = current_sent.replace("<mask>", token, 1)
            total_score += token_score
        
        filled_sentences.append((current_sent, total_score))
    
    found_counterfactuals = []
    found_prototypes = []
    
    for sent, _ in filled_sentences:
        found_prototypes.append(sent)
        cf_candidate = sent.replace('[POSITIVE]', '').replace('[NEGATIVE]', '')
        new_prediction = get_mlm_prediction(cf_candidate, model_m, tokenizer, device=device)
        if new_prediction != original_prediction:
            found_counterfactuals.append(sent)
    
    return found_counterfactuals, found_prototypes


def batched_fill_all_masks_beam(test_sentences, masked_sentences, pipeline, model_m, tokenizer, device, beam_size=15, batch_size=32):
    """
    Processes a batch of sentences simultaneously to utilize GPU parallelization.
    """
    pipeline.device = device
    
    # 1. Initialize beams for each sentence in the batch
    # beams[i] is a list of tuples: (current_sentence_string, cumulative_score)
    beams = [[(masked_sentences[i], 0.0)] for i in range(len(masked_sentences))]
    
    # 2. Find the maximum number of masks across all sentences in this batch
    max_masks = max(sent.count("<mask>") for sent in masked_sentences)
    
    # 3. Iteratively fill masks for the whole batch
    for step in range(max_masks):
        flat_candidates = []
        candidate_indices = [] # Keeps track of which original sentence a candidate belongs to
        
        # Collect all candidates across all sentences that still need processing
        for i, beam in enumerate(beams):
            for sent, score in beam:
                if "<mask>" in sent:
                    tokens = tokenizer.encode(sent, truncation=True, max_length=512, add_special_tokens=False)
                    clean_sent = tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
                    flat_candidates.append((clean_sent, score))
                    candidate_indices.append(i)
        
        if not flat_candidates:
            break # All masks in all sentences are filled
            
        query_sentences = [c[0] for c in flat_candidates]
        all_predictions = []
        
        # Run pipeline in chunks to avoid GPU Out-Of-Memory errors
        for j in range(0, len(query_sentences), batch_size):
            chunk = query_sentences[j:j+batch_size]
            chunk_preds = pipeline(chunk, top_k=beam_size, batch_size=len(chunk))
            
            # Handle HF pipeline output formatting (it varies based on inputs)
            if isinstance(chunk_preds, dict): 
                chunk_preds = [chunk_preds]
            if len(chunk) == 1 and isinstance(chunk_preds, list) and isinstance(chunk_preds[0], dict):
                chunk_preds = [chunk_preds]
                
            all_predictions.extend(chunk_preds)
            
        # Re-assemble new beams
        new_beams = [[] for _ in range(len(masked_sentences))]
        
        for idx, (original_sent, base_score) in enumerate(flat_candidates):
            orig_i = candidate_indices[idx]
            preds = all_predictions[idx]
            
            # If a sentence has multiple masks, pipeline returns a list of lists.
            if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], list):
                preds = preds[0]
            elif isinstance(preds, dict):
                preds = [preds]
                
            top_tokens = preds[:beam_size]
            for pred in top_tokens:
                token = pred.get('token_str', pred.get('word', ''))
                token_score = pred.get('score', 0.0)
                
                # Replace only the first <mask>
                new_sent = original_sent.replace("<mask>", token, 1)
                new_beams[orig_i].append((new_sent, base_score + token_score))
                
        # Update the main beams, sorting and keeping top beam_size
        for i in range(len(beams)):
            if new_beams[i]: 
                # Carry over candidates that are already finished (no masks left)
                finished_candidates = [c for c in beams[i] if "<mask>" not in c[0]]
                combined = finished_candidates + new_beams[i]
                beams[i] = sorted(combined, key=lambda x: -x[1])[:beam_size]

    # 4. Evaluate Counterfactuals
    batch_found_cfs = []
    batch_found_protos = []
    
    for i in range(len(test_sentences)):
        found_cfs = []
        found_protos = []
        
        # Get original prediction 
        orig_pred = get_mlm_prediction(test_sentences[i], model_m, tokenizer, device=device)
        
        candidates = [c[0] for c in beams[i]]
        for cand_sent in candidates:
            found_protos.append(cand_sent)
            cf_candidate = cand_sent.replace('[POSITIVE]', '').replace('[NEGATIVE]', '')
            new_pred = get_mlm_prediction(cf_candidate, model_m, tokenizer, device=device)
            
            if new_pred != orig_pred:
                found_cfs.append(cand_sent)
                
        batch_found_cfs.append(found_cfs)
        batch_found_protos.append(found_protos)
        
    return batch_found_cfs, batch_found_protos