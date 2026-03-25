import numpy as np
import nltk, torch, math
import openai
from spacy.lang.en import English
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from Levenshtein import distance as levenshtein_distance
from transformers import RobertaForSequenceClassification, RobertaForMaskedLM, RobertaTokenizer, BertTokenizer, DistilBertTokenizer, BertForSequenceClassification, BertForMaskedLM, DistilBertForSequenceClassification, DistilBertForMaskedLM, Trainer, TrainingArguments, DistilBertTokenizerFast
import json
import re
from openai import OpenAI


class Metrics:
    def __init__(self, model_id="gpt2", device=None):
        nlp = English()
        self.tokenizer = nlp.tokenizer
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load GPT-2 once for perplexity
        self.ppl_tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.ppl_model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.ppl_model.eval()

    # ---------- Counterfactual metrics ----------
    @staticmethod
    def flip_rate(original_labels, new_labels):
        if len(original_labels) != len(new_labels):
            raise ValueError("Input lists must have the same length.")
        return sum(o != n for o, n in zip(original_labels, new_labels)) / len(original_labels)

    
    @staticmethod
    def probability_change(p_orig, p_cf, y_orig, y_cf):
        orig = np.array(p_orig)
        cf = np.array(p_cf)
        
        # 1. Calculate absolute differences
        diffs = np.abs(cf - orig)
        
        # 2. Create a boolean mask of ACTUAL successes (where the label flipped)
        success_mask = np.array(y_orig) != np.array(y_cf)
        
        # 3. Apply the mask to get only the valid shifts
        valid_diffs = diffs[success_mask]
        
        if len(valid_diffs) == 0:
            return 0.0
            
        return float(np.mean(valid_diffs))

    def token_distance(self, texts_orig, texts_cf, normalized=False):
        dists = []
        for o, c in zip(texts_orig, texts_cf):
            toks_o = [t.text for t in self.tokenizer(o)]
            toks_c = [t.text for t in self.tokenizer(c)]
            dist = nltk.edit_distance(toks_o, toks_c)
            if normalized:
                dist /= max(1, len(toks_o))
            dists.append(dist)
        return float(np.mean(dists))

    # ---------- Perplexity (manual) ----------
    def score_perplexity(self, sents):
        ppl_scores = []
        for text in sents:
            enc = self.ppl_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                loss = self.ppl_model(**enc, labels=enc["input_ids"]).loss
            ppl_scores.append(math.exp(loss.item()))
        return float(sum(ppl_scores) / len(ppl_scores))

    # ---------- Diversity ----------
    def diversity(self, cf_sets):
        divs = []
        for cf_list in cf_sets:
            if len(cf_list) < 2:
                continue
            pairwise = []
            for i in range(len(cf_list)):
                for j in range(i + 1, len(cf_list)):
                    d = levenshtein_distance(
                        " ".join(cf_list[i].split()), " ".join(cf_list[j].split())
                    )
                    pairwise.append(d)
            divs.append(np.mean(pairwise))
        return float(np.mean(divs)) if divs else 0.0
    



# Text Quality Metrics (GPT-based evaluation as in CEval)

'''def evaluate_text_quality(texts, model="gpt-3.5-turbo", temperature=0.2):
    """
    Evaluates text quality metrics (Grammar, Cohesiveness, Fluency)
    using a GPT model as described in Section 3.1.2.
    Returns a dict with averaged scores on scale [1–5].
    """
    prompt_template = """
    Evaluate the following text on a scale of 1–5 for:
    1. Grammar correctness
    2. Cohesiveness (logical flow)
    3. Fluency (naturalness/readability)

    Respond with JSON: {{"grammar": <score>, "cohesiveness": <score>, "fluency": <score>}}.

    Text: "{text}"
    """

    results = {"grammar": [], "cohesiveness": [], "fluency": []}
    for t in texts:
        prompt = prompt_template.format(text=t)
        response = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            scores = eval(response["choices"][0]["message"]["content"])
            for k in results:
                results[k].append(scores[k])
        except Exception:
            continue

    return {k: np.mean(v) for k, v in results.items() if v}'''



def compute_perplexity(sentences, model_name="gpt2", device="cuda"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    ppl_scores = []
    for text in sentences:
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss
        ppl_scores.append(math.exp(loss.item()))
    return {"mean_perplexity": sum(ppl_scores)/len(ppl_scores), "perplexities": ppl_scores}



# 1. Initialize the OpenAI client with API key

client = OpenAI(api_key="") # put key
def evaluate_text_quality(texts, model="gpt-4o-mini", temperature=0.2):
    """
    Evaluates text quality metrics (Grammar, Cohesiveness, Fluency)
    using an OpenAI model. Returns a dict with averaged scores on scale [1–5].
    """
    prompt_template = """
    Evaluate the following text on a scale of 1–5 for:
    1. Grammar correctness
    2. Cohesiveness (logical flow)
    3. Fluency (naturalness/readability)

    Respond with JSON: {{"grammar": <score>, "cohesiveness": <score>, "fluency": <score>}}.

    Text: "{text}"
    """

    results = {"grammar": [], "cohesiveness": [], "fluency": []}
    
    for t in texts:
        prompt = prompt_template.format(text=t)
        
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # 3. Accessing the response content (new object-attribute syntax)
            content = response.choices[0].message.content
            
            # 4. Safe JSON parsing 
            content = content.replace("```json", "").replace("```", "").strip()
            scores = json.loads(content)
            
            for k in results:
                results[k].append(scores[k])
                
        except Exception as e:
            print(f"Failed on text: '{t[:30]}...' | Error: {e}")
            continue

    return {k: np.mean(v) for k, v in results.items() if v}


























######################################miscellaneuos functions##################

def tokenize_function(examples, tokenizer):


    return tokenizer(examples['text_with_label'], 
                     padding='max_length', 
                     truncation=True, 
                     max_length=512, 
                     add_special_tokens=False) # I remove the cls/bos token "<s>" and also "<s\>" eos token





def get_prediction(text, model, return_probas=False):
    
    inputs = tokenizer(text, return_tensors="pt", max_length= 512, truncation=True, padding=True, return_token_type_ids=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits

    # Get predicted class index
    #predicted_class = logits.argmax().item()
    predicted_class = logits.max(dim=1).indices.item() # verison if bug on mac
    
    if return_probas:
        # Convert logits to probabilities using softmax
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()  # Squeeze to convert to list
        return predicted_class, probabilities
    
    return predicted_class



def get_mlm_prediction(text, model_m, tokenizer, return_probas=False):
    """
    Get classification prediction from MLM model using control tokens.
    Returns: (predicted_class, [negative_prob, positive_prob])
    """
    # Prepare masked input
    masked_input = f"{tokenizer.mask_token}{text}"
    
    # Tokenize and move to device
    inputs = tokenizer(
        masked_input,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
        return_token_type_ids=False
    ).to(device)
    
    # Get logits for [MASK] at position 0
    with torch.no_grad():
        logits = model_m(**inputs).logits[0, 0]  # Logits for first token
    
    # Get control token IDs
    positive_id = tokenizer.convert_tokens_to_ids("[POSITIVE]")
    negative_id = tokenizer.convert_tokens_to_ids("[NEGATIVE]")
    
    # Calculate probabilities for control tokens only
    control_logits = torch.tensor([logits[negative_id], logits[positive_id]])
    probabilities = F.softmax(control_logits, dim=-1).tolist()
    
    predicted_class = int(probabilities[1] > 0.5)
    
    return (predicted_class, probabilities) if return_probas else predicted_class


def shap_wrapper(texts):
    """Wrapper that properly formats outputs for SHAP"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): 
        device = torch.device("mps")
    # Ensure input is list even for single text
    if isinstance(texts, str):
        texts = [texts]
    
    # Add [MASK] prefix and process batch
    masked_texts = [f"{tokenizer.mask_token}{text}" for text in texts]

    inputs = tokenizer(masked_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        logits = model_m(**inputs).logits[:, 0]  # Position 0 logits
    
    # Convert to probabilities and return as numpy array
    probs = F.softmax(logits[:, [NEG_ID, POS_ID]], dim=-1)
    return probs.cpu().numpy()  # Shape: [batch_size, 2]





def get_gradient_saliency(text, model_m, tokenizer):
    """Returns tokens ranked by their influence on control token prediction"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): 
        device = torch.device("mps")


    # 1. Prepare inputs
    inputs = tokenizer(f"{tokenizer.mask_token}{text}", return_tensors="pt")
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
    
    return sorted(filtered, key=lambda x: -x[1])#[:10]




def mask_tokens_with_gradients(test_sentence, n_to_mask=2):
   
    # 1. Get prediction and determine control token
    predicted_class = get_mlm_prediction(test_sentence, model_m, tokenizer)

    # 2. Get gradient-based important tokens 
    salient_tokens = get_gradient_saliency(test_sentence, model_m, tokenizer)
    important_tokens = [f' {t}' for t, _ in salient_tokens] # this is formatting is done to match SHAP list format in original mask_tokens()
    print('important_tokens: ', important_tokens)
    # Prepend control token
    label = '[POSITIVE]' if predicted_class == 0 else '[NEGATIVE]'       #  <------------------------
    modified_text = f"{label}{test_sentence}"

    # Mask most important tokens first
    masked_text = modified_text
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
    
    print('masked sentence: ', masked_text)
    return masked_text




def mask_tokens_with_shap(test_sentence, n_to_mask=4):

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): 
        device = torch.device("mps")
   
    # Get SHAP values
    predicted_class = get_mlm_prediction(test_sentence, model_m, tokenizer)
    shap_values = explainer([test_sentence], silent=True)
    pred_class = np.argmax(shap_wrapper([test_sentence])[0])
    
    # Process SHAP values
    shap_tokens = shap_values.data[0]
    shap_scores = shap_values.values[0][:, pred_class] # get the tokens for the specific class (or ~ if want opposite)
    
    # Pair tokens with their SHAP values and filter
    token_score_pairs = [(t, s) for t, s in zip(shap_tokens, shap_scores) 
                        if s > 0 and t.strip() not in {'', '<mask>'}]
    
    # Sort by SHAP score (descending) while preserving original order for ties
    token_score_pairs.sort(key=lambda x: (-x[1], x[0]))
    important_tokens = [t for t, _ in token_score_pairs]
    print('important_tokens: ', important_tokens)
    # Prepend control token
    label = '[POSITIVE]' if predicted_class == 0 else '[NEGATIVE]'       #  <------------------------
    modified_text = f"{label}{test_sentence}"

    # Mask most important tokens first
    masked_text = modified_text
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
    
    print('masked sentence: ', masked_text)
    return masked_text



def fill_all_masks_beam(test_sentence, masked_sentence, pipeline, beam_size=15):

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): 
        device = torch.device("mps")

    pipeline.device = device
    original_prediction = get_mlm_prediction(test_sentence, model_m, tokenizer)
    control_token_len = 10

    filled_sentences = [(masked_sentence, 0)]  # (sentence, score)

    for _ in range(masked_sentence.count("<mask>")):
        new_filled = []
        for sent, score in filled_sentences:
            predictions = pipeline(sent, top_k=beam_size)
            if not isinstance(predictions, list):
                predictions = [predictions]
            if isinstance(predictions[0], dict):
                # Only one mask
                predictions = [predictions]

            top_tokens = predictions[0][:beam_size]

            for pred in top_tokens:
                token = pred['token_str']
                token_score = pred['score']
                new_sent = sent.replace("<mask>", token, 1)
                new_filled.append((new_sent, score + token_score))

        # Keep top beam_size
        filled_sentences = sorted(new_filled, key=lambda x: -x[1])[:beam_size]

    # Now check predictions
    found_counterfactuals = []
    found_prototypes = []

    for sent, _ in filled_sentences:
        found_prototypes.append(sent)
        cf_candidate = sent[control_token_len:]
        print('cf candidate:', cf_candidate)
        new_prediction = get_mlm_prediction(cf_candidate, model_m, tokenizer)
        if new_prediction != original_prediction:
            #print(f"Counterfactual Found: {sent}")
            #print(f"original: {original_prediction}, new: {new_prediction}")
            found_counterfactuals.append(sent)

    return found_counterfactuals, found_prototypes
