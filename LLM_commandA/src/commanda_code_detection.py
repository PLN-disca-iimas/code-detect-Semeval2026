# -*- coding: utf-8 -*-
"""commandA_code_detection.ipynb

# Imports
"""

import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import cohere
co = cohere.ClientV2("Poner_la_API_correspondiente")

"""# Functions"""

def command_a(prompt, model="command-a-03-2025"):
    response = co.chat(
             model=model,
             messages=[
                   {"role": "system", "content": "You have to detect BETWEEN human made codes and artificial intelligence generated."},
                   {"role": "user", "content": prompt}],
                    temperature=.3,
                    max_tokens=900)
    return str(response.message.content[0])

# Cell 7 - Binary classification by batches

def clasificarBin_batch(lista_codigos, prompt_version=3, df=None):
    """
    Clasify a LIST of codes:
    - Human-Written
    - Machine-Generated

    Return a list of labels etiquetas (one per text).
    """

    # Sample of codes as example
    bloque_codes = "\n".join([f"{i+1}. {texto}" for i, texto in enumerate(lista_codigos)])

    # ========= PROMPTS =========
    if prompt_version == 1:
        # 🟢 Zero-shot
        prompt = f"""
You will receive a programming Code.
You have to detect human made codes, label 0 means the code is human made and label 1 means is not. Return EXACTLY the LABEl to classify the code.
Valid labels:
- 0
- 1

Code:
{bloque_codes}

Now output ONLY the label,with no extra text:
"""

    elif prompt_version == 2:
        # 🟣 One-shot
        if df is not None and len(df) > 0:
            ejemplo = df.sample(1).iloc[0]
            ejemplo_code = ejemplo["code"]
            ejemplo_label = ejemplo["label"]
        else:
            ejemplo_text = "# sustituye con tu path"
            ejemplo_label = "AI"

        prompt = f"""

Example:
code: "{ejemplo_code}"
Label: {ejemplo_label}

You must classify the next code as human made or artificial intelligence generated.
Your response must be only one of this labels: 0 for human made and 1 for artificial intelligence generated.

Code:
{bloque_codes}

Now output ONLY the label with no numbers extra text:
"""

    else:
        # 🔵 Few-shot
        if df is not None and len(df) > 0:
            #ejemplos = df.sample(min(4, len(df)))
            ejemplos= df
            ejemplos_code = "\n".join(
                [f'"{row["code"]}" →code made for {row["label"]}' for _, row in ejemplos.iterrows()]
            )
        else:
            ejemplos_code = '"# sustituye con tu path" → 1'

        prompt = f"""

Examples:
{ejemplos_code}

You must classify the next code as human made or artificial intelligence generated.
Your response must be only one of this labels: 0 for human made and 1 for artificial intelligence generated.

Programming Code:
{bloque_codes}

Now output ONLY a valid label, with no extra text:
"""

    raw = command_a(prompt)
    raw_lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # ===== NORMALIZATION ONLY =====
    def normalize(label):
        t = label.lower()
        if "0" in t:
            return "0"
        if "1" in t:
            return "1"
        return "1"  # safe fallback

    # Make a list of labels
    #labels = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    labels = [normalize(l) for l in raw_lines]

    # Handling less/more labels
    if len(labels) < len(lista_codigos):
        labels += ["isnotenoughdata"] * (len(lista_codigos) - len(labels))
    elif len(labels) > len(lista_codigos):
        labels = labels[:len(lista_codigos)]

    return labels

"""# Perform the classification"""

# Load data
path1=#path to load the test subset
path2=#path for load the examples
df = pd.read_parquet(path=path1)
df_exa = pd.read_parquet(path=path2)

# Perform the binary classification
output_path = ''
# lenght of dataset and batch size
num_registros = df.shape[0]
BATCH_SIZE = 1

finBin= df
subsetBin=df

# Few-Shot (Prompt 3)
preds_3 = []
for i in tqdm(range(0, len(finBin), BATCH_SIZE), desc="Binary Few-Shot (Prompt 3)"):
    batch = finBin["code"].iloc[i:i+BATCH_SIZE].tolist()
    try:
        labels = clasificarBin_batch(batch, prompt_version=3, df=subset_balanceado_exa)
    except Exception as e:
        print(f"Error (Binary Few-Shot): {e}")
        labels = ["error"] * len(batch)
    preds_3.extend(labels)

finBin.drop('code', axis=1,inplace=True)
finBin["pred_prompt3"] = preds_3
finBin.reset_index(drop=False,inplace=True)
finBin.to_csv(output_path)

print("binary classification completed!")
finBin.head()
