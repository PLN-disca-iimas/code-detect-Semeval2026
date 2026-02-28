import os
import torch
import torch.nn as nn
import pandas as pd
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader
import argparse
import logging
import wandb
import numpy as np
from tqdm import tqdm
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Cargamos DANN

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DANNCodeBERT(RobertaForSequenceClassification):
    def __init__(self, config, num_domains=3, class_weights = None):
        super().__init__(config)

        self.class_weights = class_weights
        
        # Clasificador de dominio
        self.domain_classifier = nn.Sequential(
            GradientReversal(alpha=1.0),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_domains)
        )
        
        self.num_domains = num_domains

    def forward(self, input_ids=None, attention_mask=None, labels=None, 
                domain_labels=None, lambda_domain=0.5, **kwargs):
        
        # Forward normal
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True,
            **kwargs
        )
    
        label_loss = 0
        if labels is not None:
            labels = torch.clamp(labels, 0, self.num_labels - 1)
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(outputs.logits.device) 
                if self.class_weights is not None else None
            )
            label_loss = loss_fct(outputs.logits, labels)

        # Calcular losses combinados
        total_loss = label_loss

        # Obtener embeddings para dominio
        hidden_states = outputs.hidden_states[-1]
        cls_embedding = hidden_states[:, 0, :]
        
        # Clasificación de dominio
        domain_logits = self.domain_classifier(cls_embedding)
        
        if domain_labels is not None:
            domain_loss_fn = nn.CrossEntropyLoss()
            domain_loss = domain_loss_fn(domain_logits, domain_labels)
            total_loss = total_loss + (lambda_domain * domain_loss)
        
        return {
            'loss': total_loss,
            'logits': outputs.logits,
            'domain_logits': domain_logits,
            'label_loss': label_loss if labels is not None else None,
            'domain_loss': domain_loss if domain_labels is not None else None
        }


def load_model_and_tokenizer(train_path, binary, artifact_name, device):

    # In order to initialize the model architecture, we need to know the configuration and the model state
    # To know the configuration, we can load the data in wich the model was trained to get the number of labels and domains
    # To know the model state, we will download the weights from wandb

    logger.info(f'Loading training data from {train_path} to infer model configuration')
    train_data = Dataset.from_parquet(train_path)
    train_df = train_data.to_pandas()

    if 'code' not in train_df.columns or 'label' not in train_df.columns or 'language' not in train_df.columns:
            raise ValueError("Train Dataset must contain 'code' and 'label' columns and 'language' columns")
        
    # Dictionary to map programming languages to numerical domain labels
    language_mapping = {'Python': 0, 'C++': 1, 'Java': 2, 'JavaScript': 3, 'C#': 4, 'PHP': 5, 'Go': 6, 'C':7}
    
    train_df['language_encoded'] = train_df['language'].map(language_mapping)
    train_df = train_df.dropna(subset=['code','label','language_encoded'])

    train_df['label'] = train_df['label'].astype(int)
    train_df['language_encoded'] = train_df['language_encoded'].astype(int)

    if binary == 0:
        # Binary classification (human vs non-human)
        train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 0 else 1)

    elif binary == 1:
        # Multiclass classification excluding human class
        train_df = train_df[train_df['label'] != 0]
        train_df['label'] = train_df['label'] - 1  # Reindex labels
    
    num_labels = train_df['label'].nunique()
    num_domains = train_df['language_encoded'].nunique()

    logger.info(f'Number of labels: {num_labels}, Number of domains: {num_domains}')

    model_name = "microsoft/codebert-base"
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels = num_labels
    )

    model = DANNCodeBERT.from_pretrained(
        model_name,
        config = config,
        num_domains = num_domains, 
        class_weights = None)


    #Inicializamos wandb
    run = wandb.init(project = 'CodeBERT-SemEval-2026-Task13')

    artifact = run.use_artifact(artifact_name)

    logger.info(f'Downloading model weights from wandb artifact {artifact.name}')

    artifact_dir = artifact.download() 
    pth_file = os.path.join(artifact_dir, 'best_model.pth')
    state_dict = torch.load(pth_file, map_location=torch.device(device))
    model.load_state_dict(state_dict)


    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()
    run.finish()

    return model, tokenizer


def collate_fn(batch, tokenizer, max_length):
    codes = [item["code"] for item in batch]
    ids = [item["ID"] for item in batch]
    encodings = tokenizer(
        codes,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encodings["ids"] = ids
    return encodings


@torch.no_grad()
def predict(train_path, test_path, artifact_name, artifact_name2, output_path, max_length=512, batch_size=16, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(train_path, 0, artifact_name, device)

    # Stream parquet dataset (no memory blowup!)

    df = pd.read_parquet(test_path)

    if 'ID' not in df.columns:
        df['ID'] = range(0,len(df))

    dataset = Dataset.from_pandas(df)

    # Validate schema
    first_row = next(iter(dataset))
    if not {"ID", "code"}.issubset(first_row.keys()):
        raise ValueError("Parquet file must contain 'ID' and 'code' columns")
    

    # DataLoader for streaming batches
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda x: collate_fn(x, tokenizer, max_length)
    )

    # We begin predictions first predictions with the binary model
    logger.info("Starting predictions with binary model...")

    all_ids = []
    all_preds1 = []

    for batch in tqdm(dataloader, desc = 'Predicciones del primer modelo'):
        inputs_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids = inputs_ids, attention_mask = attention_mask)
        preds = torch.softmax(outputs['logits'], dim = -1)
        preds_labels = preds.argmax(dim=-1).cpu().numpy()
        for i, id_ in enumerate(batch['ids']):
            all_ids.append(id_)
            all_preds1.append(preds_labels[i])

    #Liberamos espacio en memoria del primer modelo.
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

     # Load second model & tokenizer
    second_model, tokenizer = load_model_and_tokenizer(train_path, 1, artifact_name2, device)
    all_preds2 = []

    for batch in tqdm(dataloader, desc = 'Predicciones del segundo modelo'):
        inputs_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = second_model(input_ids = inputs_ids, attention_mask = attention_mask)
        preds = torch.softmax(outputs['logits'], dim = -1)
        preds_labels = preds.argmax(dim=-1).cpu().numpy()

        for i, id_ in enumerate(batch['ids']):
            all_preds2.append(preds_labels[i])

    #Hacemos la predicción final combinando ambas predicciones.

    final_pred = []

    for i, (a, b) in enumerate(zip(all_preds1, all_preds2)):
        if a == 0:
            final_pred.append(0)
        else:
            final_pred.append(b + 1)

    with open(output_path, 'w') as f:
        f.write('ID,prediction\n')
        for id_, pred in zip(all_ids, final_pred):
            f.write(f'{id_},{pred}\n')
    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained CodeBERT model (streaming)")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training data in witch the model was trained")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test data in parquet file with ID and code")
    parser.add_argument("--artifact_name", type=str, required=True, help="Wandb artifact name containing the trained model weights (binary-model)")
    parser.add_argument("--artifact_name2", type = str, required = True, help = "Wandb artifact name containing the trained model weights (multiclass-model)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save predictions CSV")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu or cuda")

    args = parser.parse_args()
    predict(
        args.train_path,
        args.test_path,
        args.artifact_name,
        args.artifact_name2,
        args.output_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device
    )

