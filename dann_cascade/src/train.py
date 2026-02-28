import os
import wandb
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight
from accelerate import Accelerator
from transformers import (
    RobertaTokenizer,
    AutoConfig,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Building the DANN model

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
            nn.Dropout(0.15),
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, num_domains)
        )
        
        self.num_domains = num_domains

    def forward(self, input_ids=None, attention_mask=None, labels=None, 
                domain_labels=None, lambda_domain=1, **kwargs):
        
        # Forward
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
                weight=self.class_weights.to(outputs.logits.device)  # The classes have weights to prevent imbalance 
                if self.class_weights is not None else None
            )
            label_loss = loss_fct(outputs.logits, labels)

        # starting calculating total loss (label loss + domain loss)
        total_loss = label_loss

        # Extract CLS embedding for domain classification
        hidden_states = outputs.hidden_states[-1]
        cls_embedding = hidden_states[:, 0, :]
        
        # Calculating domain logits and loss
        domain_logits = self.domain_classifier(cls_embedding)
        
        if domain_labels is not None:
            domain_labels = torch.clamp(domain_labels, 0, self.num_domains - 1)
            domain_loss_fn = nn.CrossEntropyLoss()
            domain_loss = domain_loss_fn(domain_logits, domain_labels)
            total_loss = total_loss + (lambda_domain * domain_loss)
        
        #Foward return: total_loss, label_logits (logits), domain_logits, label_loss, domain_loss
        return {
            'loss': total_loss,                                       
            'logits': outputs.logits,
            'domain_logits': domain_logits,
            'label_loss': label_loss if labels is not None else None,
            'domain_loss': domain_loss if domain_labels is not None else None
        }

class DANNDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        # To mantain domain_labels we separate them at fist.
        domain_labels = None
        if 'domain_labels' in features[0]:
            domain_labels = [f.pop('domain_labels') for f in features]
        
        # Normal padding (Normal padding cannot be done if we have domain_labels inside)
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors='pt',
            return_attention_mask = True,
        )
        
        # Now we add back domain_labels
        if domain_labels is not None:
            batch['domain_labels'] = torch.tensor(domain_labels)
            
        return batch

class CodeBERTTrainer:
    def __init__(self, task_subset='A', max_length=512, model_name="microsoft/codebert-base"):
        self.task_subset = task_subset
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.num_labels = None
        self.num_domains = None
        
    def load_and_prepare_data(self, train_path, val_path, binary = 0):
        # We load the dataset from parquet files
        logger.info(f"Loading dataset subset {self.task_subset}...")

        try:
            # Perfoming calculations like number of training and validation samples, number of labels, number of domains, etc.
            train_data = Dataset.from_parquet(train_path)
            val_data = Dataset.from_parquet(val_path)
            
            logger.info(f"Loaded {len(train_data)} training samples")
            logger.info(f"Loaded {len(val_data)} validation samples")
            
            train_df = train_data.to_pandas()
            val_df = val_data.to_pandas()
            
            logger.info(f"Dataset (training) columns: {train_df.columns.tolist()}")
            logger.info(f"Sample data:\n{train_df.head()}")

            logger.info(f"Dataset columns (validation): {val_df.columns.tolist()}")
            logger.info(f"Sample data:\n{val_df.head()}")
            
            if 'code' not in train_df.columns or 'label' not in train_df.columns or 'language' not in train_df.columns:
                raise ValueError("Train Dataset must contain 'code' and 'label' columns and 'language' columns")
            
            
            if 'code' not in val_df.columns or 'label' not in val_df.columns or 'language' not in val_df.columns:
                raise ValueError("Validation Dataset must contain 'code' and 'label' columns and 'language' columns")
            
            # Dictionary to map programming languages to numerical domain labels
            language_mapping = {'Python': 0, 'C++': 1, 'Java': 2, 'JavaScript': 3, 'C#': 4, 'PHP': 5, 'Go': 6, 'C':7}
            
            train_df['language_encoded'] = train_df['language'].map(language_mapping)
            val_df['language_encoded'] = val_df['language'].map(language_mapping)

            train_df = train_df.dropna(subset=['code', 'label', 'language_encoded'])
            val_df = val_df.dropna(subset=['code', 'label', 'language_encoded'])

            train_df['label'] = train_df['label'].astype(int)
            train_df['language_encoded'] = train_df['language_encoded'].astype(int)
            self.num_labels = train_df['label'].nunique()
            self.num_domains = train_df['language_encoded'].nunique()

            if binary == 0:
                logger.info("Number of labels (Binario): 2")
                self.num_labels = 2

            else:
                logger.info(f"Number of labels (multiclase): {self.num_labels - 1}")
                self.num_labels = self.num_labels - 1  # Because we are removing the human class in multiclass classification

            val_df['label'] = val_df['label'].astype(int)
            val_df['language_encoded'] = val_df['language_encoded'].astype(int)

            #Performing more statistical analysis

            logger.info(f'Domain label range: {train_df["language_encoded"].min()} to {train_df["language_encoded"].max()}')
            logger.info(f'Domain label distribution:\n{train_df["language_encoded"].value_counts().sort_index()}')
            
            logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
            
            # Shuffle datasets. Not statification.
            train_df_shuffled = train_df.sample(frac=1, random_state = 42).reset_index(drop=False)
            val_df_shuffled = val_df.sample(frac=1, random_state = 42).reset_index(drop=False)


            return train_df_shuffled, val_df_shuffled


        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

        
    def initialize_model_and_tokenizer(self):
        logger.info(f"Initializing {self.model_name} model and tokenizer...")
        
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )

        self.model = DANNCodeBERT.from_pretrained(self.model_name, config=config, num_domains = self.num_domains, class_weights = None)
        
        logger.info(f"Model initialized with {self.num_labels} labels")
        logger.info(f"Model initialized with {self.num_domains} domains (AKA programing languages)")
    
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['code'],
            truncation=True,
            max_length=self.max_length, 
            padding = False,                # We will use dinamic padding in the data collator
            return_attention_mask = False
        )

    def transformar_labels (self, ejemplo, binary = 0):
        
        if binary == 0:
            if ejemplo['label'] == 0:
                ejemplo['label'] = 0
            else:
                ejemplo['label'] = 1

        if binary == 1:
            ejemplo['label'] = ejemplo['label'] - 1

        return ejemplo
        
    def prepare_datasets(self, train_df, val_df, binary = 0):

        logger.info("Preparing datasets for training...")
        
        train_dataset = Dataset.from_pandas(train_df[['code', 'label', 'language_encoded']])
        val_dataset = Dataset.from_pandas(val_df[['code', 'label', 'language_encoded']])

        if binary == 0:

            logger.info("Preparing datasets for binary classification, changing labels to 0 and 1")
            train_dataset = train_dataset.map(lambda x: self.transformar_labels(x, binary))
            val_dataset = val_dataset.map(lambda x: self.transformar_labels(x, binary))
        
        elif binary == 1:

            logger.info("Preparing datasets for multiclass classification without human class, changing labels accordingly")
            train_dataset = train_dataset.filter(lambda x: x['label'] != 0)
            val_dataset = val_dataset.filter(lambda x: x['label'] != 0)

            train_dataset = train_dataset.map(lambda x: self.transformar_labels(x, binary))
            val_dataset = val_dataset.map(lambda x: self.transformar_labels(x, binary))
        
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code']
        )
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code']
        )
        
        train_dataset = train_dataset.rename_column('label', 'labels')
        train_dataset = train_dataset.rename_column('language_encoded', 'domain_labels')
        val_dataset = val_dataset.rename_column('label', 'labels')
        val_dataset = val_dataset.rename_column('language_encoded', 'domain_labels')
                
        logger.info(f"Train_dataset columns: {train_dataset.column_names}")
        logger.info(f"Val_dataset columns: {val_dataset.column_names}")


        return train_dataset, val_dataset
    
    def compute_metrics(self, labels, predictions):
        predictions, labels = predictions, labels
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')

        # Perform metrics and register in wandb

        wandb.log({
            "eval/accuracy": accuracy,
            "eval/f1": f1,
            "eval/precision": precision, 
            "eval/recall": recall,
        })
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, run_name, train_dataset, val_dataset, num_epochs=5, batch_size=16, learning_rate=2e-5, lambda_domain = 0.5, gradient_accumulation = 2, eval_step = 800, training_step = 100):
        
        logger.info("Starting training...")
        labels = train_dataset['labels']
        class_weights = compute_class_weight(
            'balanced',
            classes = np.unique(labels),
            y = labels
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        logger.info(f"Los pesos de las clases son {class_weights_tensor}")

        # Loading data collator, dataloaders, optimizer, scheduler, accelerator

        accelerator = Accelerator()

        data_collator = DANNDataCollator(tokenizer=self.tokenizer)
        
        train_dataloader = DataLoader(
            train_dataset, shuffle = True, batch_size = batch_size, collate_fn = data_collator
        )
        val_dataloader = DataLoader(
            val_dataset, shuffle = True, batch_size = batch_size, collate_fn = data_collator
        )

        optimizer = AdamW(self.model.parameters(), lr = learning_rate, weight_decay = 0.01)

        train_dl, val_dl, self.model, optimizer = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, optimizer
        )

        # Movemos a la GPU class weights

        self.class_weights = class_weights_tensor.to(accelerator.device)

        num_epochs = num_epochs

        num_training_steps = int( num_epochs * len(train_dl) / gradient_accumulation)  # Gradient accumulation modifies the number of steps (this helps to train with larger effective batch sizes)
        lr_scheduler = get_scheduler(
            'linear',
            optimizer = optimizer,
            num_warmup_steps = 800,
            num_training_steps = num_training_steps,
        )

        progress_bar = tqdm(range(num_training_steps))

        # Early_stopping (Not using callbacks)

        best_f1 = 0
        patience = 10
        patience_counter = 0

        step_counter = 0
        effective_step = 0
        eval_step = eval_step 
        training_step = training_step 

        breaking = False

        val_metrics = {'accuracy': 0, 'f1': 0}

        self.model.train()
        for epoch in range(num_epochs):

            logger.info(f'ÉPOCA NÚMERO: {epoch + 1}')

            for batch in train_dl:

                step_counter += 1

                outputs = self.model(**batch, lambda_domain = lambda_domain)
                loss = outputs['loss'] # Pérdida total
                label_loss = outputs['label_loss'] # Pérdida del clasificador principal
                domain_loss = outputs['domain_loss'] # Pérdida del calsificador secundario
                #Normalizamos la pérdida
                normalized_loss = loss / gradient_accumulation
                accelerator.backward(normalized_loss)
 
                #Effective step because of gradient accumulation
                if step_counter % gradient_accumulation == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    effective_step += 1

                if effective_step % training_step == 0:

                    wandb.log({
                        'train/loss': loss.item(),
                        'train/label_loss': label_loss.item(),
                        'train/domain_loss':domain_loss.item(),
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'train/epoch': (effective_step/num_training_steps) * num_epochs,
                        'train/step': effective_step,
                    })

                if effective_step % eval_step == 0:

                    # Validación
                    y_true, y_pred = self.evaluate_model(val_dl, accelerator.device)
                    val_metrics = self.compute_metrics(y_true, y_pred)

                    # 2. Métricas de evaluación (en cada eval_step)
                    wandb.log({
                        "eval/epoch": (effective_step/num_training_steps) * num_epochs,
                        "eval/step": effective_step
                    })

                    #Implementación Early_Stopping

                    if val_metrics['f1'] > best_f1:
                        best_f1 = val_metrics['f1']
                        patience_counter = 0
                        torch.save(self.model.state_dict(), 'best_model.pth')

                        # Registramos un artefacto

                        artifact = wandb.Artifact(
                            name = f"{run_name}-{best_f1:.4f}",
                            type = "model"
                        )

                        artifact.add_file('best_model.pth')
                        wandb.log_artifact(artifact)

                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info('Se activo el early_stopping')
                            breaking = True
                            break

            logger.info(f"Acurracy_val: {val_metrics['accuracy']:.4f}, f1_score_val: {val_metrics['f1']:.4f}")


            if breaking:
                break




        self.evaluate_model(val_dl, accelerator.device, show_classification_report= True)
        
        return best_f1
        



    def evaluate_model(self, dataloader, device, show_classification_report = False):
        logger.info("Evaluating model...")
        
        # Evaluation of the model

        self.model.eval()
        all_label_loss = 0
        all_domain_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:

                # Calculating label and domain losses, and then doing label predictions to calculate metrics
                outputs = self.model(**batch)

                loss = outputs['loss']      
                domain_loss = outputs['domain_loss']
                logits = outputs['logits']  

                predictions = torch.argmax(logits, dim = 1)
                all_label_loss = all_label_loss + loss.item()
                all_domain_loss = all_domain_loss + domain_loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
            all_label_loss = all_label_loss / len(dataloader)
            all_domain_loss = all_domain_loss / len(dataloader)

            # 2. Métricas de evaluación (en cada eval_step)
            wandb.log({
                "eval/label_loss": all_label_loss,
                "eval/domain_loss": all_domain_loss,
            })


        if show_classification_report:
            
            logger.info('Classification Report:')
            print(classification_report(all_labels, all_predictions))
                    

        return all_labels, all_predictions
    
    def run_full_pipeline(self, train_path, val_path, run_name, binary = 0, num_epochs=3, batch_size=16, learning_rate=2e-5, not_prepared = True, Task_B = 0):
        try:
            train_df, val_df = self.load_and_prepare_data(train_path, val_path, binary)
            
            self.initialize_model_and_tokenizer()
            
            train_dataset, val_dataset = self.prepare_datasets(train_df, val_df, binary)
            
            best_f1 = self.train(
                run_name,
                train_dataset, val_dataset, 
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            
    
            
            logger.info(f"Pipeline completed successfully!, con mejor f1: {best_f1}")
            
            
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            raise
        finally:
            if wandb.run is not None:
                wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Train CodeBERT on SemEval-2026-Task13')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training dataset (parquet file)')
    parser.add_argument('--val_path', type=str, required=True, help='Path to validation dataset (parquet file)')
    parser.add_argument('--name_run', type=str, required=True, help='Wandb run name')
    parser.add_argument('--binary', type=int, choices=[0,1], default=0, help = '0 is for binary classification (human and machine), 1 is for multiclass classification with-out human class')
    parser.add_argument('--task', choices=['A', 'B', 'C'], default='A', help='Task subset to use')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    args = parser.parse_args()
    
    trainer = CodeBERTTrainer(
        task_subset=args.task,
        max_length=args.max_length
    )
    
    # Inicializamos Wandb

    run_name = args.name_run

    wandb.init(
        project = 'CodeBERT-SemEval-2026-Task13',
        name = run_name)

    trainer.run_full_pipeline(
        train_path = args.train_path,
        val_path = args.val_path,
        run_name = run_name,
        binary = args.binary,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Cerramos Wandb

    wandb.finish()

if __name__ == "__main__":
    main()

    
    
