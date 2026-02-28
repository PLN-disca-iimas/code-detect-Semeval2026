import pprint
import torch
import numpy as np
import seaborn as sns
from torch_geometric.data import Data, DataLoader
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from itertools import combinations
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import gensim
from collections import Counter, defaultdict
from datasets import load_dataset
import joblib
import networkx as nx
import sys, traceback, time
import nltk
import re 
import spacy
from math import log
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import logging
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset, DatasetDict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import joblib
import time
import numpy as np
import pandas as pd
import logging
import traceback
from tqdm import tqdm
import torch
import torch.nn as nn
import networkx as nx
import scipy as sp
import scipy.sparse as sp
import gensim
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from collections import OrderedDict
import warnings
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.nn import (
    GCNConv, GATConv, TransformerConv,
    global_mean_pool, global_max_pool, global_add_pool,
    GlobalAttention, Set2Set
)

import torch
import spacy

nltk.download('stopwords')
nltk.download('punkt_tab')

import utils

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Load spaCy tokenizer ---
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        #print(validation_loss, self.min_validation_loss, self.counter)
        if validation_loss <= self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class GNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dense_hidden_dim,
        output_dim,
        dropout,
        num_layers,
        edge_attr=False,
        gnn_type='GCNConv',
        heads=1,
        task='node',
        norm_type='batchnorm',   # 'batchnorm', 'layernorm', or None
        post_mp_layers=3,        # number of layers after message passing
        pooling_type='mean'      # 'mean', 'max', 'sum', 'attention', 'set2set'
    ):
        super(GNN, self).__init__()
        self.task = task
        self.heads = heads
        self.gnn_type = gnn_type
        self.edge_attr = edge_attr
        self.dropout = dropout
        self.num_layers = num_layers
        self.norm_type = norm_type

        # First conv
        self.conv1 = self.build_conv_model(input_dim, hidden_dim, heads)
        self.norm1 = self.build_norm_layer(hidden_dim * heads)

        # Additional conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(self.build_conv_model(hidden_dim * heads, hidden_dim, heads))
            self.norms.append(self.build_norm_layer(hidden_dim * heads))

        # Global Pooling
        self.global_pool = self.get_pooling_layer(pooling_type, hidden_dim * heads)

        # Post-message-passing MLP
        dims = [hidden_dim * heads] + [dense_hidden_dim // (2 ** i) for i in range(post_mp_layers - 1)] + [output_dim]
        post_mp = []
        for i in range(len(dims) - 1):
            post_mp.append(nn.Linear(dims[i], dims[i + 1]))
            #if i < len(dims) - 2:
            #    post_mp.append(nn.ReLU())
        self.post_mp = nn.Sequential(*post_mp)

    def build_conv_model(self, input_dim, hidden_dim, heads):
        if self.gnn_type == 'GCNConv':
            return GCNConv(input_dim, hidden_dim)
        elif self.gnn_type == 'GINConv':
            return GCNConv(input_dim, hidden_dim)
        elif self.gnn_type == 'GATConv':
            return GATConv(input_dim, hidden_dim, heads=heads)
        elif self.gnn_type == 'TransformerConv':
            if self.edge_attr:
                return TransformerConv(input_dim, hidden_dim, heads=heads, edge_dim=2)
            else:
                return TransformerConv(input_dim, hidden_dim, heads=heads)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

    def build_norm_layer(self, dim):
        if self.norm_type == 'batchnorm':
            return nn.BatchNorm1d(dim)
        elif self.norm_type == 'layernrom':
            return nn.LayerNorm(dim)
        else:
            return nn.Identity()

    def get_pooling_layer(self, pooling_type, hidden_dim):
        if pooling_type == 'mean':
            return global_mean_pool
        elif pooling_type == 'max':
            return global_max_pool
        elif pooling_type == 'sum':
            return global_add_pool
        elif pooling_type == 'attention':
            gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1))
            return GlobalAttention(gate_nn)
        elif pooling_type == 'set2set':
            return Set2Set(hidden_dim, processing_steps=3)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")


    def get_graph_embedding(self, x, edge_index, edge_attr=None, batch=None):
        if self.edge_attr:
            x = self.conv1(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.num_layers):
            if self.edge_attr:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.global_pool(x, batch)
        return x

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.get_graph_embedding(x, edge_index, edge_attr, batch)
        logits = self.post_mp(x)
        return logits
     
def train_cooc(model, loader, device, optimizer, criterion):
        model.train()
        train_loss = 0.0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()
        return train_loss / len(loader)

def test_cooc(loader, model, device, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        correct += int((pred == data.y).sum())
        loss = criterion(out, data.y)
        test_loss += loss.item()

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    accuracy = correct / len(loader.dataset)
    return accuracy, f1_macro, test_loss / len(loader), all_preds, all_labels

def extract_doc_edges(text_tokenized, label, word_features, vocab, window_size):
    try:
        # Get unique words in the document
        unique_words = list(set(text_tokenized))
        unique_words = [word for word in unique_words if (word in vocab and word in word_features.keys())]  # Filter out OOV words

        # Create node features for this graph (only words in the document)
        #node_features = torch.stack([word_features[word] for word in unique_words])
        node_features = torch.stack([word_features[word] for word in unique_words if word in word_features])

        # Create a local word-to-index mapping for this graph
        local_word_to_index = {word: idx for idx, word in enumerate(unique_words)}

        # Calculate word frequencies in the document
        word_freq = Counter(text_tokenized)

        # Create word-word edges (co-occurrence within a window size) and calculate PMI
        word_word_edges = set()  # Use a set to avoid duplicate edges
        co_occurrence_matrix = defaultdict(int)
        total_pairs = 0

        for i in range(len(text_tokenized)):
            window = text_tokenized[i:i + window_size + 1]
            for word1, word2 in combinations(window, 2):
                if word1 in local_word_to_index and word2 in local_word_to_index:
                    word1_id = local_word_to_index[word1]
                    word2_id = local_word_to_index[word2]
                    word_word_edges.add((word1_id, word2_id))  # Add edge as a tuple to the set
                    co_occurrence_matrix[(word1, word2)] += 1
                    total_pairs += 1

        # Calculate PMI for each edge
        pmi_matrix = calculate_pmi(co_occurrence_matrix, word_freq, total_pairs)

        # Add PMI and frequency as edge attributes
        edge_attr_pmi = []
        edge_attr_freq = []
        edges = []
        for (word1, word2), pmi in pmi_matrix.items():
            word1_id = local_word_to_index[word1]
            word2_id = local_word_to_index[word2]
            edges.append((word1_id, word2_id))  # Add edge to the list
            edge_attr_pmi.append(pmi)
            edge_attr_freq.append(word_freq[word1] + word_freq[word2])  # Sum of frequencies of both words

        # Combine all edges and edge attributes
        edges = torch.tensor(edges, dtype=torch.long).t()  # Shape: [2, num_edges]
        edge_attr = torch.tensor([edge_attr_pmi, edge_attr_freq], dtype=torch.float).t()  # Shape: [num_edges, 2]

        # Debug: Check shapes
        #print(f"edges shape: {edges.shape}, edge_attr shape: {edge_attr.shape}")

        # Normalize edge attributes
        edge_attr = min_max_normalize(edge_attr)

        # Create the PyG Data object
        data = Data(x=node_features, edge_index=edges, edge_attr=edge_attr, y=label, unique_words=unique_words)
        return data
    except Exception as e:
        print('Error: %s', str(e))

def get_word_embeddings(texts_set, texts_tokenized, tokenizer, language_model, vocab, device, set_corpus='train', not_found_tokens='avg', embed_dim=128):
    doc_words_embeddings_dict = {}

    for idx, text in enumerate(tqdm(texts_set, desc=f"Extracting {set_corpus} word embeddings")):
        doc_tokens_freq = defaultdict(int)
        doc_words_embeddings_dict[idx] = {'tokens': {}, 'not_found_tokens': []}

        try:
            encoded_text = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
            encoded_text.to(device)

            with torch.no_grad():
                outputs_model = language_model(**encoded_text, output_hidden_states=True)
            last_hidden_state = outputs_model.hidden_states[-1]

            # Store embeddings before reduction
            token_embeddings = []
            token_list = []

            for i in range(len(last_hidden_state)):
                raw_tokens = [tokenizer.decode([token_id]) for token_id in encoded_text['input_ids'][i]]
                for token, embedding in zip(raw_tokens, last_hidden_state[i]):
                    token = str(token).strip()
                    if token not in vocab:
                        continue
                    doc_tokens_freq[token] += 1
                    token_embeddings.append(embedding.detach().cpu())  # FIX: Detach tensor
                    token_list.append(token)

            if token_embeddings:
                # Convert list to tensor
                token_embeddings_tensor = torch.stack(token_embeddings)

                # Assign reduced embeddings back
                for token, reduced_emb in zip(token_list, token_embeddings_tensor):
                    if token not in doc_words_embeddings_dict[idx]['tokens']:
                        doc_words_embeddings_dict[idx]['tokens'][token] = reduced_emb
                    else:
                        doc_words_embeddings_dict[idx]['tokens'][token] = np.add.reduce([
                            doc_words_embeddings_dict[idx]['tokens'][token], reduced_emb.detach().cpu().numpy()  
                        ])

            # Normalize embeddings based on frequency
            for token, freq in doc_tokens_freq.items():
                doc_words_embeddings_dict[idx]['tokens'][token] = torch.tensor(
                    np.divide(doc_words_embeddings_dict[idx]['tokens'][token], freq).tolist(), dtype=torch.float
                )

        except Exception as e:
            print('Error:', str(e))

    # Handling missing tokens
    for idx, doc_tokens in enumerate(tqdm(texts_tokenized)): 
        try:
            merge_tokens = list(set(vocab) & (set(doc_tokens) - set(doc_words_embeddings_dict[idx]['tokens'].keys())))
            for token in merge_tokens:
                not_found_tokens_dict = {}

                if not_found_tokens == 'ones':
                    doc_words_embeddings_dict[idx]['tokens'][token] = torch.ones(embed_dim, dtype=torch.float)
                elif not_found_tokens == 'zeros':
                    doc_words_embeddings_dict[idx]['tokens'][token] = torch.zeros(embed_dim, dtype=torch.float)
                else:  # avg
                    node_tokens = tokenizer.encode_plus(token, return_tensors="pt", padding=True, truncation=True)
                    raw_tokens = [tokenizer.decode([token_id]) for token_id in node_tokens['input_ids'][0]]
                    not_found_tokens_dict[token] = raw_tokens[1:-1]
                    
                    avg_emb, emb_list = [], []
                    for token, subtokens in not_found_tokens_dict.items():
                        if token in doc_words_embeddings_dict[idx]['tokens']:
                            continue
                        for subtoken in subtokens:
                            if subtoken in doc_words_embeddings_dict[idx]['tokens']:
                                emb_list.append(doc_words_embeddings_dict[idx]['tokens'][subtoken].detach().cpu())  # FIX: Detach tensor
                    
                    if len(emb_list) == 0:
                        avg_emb = torch.zeros(embed_dim, dtype=torch.float)
                    else:
                        avg_emb = torch.mean(torch.stack(emb_list), axis=0)
                    
                    doc_words_embeddings_dict[idx]['tokens'][token] = avg_emb
        except Exception as e:
            print('Error:', str(e))

    return doc_words_embeddings_dict

def normalize_text(texts, tokenize_pattern, special_chars=False, stop_words=False, set='train'):    
    all_texts_norm = []
    tokenized_corpus = []
    for text in tqdm(texts, desc=f"Normalizing {set} corpus"):
        text_norm = utils.text_normalize(text, special_chars, stop_words) 
        all_texts_norm.append(text_norm)
        #tokenized_corpus.append(tokenize_pattern(text))
    
    docs_nlp_spacy = nlp_pipeline(texts)
    for doc in docs_nlp_spacy:
        tokens = []
        for token in doc:
            if stop_words and token.is_stop:
                continue
            if special_chars and token.is_punct:
                continue
            tokens.append(token.text) # text, lemma_, pos_
        tokenized_corpus.append(tokens)
    return all_texts_norm, tokenized_corpus, docs_nlp_spacy

def regex_tokenizer(text):
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

def create_vocab(docs_nlp, stop_words=False, special_chars=False, min_df=1, max_features=5000):
    vocab = set()
    word_freq = Counter()
    for doc in docs_nlp:
        tokens = []
        for token in doc:
            if stop_words and token.is_stop:
                continue
            if special_chars and token.is_punct:
                continue
            vocab.add(token.text)
            word_freq[token.text] += 1
            tokens.append(token.text) # text, lemma_, pos_
    vocab = list(vocab)
    filtered_vocab = [word for word in vocab if word_freq[word] >= min_df]
    filtered_vocab = sorted(filtered_vocab, key=lambda w: word_freq[w], reverse=True)[:max_features]
    vocab = filtered_vocab
    word_to_index = {w: i for i, w in enumerate(vocab)}
    return vocab, word_to_index

def calculate_pmi(co_occurrence_matrix, word_freq, total_pairs):
    pmi_matrix = {}
    for (word1, word2), count in co_occurrence_matrix.items():
        pmi = log((count * total_pairs) / (word_freq[word1] * word_freq[word2]))
        pmi_matrix[(word1, word2)] = pmi
    return pmi_matrix

def min_max_normalize(tensor):
    min_vals = tensor.min(dim=0).values  # Min values for each feature
    max_vals = tensor.max(dim=0).values  # Max values for each feature
    edge_attr_normalized = (tensor - min_vals) / (max_vals - min_vals + 1e-8)  # Add small epsilon to avoid division by zero
    return edge_attr_normalized

def nlp_pipeline(docs: list):
    doc_lst = []
    for nlp_doc in tqdm(nlp.pipe(docs, batch_size=64, n_process=4), total=len(docs), desc="nlp_spacy_docs"):
        doc_lst.append(nlp_doc)
    return doc_lst

def balance_df(df):
    if 'source' not in df.columns or 'label' not in df.columns:
        raise ValueError("DataFrame must contain 'source' and 'label' columns")

    balanced_parts = []
    for domain, domain_df in df.groupby("source"):
        label_counts = domain_df['label'].value_counts()
        if len(label_counts) < 2:
            balanced_parts.append(domain_df)
            continue

        min_count = min(label_counts[0], label_counts[1])
        df_0 = domain_df[domain_df['label'] == 0].sample(min_count, random_state=42)
        df_1 = domain_df[domain_df['label'] == 1].sample(min_count, random_state=42)
        balanced_parts.append(pd.concat([df_0, df_1]))

    return pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

def main(config):    
    
    # ruta del directorio de salida (modificar para cambiar directorio)
    output_dir = f'{utils.EXTERNAL_DISK_PATH}cooc_graph_code'
    
    # nombre del archivo donde se almacena la data: modelo, features, etc
    file_name_data = f"cooc_data_{config['dataset_name']}_{config['cut_off_dataset']}perc"

    nfi_dir = config["llm_name"].split("/")[1] # nfi -> llm

    device = torch.device(f"cuda:{config['cuda_num']}" if torch.cuda.is_available() else "cpu")
    pprint.pprint(config)


    if config['build_graph'] == True:
        start = time.time()

        # *** Read datasets
        ds = load_dataset("DaniilOr/SemEval-2026-Task13", "A")

        print(ds)            
        print(ds.keys())       

        train_set = ds["train"].to_pandas()
        val_set = ds["validation"].to_pandas()
        test_set = ds["test"].to_pandas()
        print(train_set.head())

        train_set = utils.stratified_sample(train_set, config['perc_dataset'], seed=42)
        val_set   = utils.stratified_sample(val_set, config['perc_dataset'], seed=42)
        #test_set  = utils.stratified_sample(test_set, FRAC, SEED)

        print(train_set['label'].value_counts(normalize=False))
        print(val_set['label'].value_counts(normalize=False))
        print(test_set['label'].value_counts(normalize=False))

        train_texts = list(train_set['code'])[:]
        val_texts = list(val_set['code'])[:]
        test_texts = list(test_set['code'])[:]

        # Labels (binary classification: 0 or 1)
        train_labels = list(train_set['label'])[:]
        val_labels = list(val_set['label'])[:]
        test_labels = list(test_set['label'])[:]

        # Normalize and Tokenize the corpus 
        tokenize_pattern = regex_tokenizer
        train_texts_norm, train_texts_tokenized, docs_nlp = normalize_text(train_texts, tokenize_pattern, special_chars=config['special_chars'], stop_words=config['stop_words'], set='train')
        val_texts_norm, val_texts_tokenized, docs_nlp = normalize_text(val_texts, tokenize_pattern, special_chars=config['special_chars'], stop_words=config['stop_words'], set='val')
        test_texts_norm, test_texts_tokenized, docs_nlp = normalize_text(test_texts, tokenize_pattern, special_chars=config['special_chars'], stop_words=config['stop_words'], set='test')
        
        # create a vocabulary
        vocab, word_to_index = create_vocab(docs_nlp, stop_words=config['stop_words'], special_chars=config['special_chars'], min_df=config['min_df'], max_features=config['max_features'])
        
        print("not_found_tokens approach: ", config['not_found_tokens'])
        print(f'vocab len: ', len(vocab))

        # Init Pretrained-LLM        
        tokenizer = AutoTokenizer.from_pretrained(config['llm_name'], model_max_length=512)
        language_model = AutoModel.from_pretrained(config['llm_name'], output_hidden_states=True).to(device)

        # Extract word/node embeddings
        train_words_emb = get_word_embeddings(train_texts_norm, train_texts_tokenized, tokenizer, language_model, vocab, device, set_corpus='train', not_found_tokens=config['not_found_tokens'], embed_dim=config['embed_dim'])
        val_words_emb = get_word_embeddings(val_texts_norm, val_texts_tokenized, tokenizer, language_model, vocab, device, set_corpus='val', not_found_tokens=config['not_found_tokens'], embed_dim=config['embed_dim'])
        test_words_emb = get_word_embeddings(test_texts_norm, test_texts_tokenized, tokenizer, language_model, vocab, device, set_corpus='test', not_found_tokens=config['not_found_tokens'], embed_dim=config['embed_dim'])
            
        # Extract doc edges (save tokens for each doc) 
        train_data, val_data, test_data = [], [], [] 
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(train_texts_tokenized, desc="Extracting doc train edges"), train_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, train_words_emb[idx]['tokens'], vocab, config['window_size'])
            if doc_edges:
                train_data.append(doc_edges)
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(val_texts_tokenized, desc="Extracting doc val edges"), val_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, val_words_emb[idx]['tokens'], vocab, config['window_size'])
            if doc_edges:
                val_data.append(doc_edges)
        for idx, (text_tokenized, label) in enumerate(zip(tqdm(test_texts_tokenized, desc="Extracting doc test edges"), test_labels)):
            doc_edges = extract_doc_edges(text_tokenized, label, test_words_emb[idx]['tokens'], vocab, config['window_size'])
            if doc_edges:
                test_data.append(doc_edges)

        # *** Save data
        all_data = [train_data, val_data, test_data]
        data_obj = {
            "vocab": vocab,
            "all_data": all_data,
            "word_to_index": word_to_index,
            "time_to_build_graph": time.time() - start,
            "config": config,
        }
        utils.save_data(data_obj, file_name_data, path=f'{output_dir}/{nfi_dir}/', format_file='.pkl', compress=False)
    else:
        data_obj = utils.load_data(file_name_data, path=f'{output_dir}/{nfi_dir}/', format_file='.pkl', compress=False)
        train_data = data_obj['all_data'][0]
        val_data = data_obj['all_data'][1]
        test_data = data_obj['all_data'][2]
        vocab = data_obj['vocab']
        word_to_index = data_obj['word_to_index']


    print("vocab: ", len(vocab))
    print("train_data: ", len(train_data))
    print("val_data: ", len(val_data))
    print("test_data: ", len(test_data))

 
    if config['graph_direction'] == 'undirected':
        for data in train_data + val_data + test_data:
            # Add reverse edges
            data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
            # Add reverse edge attributes
            if config['add_edge_attr']:
                data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
            else:
                del data.edge_attr  # Remove edge attributes if not needed

    # Create DataLoader for train, validation, and test partitions
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    print(train_data[0])
    for batch in train_loader:
        print(batch)
        break

    # Initialize the model
    input_dim = train_data[0].x.shape[1]
    utils.set_random_seed(42)
    model = GNN(input_dim, 
                config['hidden_dim'], 
                config['dense_hidden_dim'], 
                config['output_dim'], 
                config['dropout'], 
                config['num_layers'], 
                config['add_edge_attr'], 
                gnn_type=config['gnn_type'], 
                heads=config['heads'], 
                task='graph',
                norm_type=config['norm_type'],  
                post_mp_layers=config['post_mp_layers'],       
                pooling_type=config['pooling_type']   
            )

        
    model = model.to(device)
    print(model)

    # Training loop (example)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learnin_rate'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=config['patience'], min_delta=0)

    logger.info("Init GNN training!")
    best_test_acc_score = 0
    best_test_f1_score = 0
    stop_epoch = 0
    for epoch in range(1, config['epochs']):
        loss_train = train_cooc(model, train_loader, device, optimizer, criterion)
        val_acc, val_f1_macro, val_loss, preds_val, labels_val = test_cooc(val_loader, model, device, criterion)

        if epoch % 1 == 0:
            test_acc, test_f1_macro, test_loss, _, _ = test_cooc(test_loader, model, device, criterion)
            print(f'Ep {epoch + 1}, Loss Val: {val_loss:.4f}, Loss Test: {test_loss:.4f}, '
            f'Val Acc: {val_acc:.4f}, Val F1s: {val_f1_macro:.4f}, Test Acc: {test_acc:.4f}, Test F1s: {test_f1_macro:.4f}')
        else:
            print(f'Ep {epoch + 1}, Loss Val: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1s: {val_f1_macro:.4f}')

        if test_acc > best_test_acc_score:
            best_test_acc_score = test_acc
        if test_f1_macro > best_test_f1_score:
            best_test_f1_score = test_f1_macro

        stop_epoch = epoch
        
        if early_stopper.early_stop(val_loss):
            print('Early stopping fue to not improvement!')
            break
        
        torch.cuda.empty_cache()

    logger.info("Done GNN training!")

    # Final evaluation on the test set
    test_acc, test_f1_macro, test_loss, preds_test, labels_test = test_cooc(test_loader, model, device, criterion)
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1Score: {test_f1_macro:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    
    print("preds_test:  ", sorted(Counter(preds_test).items()))
    print("labels_test: ", sorted(Counter(labels_test).items()))
    cm = confusion_matrix(preds_test, labels_test)
    print(cm)


if __name__ == '__main__':
    config = {
        'build_graph': True,
        'dataset_name': 'codedetect-taskA', # codedetect-taskA, codedetect-taskB
        'perc_dataset': 0.1, # procentaje del dataset: 0.1 -> 10% | 1 -> 100% (solo aplica para TRAIN y VAL, el conjunto de TEST se toma completo)
        'cut_off_dataset': '10-10-100', # train-val-test
        "nfi": 'llm', # llm, w2v, random
        'cuda_num': 0,

        'window_size': 10,
        'graph_direction': 'undirected', # undirected | directed 
        'special_chars': False, # punctuation
        'stop_words':    False,
        'min_df': 2, # 1->autext | 5->semeval | 5-coling
        'max_df': 1.0,
        'max_features': 50_000, # None -> all | 5000, 10000, 50000
        'not_found_tokens': 'avg', # avg, remove, zeros, ones
        'add_edge_attr': True,
        'embed_reduction': False, # False -> 768 
        'embed_dim': 768, # 128, 256, 768

        "gnn_type": 'TransformerConv', # GCNConv, GINConv, GATConv, TransformerConv
        "dropout": 0.5,
        "patience": 10, # 5-autext23 | 10-semeval | 10-coling
        "learnin_rate": 0.0001, # autext23_s2 -> llm: 0.0002 | autext23 -> llm: 0.00001 | semeval -> llm: 0.000005  | coling -> llm: 0.0001 
        "batch_size": 256 * 1,
        "hidden_dim": 100, # 300 autext_s2, 100 others
        "dense_hidden_dim": 32, # 64-autext23 | 32-semeval | 64-coling
        "num_layers": 2,
        "heads": 2,
        "norm_type": None,   # 'batchnorm', 'layernorm', or None
        "post_mp_layers": 2,        # number of layers after message passing
        "pooling_type": 'mean',      # 'mean', 'max', 'sum', 'attention', 'set2set'

        "weight_decay": 5e-2, 
        'input_dim': 256, # 768, 128
        'epochs': 200,
        "output_dim": 2, # 2-bin | 6-multi 
        "llm_name": 'microsoft/codebert-base',
        'leave_out_sources': True, # True, False
    }

    main(config)
