

import torch
import numpy as np
import random
import os
import nltk 
import re
import contractions
from torch_geometric.utils import degree
import numpy as np
import networkx as nx
import json 
import logging
import joblib
import pandas as  pd
import sys
from nltk.corpus import stopwords
nltk.download('stopwords')


#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXTERNAL_DISK_PATH = '/media/discoexterno/andric/data/experiments/' # hetero_graph, cooc_graph
ROOT_DIR = os.path.dirname(os.path.dirname(__file__)) #+ '/GraphDeepLearning'
#DATASET_DIR = ROOT_DIR + '/datasets/'
DATASET_DIR = '/home/avaldez/projects/GraphDeepLearning/datasets/'
OUTPUT_DIR_PATH = ROOT_DIR + '/outputs/'
INPUT_DIR_PATH = ROOT_DIR + '/inputs/'

def read_json(dir_path):
    logger.debug("*** Using dataset: %s", dir_path)
    return pd.read_json(path_or_buf=dir_path, lines=True)

def read_csv(file_path):
  df = pd.read_csv(file_path)
  return df

def save_data(data, file_name, path=OUTPUT_DIR_PATH, format_file='.pkl', compress=False):
    logger.info('Saving data: %s', file_name)
    path_file = os.path.join(path, file_name + format_file)
    joblib.dump(data, path_file, compress=compress)

def load_data(file_name, path=INPUT_DIR_PATH, format_file='.pkl', compress=False):
    logger.info('Loading data: %s', file_name)
    path_file = os.path.join(path, file_name + format_file)
    return joblib.load(path_file)

def to_lowercase(text):
    return text.lower()

def handle_contraction_apostraphes(text):
    text = re.sub('([A-Za-z]+)[\'`]([A-Za-z]+)', r'\1'r'\2', text)
    return text

def handle_contraction(text):
  expanded_words = []
  for word in text.split():
    expanded_words.append(contractions.fix(word))
  return ' '.join(expanded_words)

def remove_blank_spaces(text):
    return re.sub(r'\s+', ' ', text).strip() # remove blank spaces

def remove_html_tags(text):
    return re.compile('<.*?>').sub(r'', text) # remove html tags

def remove_special_chars(text):
    text = re.sub('[^A-Za-z0-9]+ ', ' ', text) # remove special chars
    text = re.sub('\W+', ' ', text) # remove special chars
    text = text.replace('"'," ")
    text = text.replace('('," ")
    text = re.sub(r'\s+', ' ', text).strip() # remove blank spaces
    return text

def remove_stop_words(text):
    # remove stop words
    tokens = nltk.word_tokenize(text)
    without_stopwords = [word for word in tokens if not word.lower().strip() in set(stopwords.words('english'))]
    text = " ".join(without_stopwords)
    return text

def text_normalize(text, special_chars=False, stop_words=False):
    text = to_lowercase(text)
    text = handle_contraction(text)
    text = handle_contraction_apostraphes(text)
    text = remove_blank_spaces(text)
    text = remove_html_tags(text)
    if special_chars:
        text = remove_special_chars(text)
    if stop_words: 
        text = remove_stop_words(text)
    return text

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def read_dataset(dataset_name, print_info=True):
    if dataset_name in ['semeval24', 'semeval24_s2']:
        
        #dataset_name = 'autext23' # autext23, autext23_s2
        if dataset_name == 'semeval24': # subtask1, subtask2
            subtask = 'subtask1' 
        if dataset_name == 'semeval24_s2':
            subtask = 'subtask2' #

        autext_train_set = read_json(dir_path=f'{DATASET_DIR}semeval2024/{subtask}/train_set.jsonl')
        autext_val_set = read_json(dir_path=f'{DATASET_DIR}semeval2024/{subtask}/dev_set.jsonl')
        autext_test_set = read_json(dir_path=f'{DATASET_DIR}semeval2024/{subtask}/test_set.jsonl')
        
        autext_train_set = autext_train_set.sample(frac=1).reset_index(drop=True)
        autext_val_set = autext_val_set.sample(frac=1).reset_index(drop=True)
        #autext_test_set = autext_test_set.sample(frac=1).reset_index(drop=True)
        
        autext_test_set['source'] = 'unknown'
        autext_test_set['model'] = 'unknown'
        autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
        autext_val_set['word_len'] = autext_val_set['text'].str.split().str.len()
        autext_test_set['word_len'] = autext_test_set['text'].str.split().str.len()

        if print_info:
            print("autext_train_set: ", autext_train_set.info())
            print("\n min_max_avg_token Train: ", autext_train_set['word_len'].min(), autext_train_set['word_len'].max(), int(autext_train_set['word_len'].mean()))
            print("min_max_avg_token Val:   ", autext_val_set['word_len'].min(), autext_val_set['word_len'].max(),  int(autext_val_set['word_len'].mean()))
            print("min_max_avg_token Test:  ", autext_test_set['word_len'].min(), autext_test_set['word_len'].max(), int(autext_test_set['word_len'].mean()))
            print("total_distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)

        min_token_len = 1
        max_token_len = 5000
        autext_train_set = autext_train_set[(autext_train_set['word_len'] >= min_token_len) & (autext_train_set['word_len'] <= max_token_len)]
        autext_val_set = autext_val_set[(autext_val_set['word_len'] >= min_token_len) & (autext_val_set['word_len'] <= max_token_len)]

        #autext_train_set, autext_val_set = train_test_split(autext_train_set, test_size=0.3)
        if print_info:
            print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))

            print(autext_train_set.nlargest(5, ['word_len']) )
            #autext_val_set = pd.concat([autext_val_set, autext_val_set_2], axis=0)

            print("autext_train_set: ", autext_train_set.info())
            print("autext_val_set: ", autext_val_set.info())
            print("autext_test_set: ", autext_test_set.info())
            print(autext_train_set['model'].value_counts())
            print(autext_val_set['model'].value_counts())

            # Model distribution for each source
            print("Model distribution per source in Train set:\n", autext_train_set["source"].value_counts())
            print("Model distribution per source in Validation set:\n", autext_val_set["source"].value_counts())

        
        return autext_train_set, autext_val_set, autext_test_set


    # ****************************** READ DATASET AUTEXT 2023
    if dataset_name in ['autext23', 'autext23_s2']:
        
        #dataset_name = 'autext23' # autext23, autext23_s2
        if dataset_name == 'autext23': # subtask1, subtask2
            subtask = 'subtask1' 
        if dataset_name == 'autext23_s2':
            subtask = 'subtask2' #
        
        autext_train_set = read_csv(file_path=f'{DATASET_DIR}autext2023/{subtask}/train_set.csv') 
        autext_val_set = read_csv(file_path=f'{DATASET_DIR}autext2023/{subtask}/val_set.csv') 
        autext_test_set = read_csv(file_path=f'{DATASET_DIR}autext2023/{subtask}/test_set.csv') 
        
        autext_train_set = autext_train_set.sample(frac=1).reset_index(drop=True)
        autext_val_set = autext_val_set.sample(frac=1).reset_index(drop=True)
        #autext_test_set = autext_test_set.sample(frac=1).reset_index(drop=True)
        
        autext_train_set.rename(columns={'domain': 'source'}, inplace=True)
        autext_val_set.rename(columns={'domain': 'source'}, inplace=True)
        autext_test_set.rename(columns={'domain': 'source'}, inplace=True)

        if print_info:
            print("autext_train_set: ", autext_train_set.info())
            print("autext_val_set: ", autext_val_set.info())
            print("autext_test_set: ", autext_test_set.info())
            print("total_distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)
            print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))
            print("source_distro_train_val_test: ", autext_train_set.value_counts('source'), autext_val_set.value_counts('source'), autext_test_set.value_counts('source'))
            print("model_distro_train_val_test: ", autext_train_set.value_counts('model'), autext_val_set.value_counts('model'), autext_test_set.value_counts('model'))
            
            # Model distribution for each source
            print("Model distribution per source in Train set:\n", autext_train_set.groupby("source")["model"].value_counts())
            print("Model distribution per source in Validation set:\n", autext_val_set.groupby("source")["model"].value_counts())
            print("Model distribution per source in Test set:\n", autext_test_set.groupby("source")["model"].value_counts())

            # Label distribution for each source
            print("Label distribution per source in Train set:\n", autext_train_set.groupby("source")["label"].value_counts())
            print("Label distribution per source in Validation set:\n", autext_val_set.groupby("source")["label"].value_counts())
            print("Label distribution per source in Test set:\n", autext_test_set.groupby("source")["label"].value_counts())


        autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
        autext_val_set['word_len'] = autext_val_set['text'].str.split().str.len()
        autext_test_set['word_len'] = autext_test_set['text'].str.split().str.len()
        if print_info: 
            print("min_max_avg_token Train: ", autext_train_set['word_len'].min(), autext_train_set['word_len'].max(), int(autext_train_set['word_len'].mean()))
            print("min_max_avg_token Val:   ", autext_val_set['word_len'].min(), autext_val_set['word_len'].max(),  int(autext_val_set['word_len'].mean()))
            print("min_max_avg_token Test:  ", autext_test_set['word_len'].min(), autext_test_set['word_len'].max(), int(autext_test_set['word_len'].mean()))
        
        return autext_train_set, autext_val_set, autext_test_set

    # ****************************** READ DATASET AUTEXT 2023
    if dataset_name in ['autext24', 'autext24_s2']:
        
        #dataset_name = 'autext23' # autext23, autext23_s2
        if dataset_name == 'autext24': # subtask1, subtask2
            subtask = 'subtask1' 
        if dataset_name == 'autext24_s2':
            subtask = 'subtask2' #
        
        autext_train_set = read_csv(file_path=f'{DATASET_DIR}autext2024/{subtask}/train_set.csv') 
        autext_val_set = read_csv(file_path=f'{DATASET_DIR}autext2024/{subtask}/val_set.csv') 
        autext_test_set = read_csv(file_path=f'{DATASET_DIR}autext2024/{subtask}/test_set.csv') 
        
        autext_train_set = autext_train_set.sample(frac=1).reset_index(drop=True)
        autext_val_set = autext_val_set.sample(frac=1).reset_index(drop=True)
        #autext_test_set = autext_test_set.sample(frac=1).reset_index(drop=True)
        
        autext_train_set.rename(columns={'domain': 'source'}, inplace=True)
        autext_val_set.rename(columns={'domain': 'source'}, inplace=True)
        autext_test_set.rename(columns={'domain': 'source'}, inplace=True)

        if print_info:
            print("autext_train_set: ", autext_train_set.info())
            print("autext_val_set: ", autext_val_set.info())
            print("autext_test_set: ", autext_test_set.info())
            print("total_distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)
            print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))
            print("source_distro_train_val_test: ", autext_train_set.value_counts('source'), autext_val_set.value_counts('source'), autext_test_set.value_counts('source'))
            print("model_distro_train_val_test: ", autext_train_set.value_counts('model'), autext_val_set.value_counts('model'), autext_test_set.value_counts('model'))
            print("language_distro_train_val_test: ", autext_train_set.value_counts('language'), autext_val_set.value_counts('language'), autext_test_set.value_counts('language'))
            
            # Model distribution for each source
            print("Model distribution per source in Train set:\n", autext_train_set.groupby("source")["model"].value_counts())
            print("Model distribution per source in Validation set:\n", autext_val_set.groupby("source")["model"].value_counts())
            print("Model distribution per source in Test set:\n", autext_test_set.groupby("source")["model"].value_counts())

            # Label distribution for each source
            print("Label distribution per source in Train set:\n", autext_train_set.groupby("source")["label"].value_counts())
            print("Label distribution per source in Validation set:\n", autext_val_set.groupby("source")["label"].value_counts())
            print("Label distribution per source in Test set:\n", autext_test_set.groupby("source")["label"].value_counts())

            # Label distribution for each source
            print("language distribution per source in Train set:\n", autext_train_set.groupby("language")["label"].value_counts())
            print("language distribution per source in Validation set:\n", autext_val_set.groupby("language")["label"].value_counts())
            print("language distribution per source in Test set:\n", autext_test_set.groupby("language")["label"].value_counts())

        autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
        autext_val_set['word_len'] = autext_val_set['text'].str.split().str.len()
        autext_test_set['word_len'] = autext_test_set['text'].str.split().str.len()
        if print_info:
            print("min_max_avg_token Train: ", autext_train_set['word_len'].min(), autext_train_set['word_len'].max(), int(autext_train_set['word_len'].mean()))
            print("min_max_avg_token Val:   ", autext_val_set['word_len'].min(), autext_val_set['word_len'].max(),  int(autext_val_set['word_len'].mean()))
            print("min_max_avg_token Test:  ", autext_test_set['word_len'].min(), autext_test_set['word_len'].max(), int(autext_test_set['word_len'].mean()))
            
        return autext_train_set, autext_val_set, autext_test_set

    # ****************************** READ DATASET COLING 2024
    if dataset_name in ['coling24']:   
        #dataset_name = 'coling24' 
        autext_train_set = read_json(dir_path=f'{DATASET_DIR}coling2024/en_train.jsonl')
        autext_val_set = read_json(dir_path=f'{DATASET_DIR}coling2024/en_dev.jsonl')
        autext_test_set = read_json(dir_path=f'{DATASET_DIR}coling2024/test_set_en_with_label.jsonl')
        
        autext_train_set = autext_train_set.sample(frac=1).reset_index(drop=True)
        autext_val_set = autext_val_set.sample(frac=1).reset_index(drop=True)
        #autext_test_set = autext_test_set.sample(frac=1).reset_index(drop=True)

        autext_test_set['id'] = range(len(autext_test_set))
        
        #autext_test_set = autext_test_set[['testset_id', 'label', 'text']]
        if print_info:
            #print("distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)
            print("autext_train_set: ", autext_train_set.info())
            print("autext_val_set: ", autext_val_set.info())
            print("autext_test_set: ", autext_test_set.info())

            # Model distribution for each source
            print("Model distribution per source in Train set:\n", autext_train_set.groupby("source")["model"].value_counts())
            print("Model distribution per source in Validation set:\n", autext_val_set.groupby("source")["model"].value_counts())
            print("Model distribution per source in Test set:\n", autext_test_set.groupby("source")["model"].value_counts())

            # Label distribution for each source
            print("Label distribution per source in Train set:\n", autext_train_set.groupby("source")["label"].value_counts())
            print("Label distribution per source in Validation set:\n", autext_val_set.groupby("source")["label"].value_counts())
            print("Label distribution per source in Test set:\n", autext_test_set.groupby("source")["label"].value_counts())

        autext_train_set['word_len'] = autext_train_set['text'].str.split().str.len()
        autext_val_set['word_len'] = autext_val_set['text'].str.split().str.len()
        autext_test_set['word_len'] = autext_test_set['text'].str.split().str.len()
        if print_info:
            print("min_max_avg_token Train: ", autext_train_set['word_len'].min(), autext_train_set['word_len'].max(), int(autext_train_set['word_len'].mean()))
            print("min_max_avg_token Val:   ", autext_val_set['word_len'].min(), autext_val_set['word_len'].max(),  int(autext_val_set['word_len'].mean()))
            print("min_max_avg_token Test:  ", autext_test_set['word_len'].min(), autext_test_set['word_len'].max(), int(autext_test_set['word_len'].mean()))
            print("total_distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)

        min_token_text = 1
        max_token_text = 1500
        autext_train_set = autext_train_set[(autext_train_set['word_len'] >= min_token_text) & (autext_train_set['word_len'] <= max_token_text)]
        autext_val_set = autext_val_set[(autext_val_set['word_len'] >= min_token_text) & (autext_val_set['word_len'] <= max_token_text)]

        if print_info:
            print("label_distro_train_val_test: ", autext_train_set.value_counts('label'), autext_val_set.value_counts('label'), autext_test_set.value_counts('label'))
            #print(autext_train_set.nlargest(5, ['word_len']) )

            print("distro_train_val_test: ", autext_train_set.shape, autext_val_set.shape, autext_test_set.shape)
            #print(autext_train_set['model'].value_counts())
            #print(autext_val_set['model'].value_counts())
            
        return autext_train_set, autext_val_set, autext_test_set


def stratified_sample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    # Muestra frac dentro de cada clase
    out = (
        df.groupby('label', group_keys=False)
          .apply(lambda x: x.sample(frac=frac, random_state=seed))
          .reset_index(drop=True)
    )
    return out
