import re
import seaborn as sns
import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Definimos las funciones principales para el análisis de errores

def read_data(prediction_path, test_path):
    # Leer el archivo de la predicción
    pred_df = pd.read_csv(prediction_path, sep=',')
    # Leer el archivo de test
    test_df = pd.read_parquet(test_path)

    columnas = test_df.columns
    logger.info(f"Columns in test dataframe: {columnas}")

    return pred_df, test_df


# Obtenemos la matriz de confusión y las métricas de evaluación
def confusion_matriz(pred_df, test_df, save_path):
    # Accurracy, recall, precision, f1-score
    pred = pred_df['prediction'].tolist()
    test = test_df['label'].tolist()
    print("Accuracy:", accuracy_score(test, pred))
    print("Recall:", recall_score(test, pred, average='macro'))
    print("Precision:", precision_score(test, pred, average='macro'))
    print("F1 Score:", f1_score(test, pred, average='macro'))

    cm = confusion_matrix(test, pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Guardamos la figura
    plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix saved to {save_path}")

def extract_generator_family(generator_name):

    patterns = {
    'Human': r'human',
    'Deepseek': r'deepseek',
    'Qwen': r'qwen',
    'phi': r'phi',
    'GPT': r'gpt',
    'Llama': r'llama',
    'Mistral': r'mistral', 
    'gemma': r'gemma',
    'bigcode': r'bigcode',
    'Gemini': r'gemini',
    'Granite': r'granite',
    '01-AI': r'01-ai',
    }
        
    generator_lower = str(generator_name).lower()
    for family, pattern in patterns.items():
        if re.search(pattern, generator_lower, re.IGNORECASE):
            return family
    return 'other'

# Con esta función simplemente esperamos definir dos nuevas columnas: longitud de código y las familias de los generadores.
def data_preprocessing(test_df):
    test_df['code_length'] = test_df['code'].str.len()
    test_df['generator_family'] = test_df['generator'].apply(extract_generator_family)
    return test_df

# Analizar los errores para la tarea A corresponde en dividir el dataframe en los cuatro tipos de respuestas: TP, TN, FP y FN.
# Para analizar errores en las otras tareas que son tareas multiclase el análisis tendría que ser similar.
# Especificamente por el tamaño reducido de las muestras de testeo para la tarea B la comparativa en errores podría ser similar a la A, considerar clase humana vs maquina.
    
def error_division_multiclase(pred_df, test_df):
    dataframes = {}
    for label in test_df['label'].unique():
        TN_df = test_df[(test_df['label'] == pred_df['prediction']) & (test_df['label'] == label)]
        FN_df = test_df[(test_df['label'] != pred_df['prediction']) & (test_df['label'] != label) & (pred_df['prediction'] == label)]
        FP_df = test_df[(test_df['label'] != pred_df['prediction']) & (test_df['label'] == label) & (pred_df['prediction'] != label)]
        dataframes[f'TN_{label}'] = TN_df
        dataframes[f'FP_{label}'] = FP_df
        dataframes[f'FN_{label}'] = FN_df

    return dataframes

# La siguiente función se ecargará de mostrar estadísticas de los subdaframes por columnas
# Solo nos mostrará estadísticas representativas

def error_analysis(dataframes, columnas, save_path):
    for nombre, dataframe in dataframes.items():

        # Filtramos aquellos dataframes con menos de 100 muestras para evitar ruido en los histogramas
        if len(dataframe) >= 65:

            for col in columnas:
                if type(dataframe.iloc[0][col]) is str:
                    plt.figure(figsize=(10,5))
                    sns.countplot(data=dataframe, x = col, palette = 'tab10')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.title(f'Histograma de {col}')
                    plt.xticks(rotation=45)
                    plt.savefig(f'{save_path}/{nombre}_{col}_histogram.png', dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.figure(figsize=(10,5))
                    plt.hist(dataframe[col], bins=40, range=(0,4000), alpha=0.7, color='blue')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.title(f'Histograma de {col} (C)')
                    plt.savefig(f'{save_path}/{nombre}_{col}_histogram.png', dpi=300, bbox_inches='tight')
                    plt.close()
                logger.info(f"Saved histogram for {nombre}_{col}")
    logger.info(f"All histograms saved to {save_path}")

# Esta función es el pipeline que hará el proceso completo

def pipeline(prediction_path, test_path, save_path):
    pred_df, test_df = read_data(prediction_path, test_path)
    confusion_matriz(pred_df, test_df, save_path)
    test_df = data_preprocessing(test_df)
    dataframes = error_division_multiclase(pred_df, test_df)
    columnas = ['language','generator_family', 'code_length']
    error_analysis(dataframes, columnas, save_path)

def main():
    parser = argparse.ArgumentParser(description= 'Analisis de errores de modelos en las tareas de SemEval-2026')
    parser.add_argument('--prediction_path', type=str, required= True, help='Path archivo .csv de las prediccionds')
    parser.add_argument('--test_path', type=str, required= True, help='Path para el archivo .parquet de este se ontendrán los errores y aciertos')
    parser.add_argument('--save_path', type=str, required=True, help='Directory where all the histograms will be stored')
    args = parser.parse_args()

    #Ejecutamos el pipeline completo
    logger.info("Starting error analysis pipeline...")

    pipeline(args.prediction_path, 
             args.test_path, 
             args.save_path
             )
if __name__ == "__main__":
    main()