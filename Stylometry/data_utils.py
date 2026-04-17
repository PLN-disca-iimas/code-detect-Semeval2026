import pandas as pd
from features_extractor import extract_all_features

def build_features_df(df, is_test=False):
    final_data = []
    for idx, row in df.iterrows():
        current_id = row['ID'] if is_test else idx
        feat = extract_all_features(row['code'])
        
        registro = {
            'index_original': current_id,
            'unique_tokens': feat[0],
            'total_tokens': feat[1],
            'token_ratio': feat[2],
            'avg_length_line': feat[3],
            'ident_mean': feat[4],
            'ident_std': feat[5],
            'ident_levels': feat[6],
            'code_length': feat[7],
            'total_comments': feat[8],
            'comments_density': feat[9]
        }
        if not is_test:
            registro['label'] = row['label']
            
        final_data.append(registro)
    
    return pd.DataFrame(final_data)