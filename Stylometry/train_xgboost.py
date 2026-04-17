import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from data_utils import build_features_df


ROUTE = 'Your/route/here/'

train_df = pd.read_parquet(ROUTE + 'train.parquet')
df_train_feats = build_features_df(train_df).dropna()

scaler = StandardScaler()
X_train = df_train_feats.drop(['index_original', 'label'], axis=1)
y_train = df_train_feats['label']
X_train_scaled = scaler.fit_transform(X_train)


xgb = XGBClassifier(
    device='cuda',
    n_estimators=10000,
    learning_rate=0.1,
    max_depth=6,
    objective='multi:softprob',
    num_class=y_train.nunique(),
    random_state=42
)
xgb.fit(X_train_scaled, y_train)

val_df = pd.read_parquet(ROUTE + 'validation.parquet')
df_val_feats = build_features_df(val_df)
X_val_scaled = scaler.transform(df_val_feats.drop(['index_original', 'label'], axis=1))

preds = xgb.predict(X_val_scaled)
print(classification_report(df_val_feats['label'], preds))
print(f"Weighted F1: {f1_score(df_val_feats['label'], preds, average='weighted')}")