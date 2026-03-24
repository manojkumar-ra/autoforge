import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(df, target_col, task_type):
    df = df.copy()

    # drop columns that are mostly empty
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    if target_col not in df.columns:
        return None, None, None, "Target column was dropped because too many missing values"

    df = df.dropna(subset=[target_col])

    if len(df) < 10:
        return None, None, None, "Not enough data after cleaning (need at least 10 rows)"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    label_encoder = None
    if task_type == "classification" and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=target_col)

    # drop useless columns
    drop_cols = []
    for col in X.columns:
        if col.lower() in ['id', 'index', 'unnamed: 0', 'unnamed']:
            drop_cols.append(col)
            continue
        if X[col].nunique() <= 1:
            drop_cols.append(col)
            continue
        if X[col].dtype == 'object' and X[col].nunique() > 50:
            drop_cols.append(col)
            continue

    X = X.drop(columns=drop_cols, errors='ignore')

    if len(X.columns) == 0:
        return None, None, None, "No usable features found after cleaning"

    # one hot encode categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)

    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())

    X = X.dropna(axis=1)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    preprocessing_info = {
        "dropped_columns": drop_cols,
        "encoded_categories": cat_cols,
        "final_features": len(X_scaled.columns),
        "final_rows": len(X_scaled),
        "feature_names": X_scaled.columns.tolist()
    }

    return X_scaled, y, preprocessing_info, None
