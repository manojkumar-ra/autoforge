import pandas as pd
import numpy as np


def analyze_dataset(df):
    total_rows = len(df)
    total_cols = len(df.columns)

    columns_info = []
    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        missing = int(col_data.isnull().sum())
        missing_pct = round(missing / total_rows * 100, 1)
        unique = int(col_data.nunique())

        if col_data.dtype in ['int64', 'float64', 'int32', 'float32']:
            col_type = "numeric"
            stats = {
                "mean": round(float(col_data.mean()), 2) if not col_data.isnull().all() else None,
                "median": round(float(col_data.median()), 2) if not col_data.isnull().all() else None,
                "min": round(float(col_data.min()), 2) if not col_data.isnull().all() else None,
                "max": round(float(col_data.max()), 2) if not col_data.isnull().all() else None,
                "std": round(float(col_data.std()), 2) if not col_data.isnull().all() else None,
            }
        elif col_data.dtype == 'object' or col_data.dtype.name == 'category':
            if unique > 50 or (unique / total_rows > 0.5):
                col_type = "text"
            else:
                col_type = "categorical"
            top_vals = col_data.value_counts().head(5).to_dict()
            stats = {"top_values": {str(k): int(v) for k, v in top_vals.items()}}
        elif 'datetime' in str(col_data.dtype):
            col_type = "datetime"
            stats = {}
        else:
            col_type = "other"
            stats = {}

        columns_info.append({
            "name": col,
            "dtype": dtype,
            "type": col_type,
            "missing": missing,
            "missing_pct": missing_pct,
            "unique": unique,
            "stats": stats
        })

    potential_targets = []
    for col_info in columns_info:
        if col_info["type"] == "categorical" and col_info["unique"] >= 2 and col_info["unique"] <= 20:
            potential_targets.append({"name": col_info["name"], "task": "classification", "classes": col_info["unique"]})
        elif col_info["type"] == "numeric" and col_info["unique"] > 10:
            potential_targets.append({"name": col_info["name"], "task": "regression", "classes": None})

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "columns": columns_info,
        "potential_targets": potential_targets,
        "sample_data": df.head(5).fillna("").to_dict(orient="records")
    }


def detect_task_type(df, target_col):
    col = df[target_col]
    unique_vals = col.nunique()

    if col.dtype == 'object' or col.dtype.name == 'category':
        return "classification"

    if unique_vals <= 15:
        return "classification"

    return "regression"
