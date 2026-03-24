import numpy as np
import os
import pickle
import time
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, confusion_matrix, classification_report

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_models(task_type):
    if task_type == "classification":
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel='rbf', random_state=42, probability=True),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "SVR": SVR(kernel='rbf'),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42)

    return models


def train_and_compare(X, y, task_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = get_models(task_type)
    results = []

    for name, model in models.items():
        print(f"training {name}...")
        start = time.time()

        try:
            model.fit(X_train, y_train)
            train_time = round(time.time() - start, 2)

            y_pred = model.predict(X_test)

            if task_type == "classification":
                acc = round(accuracy_score(y_test, y_pred) * 100, 2)
                f1 = round(f1_score(y_test, y_pred, average='weighted') * 100, 2)

                cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                cv_mean = round(cv_scores.mean() * 100, 2)

                results.append({
                    "name": name,
                    "accuracy": acc,
                    "f1_score": f1,
                    "cv_score": cv_mean,
                    "train_time": train_time,
                    "model": model
                })
                print(f"  {name}: accuracy={acc}%, f1={f1}%, cv={cv_mean}%")

            else:
                r2 = round(r2_score(y_test, y_pred) * 100, 2)
                rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                cv_mean = round(cv_scores.mean() * 100, 2)

                results.append({
                    "name": name,
                    "r2_score": r2,
                    "rmse": rmse,
                    "cv_score": cv_mean,
                    "train_time": train_time,
                    "model": model
                })
                print(f"  {name}: r2={r2}%, rmse={rmse}, cv={cv_mean}%")

        except Exception as e:
            print(f"  {name} failed: {e}")
            results.append({
                "name": name,
                "error": str(e),
                "train_time": 0
            })

    if task_type == "classification":
        results.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
    else:
        results.sort(key=lambda x: x.get("r2_score", -999), reverse=True)

    best = results[0] if results else None

    best_details = None
    if best and "model" in best:
        best_model = best["model"]
        y_pred = best_model.predict(X_test)

        if task_type == "classification":
            cm = confusion_matrix(y_test, y_pred).tolist()
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            best_details = {
                "confusion_matrix": cm,
                "classification_report": report
            }
        else:
            best_details = {
                "predictions_vs_actual": {
                    "actual": y_test.head(20).tolist(),
                    "predicted": y_pred[:20].tolist()
                }
            }

    feature_importance = None
    if best and "model" in best:
        try:
            if hasattr(best["model"], 'feature_importances_'):
                importances = best["model"].feature_importances_
                feat_names = X.columns.tolist()
                feature_importance = sorted(
                    [{"feature": feat_names[i], "importance": round(float(importances[i]), 4)} for i in range(len(feat_names))],
                    key=lambda x: x["importance"], reverse=True
                )[:15]  # top 15
            elif hasattr(best["model"], 'coef_'):
                coefs = best["model"].coef_
                if len(coefs.shape) > 1:
                    coefs = coefs[0]
                feat_names = X.columns.tolist()
                feature_importance = sorted(
                    [{"feature": feat_names[i], "importance": round(abs(float(coefs[i])), 4)} for i in range(len(feat_names))],
                    key=lambda x: x["importance"], reverse=True
                )[:15]
        except Exception as e:
            print(f"couldnt get feature importance: {e}")

    model_path = None
    if best and "model" in best:
        model_filename = f"model_{int(time.time())}.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(best["model"], f)
        print(f"saved best model to {model_path}")

    # cant send model objects in json so remove them
    clean_results = []
    for r in results:
        clean = {k: v for k, v in r.items() if k != "model"}
        clean_results.append(clean)

    return {
        "results": clean_results,
        "best_model": best["name"] if best else None,
        "best_score": best.get("accuracy") or best.get("r2_score") if best else None,
        "best_details": best_details,
        "feature_importance": feature_importance,
        "model_path": model_path,
        "task_type": task_type
    }
