import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set MLflow experiment
mlflow.set_experiment("RealEstate_Investment_Advisor")

def evaluate_classification(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = 0.0
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm

def evaluate_regression(y_true, y_pred):
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def train():
    print("Loading preprocessed data...")
    df = pd.read_csv("cleaned_data.csv")
    
    # Load required feature columns expected by models
    feature_cols = joblib.load("models/feature_columns.pkl")
    X = df[feature_cols]
    
    # -------------------------------------------------------------
    # Classification Task: target = Good_Investment
    # -------------------------------------------------------------
    y_class = df['Good_Investment']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    best_class_model = None
    best_class_f1 = -1
    best_classifier_name = ""
    
    class_metrics_results = {}
    best_cm = None

    print("Training Classification Models...")
    for model_name, model in classifiers.items():
        with mlflow.start_run(run_name=f"{model_name}"):
            model.fit(X_train_c, y_train_c)
            y_pred = model.predict(X_test_c)
            y_prob = model.predict_proba(X_test_c)[:, 1] if hasattr(model, "predict_proba") else None
            
            metrics, cm = evaluate_classification(y_test_c, y_pred, y_prob)
            
            # Logging
            mlflow.log_param("model_type", "classification")
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            
            if "XGB" in model_name:
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            print(f"{model_name} Metrics: {metrics}")
            class_metrics_results[model_name] = metrics
            
            if metrics['f1'] > best_class_f1:
                best_class_f1 = metrics['f1']
                best_class_model = model
                best_classifier_name = model_name
                best_cm = cm

    print(f"Best Classifier: {best_classifier_name} (F1: {best_class_f1:.4f})")
    joblib.dump(best_class_model, "models/best_classifier.pkl")
    
    with open("models/classification_metrics.json", "w") as f:
        json.dump(class_metrics_results, f)
        
    plt.figure(figsize=(8,6))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_classifier_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('models/best_classifier_cm.png')
    plt.close()

    # -------------------------------------------------------------
    # Regression Task: target = Future_Price_5Y
    # -------------------------------------------------------------
    y_reg = df['Future_Price_5Y']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor()
    }
    
    best_reg_model = None
    best_reg_r2 = -float("inf")
    best_reg_name = ""
    
    reg_metrics_results = {}

    print("\nTraining Regression Models...")
    for model_name, model in regressors.items():
        with mlflow.start_run(run_name=f"{model_name}"):
            model.fit(X_train_r, y_train_r)
            y_pred = model.predict(X_test_r)
            
            metrics = evaluate_regression(y_test_r, y_pred)
            
            # Logging
            mlflow.log_param("model_type", "regression")
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            
            if "XGB" in model_name:
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
                
            print(f"{model_name} Metrics: {metrics}")
            reg_metrics_results[model_name] = metrics
            
            if metrics['r2'] > best_reg_r2:
                best_reg_r2 = metrics['r2']
                best_reg_model = model
                best_reg_name = model_name

    print(f"Best Regressor: {best_reg_name} (R2: {best_reg_r2:.4f})")
    joblib.dump(best_reg_model, "models/best_regressor.pkl")
    
    with open("models/regression_metrics.json", "w") as f:
        json.dump(reg_metrics_results, f)
    
    print("\nModels successfully trained and serialized.")

if __name__ == "__main__":
    if not os.path.exists("cleaned_data.csv") or not os.path.exists("models/feature_columns.pkl"):
        print("Error: Missing cleaned_data.csv or feature columns list. Run preprocessing.py first.")
    else:
        train()
