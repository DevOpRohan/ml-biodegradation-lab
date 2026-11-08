"""Shared CSV contract and binary-classification evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

FEATURES = [
    'SpMax_L', 'J_Dz(e)', 'nHM', 'F01[N-N]', 'F04[C-N]', 'NssssC',
    'nCb-', 'C%', 'nCp', 'nO', 'F03[C-N]', 'SdssC', 'HyWi_B(m)',
    'LOC', 'SM6_L', 'F03[C-O]', 'Me', 'Mi', 'nN-N', 'nArNO2',
    'nCRX3', 'SpPosA_B(p)', 'nCIR', 'B01[C-Br]', 'B03[C-Cl]', 'N-073',
    'SpMax_A', 'Psi_i_1d', 'B04[C-Br]', 'SdO', 'TI2_L', 'nCrt',
    'C-026', 'F02[C-N]', 'nHDon', 'SpMax_B(m)', 'Psi_i_A', 'nN',
    'SM6_B(m)', 'nArCOOR', 'nX',
]
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'svm.joblib',
}


def validate_test_data(frame):
    required = FEATURES + ['target']
    missing = sorted(set(required) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(required))
    if missing or extra or frame.columns.duplicated().any():
        raise ValueError(f'CSV columns do not match. Missing: {missing}; unexpected: {extra}.')
    if not 1 <= len(frame) <= 10000:
        raise ValueError('Upload between 1 and 10,000 test rows.')
    try:
        X = frame[FEATURES].apply(pd.to_numeric, errors='raise').astype(float)
        y = pd.to_numeric(frame['target'], errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Features must be numeric; target must be 0 or 1.') from exc
    if not np.isfinite(X.to_numpy()).all() or not y.isin([0, 1]).all():
        raise ValueError('Remove missing/infinite values and encode target as 0 or 1.')
    return X, y.astype(int)


def evaluate_predictions(y_true, y_pred, scores):
    return {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'AUC': float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float('nan'),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'MCC': float(matthews_corrcoef(y_true, y_pred)),
    }


def positive_scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, list(model.classes_).index(1)]
    return model.decision_function(X)


def model_metrics(model, X, y):
    return evaluate_predictions(y, model.predict(X), positive_scores(model, X))
