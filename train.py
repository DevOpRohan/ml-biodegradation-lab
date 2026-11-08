"""Reproduce the fixed holdout experiment and save all five trained pipelines."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from ml_utils import FEATURES, MODEL_FILES, model_metrics

ROOT = Path(__file__).resolve().parent
SEED = 26


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', default='local', help='Execution environment label')
    args = parser.parse_args()
    raw_path = ROOT / 'data/biodeg.csv'
    raw = pd.read_csv(raw_path, sep=';', header=None, names=FEATURES + ['target'])
    assert raw.shape == (1055, 42)
    assert set(raw.target) == {'RB', 'NRB'}
    # Remove exact duplicate compounds before splitting, preventing train/test leakage.
    data = raw.drop_duplicates().copy()
    data['target'] = data.target.map({'NRB': 0, 'RB': 1})
    assert np.isfinite(data.to_numpy()).all()
    train, test = train_test_split(data, test_size=.20, stratify=data.target, random_state=SEED)
    train.to_csv(ROOT / 'data/train_data.csv', index=False)
    test.to_csv(ROOT / 'test_data.csv', index=False)
    X_train, y_train = train[FEATURES], train.target
    X_test, y_test = test[FEATURES], test.target
    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=3000, random_state=SEED)),
        'Decision Tree': make_pipeline(DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=SEED)),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11, weights='distance')),
        'Naive Bayes': make_pipeline(GaussianNB(var_smoothing=1e-9)),
        'Support Vector Machine': make_pipeline(StandardScaler(), SVC(C=3., kernel='rbf', gamma='scale', random_state=SEED)),
    }
    (ROOT / 'model').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    rows, reports = [], {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for (name, model), axis in zip(models.items(), axes.flat):
        model.fit(X_train, y_train)
        joblib.dump(model, ROOT / 'model' / MODEL_FILES[name], compress=3)
        rows.append({'Model': name, **model_metrics(model, X_test, y_test)})
        prediction = model.predict(X_test)
        reports[name] = classification_report(y_test, prediction, labels=[0, 1],
            target_names=['NRB', 'RB'], output_dict=True, zero_division=0)
        ConfusionMatrixDisplay.from_predictions(y_test, prediction, labels=[0, 1],
            display_labels=['NRB', 'RB'], ax=axis, colorbar=False, cmap='Blues')
        axis.set_title(name)
    axes.flat[-1].axis('off')
    fig.suptitle('Fixed held-out test set | positive class: ready biodegradable (RB)')
    fig.tight_layout()
    fig.savefig(ROOT / 'results/confusion_matrices.png', dpi=160)
    plt.close(fig)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ROOT / 'results/metrics.csv', index=False)
    (ROOT / 'results/classification_reports.json').write_text(json.dumps(reports, indent=2))
    metadata = {
        'executed_at_utc': datetime.now(timezone.utc).isoformat(),
        'environment': args.environment, 'os': platform.system(),
        'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
        'numpy': np.__version__, 'pandas': pd.__version__, 'random_seed': SEED,
        'raw_rows': len(raw), 'duplicates_removed': len(raw) - len(data),
        'feature_count': len(FEATURES), 'train_rows': len(train), 'test_rows': len(test),
        'train_class_counts': train.target.value_counts().sort_index().to_dict(),
        'test_class_counts': test.target.value_counts().sort_index().to_dict(),
        'source_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'test_sha256': hashlib.sha256((ROOT / 'test_data.csv').read_bytes()).hexdigest(),
        'model_sha256': {name: hashlib.sha256((ROOT / 'model' / filename).read_bytes()).hexdigest()
                         for name, filename in MODEL_FILES.items()},
    }
    (ROOT / 'results/run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print('QSAR BIODEGRADATION | FIVE-CLASSIFIER EXPERIMENT')
    print(f'Environment: {args.environment} | {platform.system()} | sklearn {sklearn.__version__}')
    print(f'Raw: {len(raw)} rows | removed duplicates: {len(raw)-len(data)} | features: {len(FEATURES)}')
    print(f'Train: {len(train)} | Test: {len(test)} | seed: {SEED} | positive: RB=1')
    print(metrics.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('Saved: model/*.joblib, test_data.csv, results/metrics.csv and execution metadata')


if __name__ == '__main__':
    main()
