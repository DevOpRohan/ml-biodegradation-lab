"""Interactive evaluation of saved classifiers; uploaded data is never used to fit."""
from pathlib import Path
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve

from ml_utils import FEATURES, MODEL_FILES, model_metrics, positive_scores, validate_test_data

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Biodegradation Model Lab', page_icon='🌱', layout='wide')


@st.cache_resource
def load_models():
    # Only trusted, repository-owned artifacts are deserialized.
    return {name: joblib.load(ROOT / 'model' / filename) for name, filename in MODEL_FILES.items()}


st.caption('MACHINE LEARNING LAB / QSAR CLASSIFICATION')
st.title('Will this molecule biodegrade?')
st.write('Compare five classifiers using 41 molecular descriptors. Explore the held-out test set or upload your own labelled test CSV.')
st.sidebar.header('Experiment controls')
choice = st.sidebar.selectbox('Classifier', list(MODEL_FILES))
upload = st.sidebar.file_uploader('Upload test data (CSV)', type=['csv'],
    help='All 41 feature columns and target are required. Target: 0 = NRB, 1 = RB. Maximum 10,000 rows / 5 MB.')
sample_bytes = (ROOT / 'test_data.csv').read_bytes()
st.sidebar.download_button('Download test CSV', sample_bytes, 'test_data.csv', 'text/csv')
st.sidebar.caption('Target 0: not ready biodegradable (NRB)\n\nTarget 1: ready biodegradable (RB)')
st.sidebar.markdown('[Dataset source: UCI](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation)')

try:
    if upload is not None and upload.size > 5_000_000:
        raise ValueError('CSV exceeds the 5 MB limit.')
    frame = pd.read_csv(io.BytesIO(upload.getvalue() if upload else sample_bytes))
    X, y = validate_test_data(frame)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(str(exc))
    st.info('Download the test CSV above to see the required format.')
    st.stop()

models = load_models()
comparison = pd.DataFrame([{'Model': name, **model_metrics(model, X, y)}
                           for name, model in models.items()]).set_index('Model')
model = models[choice]
prediction = model.predict(X)
scores = positive_scores(model, X)
source = 'Uploaded test CSV' if upload else 'Bundled held-out test set'
st.info(f'{source} · {len(X):,} molecules · 41 features · {int(y.sum())} RB / {int((y == 0).sum())} NRB. Saved models only; no training occurs here.')
st.subheader(choice)
for column, (metric, value) in zip(st.columns(6), comparison.loc[choice].items()):
    column.metric(metric, 'N/A' if pd.isna(value) else f'{value:.4f}')
st.caption('Precision, recall and F1 use RB=1 as the positive class. AUC uses probabilities, or the SVM decision score. MCC accounts for all four confusion-matrix cells.')
if y.nunique() < 2:
    st.warning('AUC requires both target classes and is unavailable for this CSV. Other metrics are still shown.')

left, right = st.columns([1, 1.25])
with left:
    st.subheader('Where does the model make errors?')
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, prediction, labels=[0, 1],
        display_labels=['NRB (0)', 'RB (1)'], ax=ax, colorbar=False, cmap='YlGnBu')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader('All five models on this test data')
    st.dataframe(comparison.style.format('{:.4f}', na_rep='N/A'), width='stretch')
    st.download_button('Download comparison', comparison.to_csv().encode(), 'model_comparison.csv', 'text/csv')
    st.caption('A false positive predicts RB for an NRB molecule; a false negative misses an RB molecule. Compare MCC and recall as well as accuracy.')

with st.expander('Classification report and ROC comparison'):
    report = classification_report(y, prediction, labels=[0, 1], target_names=['NRB', 'RB'],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, width='stretch')
    if y.nunique() == 2:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, fitted in models.items():
            fpr, tpr, _ = roc_curve(y, positive_scores(fitted, X))
            ax.plot(fpr, tpr, label=f'{name} ({comparison.loc[name, "AUC"]:.3f})')
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curves (AUC in legend)')
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with st.expander('Inspect rows and download predictions'):
    output = frame.copy()
    output['predicted_target'] = prediction
    output['positive_score'] = scores
    st.dataframe(output.head(100), width='stretch')
    st.download_button('Download predictions', output.to_csv(index=False).encode(), 'predictions.csv', 'text/csv')
    st.caption('SVM scores are signed decision values; the other models provide RB probabilities.')

with st.expander('How the experiment works'):
    st.markdown('''The UCI dataset has 1,055 rows. Three exact duplicates are removed before an 80/20 stratified split (seed 26), leaving 841 training rows and 211 test rows.

Logistic Regression, KNN and SVM include a StandardScaler fitted only on training rows. Decision Tree and Gaussian Naive Bayes use the original descriptors. Hyperparameters are fixed before inspecting the holdout; this is a reproducible baseline comparison, not a tuned leaderboard.

The app loads five saved pipelines from `model/`. Uploads are validated and used only for evaluation. This educational benchmark does not establish performance on new chemical families or replace laboratory testing.''')
    st.code(', '.join(FEATURES) + ', target', language=None)
