"""Run with python -m unittest -v after python train.py."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml_utils import FEATURES, MODEL_FILES, evaluate_predictions, validate_test_data


class EvaluationTests(unittest.TestCase):
    def test_metrics_match_hand_calculation(self):
        actual = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 0], [.1, .4, .8, .3])
        expected = dict(Accuracy=.5, AUC=.75, Precision=.5, Recall=.5, F1=.5, MCC=0.)
        self.assertEqual(actual, expected)

    def test_single_class_auc_is_unavailable(self):
        result = evaluate_predictions([0, 0], [0, 1], [.1, .6])
        self.assertTrue(np.isnan(result['AUC']))

    def test_upload_schema_and_label_validation(self):
        frame = pd.DataFrame([{**{c: 1 for c in FEATURES}, 'target': 1}])
        X, y = validate_test_data(frame[frame.columns[::-1]])
        self.assertEqual(list(X.columns), FEATURES)
        self.assertEqual(y.tolist(), [1])
        for invalid in [frame.drop(columns=FEATURES[0]), frame.assign(target=1.5),
                        frame.assign(target='RB'), frame.assign(**{FEATURES[0]: np.inf}),
                        frame.iloc[:0], frame.assign(extra=0)]:
            with self.assertRaises(ValueError):
                validate_test_data(invalid)


class ArtifactTests(unittest.TestCase):
    def test_models_reproduce_report_without_training(self):
        import joblib
        from ml_utils import model_metrics
        frame = pd.read_csv('test_data.csv')
        X, y = validate_test_data(frame)
        expected = pd.read_csv('results/metrics.csv').set_index('Model')
        self.assertEqual(len(MODEL_FILES), 5)
        for name, filename in MODEL_FILES.items():
            model = joblib.load(Path('model') / filename)
            values = model_metrics(model, X, y)
            np.testing.assert_allclose(list(values.values()), expected.loc[name].values, atol=1e-12)
        train = pd.read_csv('data/train_data.csv')
        train_rows = set(map(tuple, train[FEATURES].to_numpy()))
        test_rows = set(map(tuple, X.to_numpy()))
        self.assertFalse(train_rows & test_rows, 'Feature duplicates leak across split')
        self.assertEqual(len(train) + len(frame), 1052)

    def test_streamlit_default_data_and_model_switch(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file('app.py', default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 6)
        for name in MODEL_FILES:
            app.selectbox[0].select(name).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.metric), 6)


if __name__ == '__main__':
    unittest.main()
