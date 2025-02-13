import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import pandas as pd

ALL_METRICS = {
    'precision': (precision_score, 'micro'),
    'f1_micro': (f1_score, 'micro'),
    'f1_macro': (f1_score, 'macro'),
    'f1_weighted': (f1_score, 'weighted')
}
"""
Class to compute metrics
"""
class Metrics:

    def __init__(self, model: str):

        self._model = model
        self._metrics = {
            'precision': list(),
            'f1_micro': list(),
            'f1_macro': list(),
            'f1_weighted': list()
        }
        
    def add_results(self, y_true, y_pred):

        for key, (func, param) in ALL_METRICS.items():
            if param is None:
                self._metrics[key].append(func(y_true, y_pred))
            else:
                self._metrics[key].append(func(y_true, y_pred, average=param))
    
    def compute_stats(self):
        stats_results = {}
        for key, values in self._metrics.items():
            mean_value = np.mean(values)
            std_dev = np.std(values)
            margin_of_error = self._compute_margin_of_error(values)
            confidence_interval = self._compute_confidence_interval(values)
            
            stats_results[key] = {
                'model': self._model,
                'mean': mean_value,
                'std': std_dev,
                'confidence_interval': confidence_interval,
                'error_margin': margin_of_error
            }
        
        self._stats_results = stats_results
        return stats_results

    def _compute_margin_of_error(self, values, confidence=0.95):
        std_dev = np.std(values)
        n = len(values)
        h = std_dev / np.sqrt(n) * stats.t.ppf((1 + confidence) / 2., n-1)
        return h

    def _compute_confidence_interval(self, values, confidence=0.95):
        mean = np.mean(values)
        std_dev = np.std(values)
        n = len(values)
        h = std_dev / np.sqrt(n) * stats.t.ppf((1 + confidence) / 2., n-1)
        return (mean - h, mean + h)
    

"""
Function to flatten results
"""
def flatten_metrics(metrics_data: list):

    flattened_data = []
    for metric_data in metrics_data:
        for metric_name, metric_details in metric_data.items():
            metric_details['metric'] = metric_name
            flattened_data.append(metric_details)

    df = pd.DataFrame(flattened_data)

    df = df[['model', 'metric', 'mean', 'std', 'confidence_interval', 'error_margin']]

    print(df)