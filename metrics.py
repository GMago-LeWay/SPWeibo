from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_error

__all__ = ['MetricsTop']

class Metrics():
    def __init__(self):
        self.metrics_dict = {
            'match': self._eval_binary,
            'rnn': self._eval_regression,
            'tcn': self._eval_regression,
            'spwrnn': self._eval_regression,
        }
    
    def get_metrics(self, modelName):
        return self.metrics_dict[modelName]

    def _eval_binary(self, y_pred, y_true):
        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()

        eval_results = {
            "acc": round(accuracy_score(y_true=y_true, y_pred=y_pred), 4),
            "recall": round(recall_score(y_true=y_true, y_pred=y_pred), 4),
            "f1": round(f1_score(y_true=y_true, y_pred=y_pred) , 4),
        }
        return eval_results

    def _eval_regression(self, y_pred, y_true):
        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()

        eval_results = {
            "mse": round(mean_squared_error(y_true=y_true, y_pred=y_pred), 4),
            "mse_last": round(mean_squared_error(y_true=y_true[:, -1:], y_pred=y_pred[:, -1:]), 4),
        }
        return eval_results        

