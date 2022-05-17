from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_error
import torch

__all__ = ['MetricsTop']

class Metrics():
    def __init__(self):
        self.metrics_dict = {
            'framing': self._eval_multi,
            'rnn': self._eval_regression,
            'tcn': self._eval_regression,
            'spwrnn': self._eval_regression,
            'spwrnn_beta': self._eval_regression,
            'spwrnn_wo_l': self._eval_regression,
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

    def _eval_multi(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # y_pred probit -> 0,1
        y_pred = ( y_pred.detach() > 0.5 ).int()
        eval_results = {'acc_avg': 0., 'recall_avg': 0., 'f1_avg': 0.}
        items = y_pred.shape[1]
        for i in range(items):
            # dim i binary metrics
            results_i = self._eval_binary(y_pred=y_pred[:, i:i+1], y_true=y_true[:, i:i+1])
            eval_results['acc_avg'] += results_i['acc']
            eval_results['recall_avg'] += results_i['recall']
            eval_results['f1_avg'] += results_i['f1']
            for key in results_i:
                eval_results[f'{key}_{i}'] = results_i[key]

        eval_results['acc_avg'] /= items
        eval_results['recall_avg'] /= items
        eval_results['f1_avg'] /= items
        return eval_results

    def _eval_regression(self, y_pred, y_true):
        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()

        eval_results = {
            "mse": round(mean_squared_error(y_true=y_true, y_pred=y_pred), 4),
            "mse_last": round(mean_squared_error(y_true=y_true[:, -1:], y_pred=y_pred[:, -1:]), 4),
        }
        return eval_results        

