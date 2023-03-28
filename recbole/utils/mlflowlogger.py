r"""
recbole.utils.MLFlow
################################
"""

import time
class MLFlowLogger(object):
    """WandbLogger to log metrics to Weights and Biases."""

    def __init__(self, config):
        """
        Args:
            config (dict): A dictionary of parameters used by RecBole.
        """
        self.config = config
        self.log_mlflow = config.log_mlflow
        self.setup()

    def setup(self):
        if self.log_mlflow:
            try:
                import mlflow
                if mlflow.active_run() is None:                
                    self._mlflow = mlflow
                    self._mlflow.end_run()
                else:
                    self._mlflow = mlflow
                    self._mlflow.end_run()
                self._mlflow.set_tracking_uri('sqlite:///mlflow.db')
                self._mlflow.set_experiment('recbole')
                self._mlflow.start_run(run_name = self.config.model + '_' + str(int(time.time())))
                self._mlflow.log_params(self.config.final_config_dict)
                
            except ImportError:
                raise ImportError(
                    "To use mlflow Logger please install wandb."
                    "Run `pip install mlflow` to install it."
                )
                
            self._set_steps()

    def log_metrics(self, metrics, head="train", commit=True):
        if self.log_mlflow:
            if head == 'train':
                step = metrics['epoch']
                metrics = self._add_head_to_metrics(metrics, head)
                metrics_new = {}
                for metric in metrics.keys():
                    metrics_new[metric.replace('@','_')] = metrics[metric]
                
                
                self._mlflow.log_metrics(metrics_new,step = step)
            else:
                metrics_new = {}
                step = metrics['valid_step']
                metrics = self._add_head_to_metrics(metrics, head)
                for metric in metrics:
                    metrics_new[metric.replace('@','_')] = metrics[metric]
                self._mlflow.log_metrics(metrics_new,step = step)

    def log_eval_metrics(self, metrics, head="eval",step = 0):
        if self.log_mlflow:
            metrics_new = {}
            for metric in metrics.keys():
                metrics_new[metric.replace('@','_')] = metrics[metric]
            
            metrics = self._add_head_to_metrics(metrics_new, head)
            self._mlflow.log_metrics(metrics,step = step)

    def _set_steps(self):
        pass

    def _add_head_to_metrics(self, metrics, head):
        head_metrics = dict()
        for k, v in metrics.items():
            if "_step" in k:
                head_metrics[k] = v
            else:
                head_metrics[f"{head}/{k}"] = v

        return head_metrics
