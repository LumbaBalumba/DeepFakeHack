from typing import Optional
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve


def compute_eer(y_true, y_score):
    fpr, tpr, threshold = roc_curve(y_true, y_score)

    eps = 1e-3
    threshold[0] = max(threshold[1:]) + eps

    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fnr[eer_index]
    return eer


IMPLEMENTED_METRICS = {"eer": compute_eer}


class MetricsLogger:
    def __init__(
        self,
        metrics: list[str],
        logging_dir: Optional[Path] = None,
        eps: float = 1e-8,
    ):
        self.eps = eps
        self.metrics = self.create_metrics_dict(metrics)
        if logging_dir is not None:
            self.writer = SummaryWriter(logging_dir)

    def create_metrics_dict(self, metric_names):
        if not set(metric_names).issubset(IMPLEMENTED_METRICS):
            unknown_metrics = set(IMPLEMENTED_METRICS) - set(metric_names)
            raise NotImplementedError(
                f"These metrics are not implemented: {unknown_metrics}\n",
                f"Currently available: {IMPLEMENTED_METRICS}",
            )
        return {metric_name: [] for metric_name in metric_names}

    def compute_metrics(self, gt, pred):
        for metric_name in self.metrics.keys():
            self.metrics[metric_name].append(
                IMPLEMENTED_METRICS[metric_name](gt=gt, pred=pred)
            )

    def log_scalar(self, value, mode, epoch, name="Loss"):
        self.writer.add_scalar(f"{mode}/{name}", value, epoch + 1)

    def reset_metrics(self):
        for key in self.metrics.keys():
            self.metrics[key] = []

    def log_metrics(self, mode, epoch):
        for metric_name, metric_values in self.metrics.items():
            self.writer.add_scalar(
                f"{mode}/{metric_name}", np.mean(metric_values), epoch + 1
            )
        self.write_image(epoch + 1)
        self.reset_metrics()

    def get_metrics(self):
        return self.metrics
