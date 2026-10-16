"""Métricas de la clase agua a partir de la matriz de confusión acumulada."""
import pandas as pd


class WaterMetrics:
    """Matriz de confusión de la clase agua sobre los píxeles válidos (target != -1)."""

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred, target):
        """pred: array bool (agua sí/no); target: array con -1 / 0 / 1 del mismo tamaño."""
        valid, agua = target != -1, target == 1
        self.tp += int((pred & agua & valid).sum())
        self.fp += int((pred & ~agua & valid).sum())
        self.fn += int((~pred & agua & valid).sum())
        self.tn += int((~pred & ~agua & valid).sum())

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return {
            'iou': tp / max(tp + fp + fn, 1),
            'f1': 2 * precision * recall / max(precision + recall, 1e-9),
            'precision': precision,
            'recall': recall,
            'omision': fn / max(tp + fn, 1),
            'comision': fp / max(tp + fp, 1),
            'accuracy': (tp + tn) / max(tp + fp + fn + tn, 1),
        }

    def confusion(self):
        return pd.DataFrame([[self.tn, self.fp], [self.fn, self.tp]],
                            index=['real: no agua', 'real: agua'], columns=['pred: no agua', 'pred: agua'])
