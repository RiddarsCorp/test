"""Metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def report(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def format_report(name: str, metrics: dict) -> str:
    return f"{name}: MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.3f}"
