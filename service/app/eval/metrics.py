import numpy as np
from service.app.model.utils import inverse_transform
import torch

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns) + 1e-9
    return mean_excess_return / std_excess_return * np.sqrt(365 * 24)


def compute_metrics_multistep(y_true_scaled, y_pred_scaled, scaler, target_feature_index=0):
    horizon = y_true_scaled.shape[1]
    metrics_per_step = []
    for i in range(horizon):
        y_true = inverse_transform(scaler, y_true_scaled[:, i], target_feature_index)
        y_pred = inverse_transform(scaler, y_pred_scaled[:, i], target_feature_index)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        returns_true = np.diff(y_true) / y_true[:-1]
        returns_pred = np.diff(y_pred) / y_pred[:-1]

        da = np.mean(np.sign(returns_true) == np.sign(returns_pred))
        sr = sharpe_ratio(returns_pred)

        metrics_per_step.append({
            'step': i + 1,
            'RMSE': rmse,
            'Directional Accuracy': da,
            'Sharpe Ratio': sr
        })
    return metrics_per_step

