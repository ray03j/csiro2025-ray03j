import numpy as np


def weighted_r2_score_global(cfg, y_true: np.ndarray, y_pred: np.ndarray):
    weights = cfg.metric.r2_weights
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    w = np.tile(weights, y_true.shape[0])
    mean_w = np.sum(w * flat_true) / np.sum(w)
    ss_res = np.sum(w * (flat_true - flat_pred) ** 2)
    ss_tot = np.sum(w * (flat_true - mean_w) ** 2)
    global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    avg_r2, per_r2 = weighted_r2_score(y_true, y_pred)
    return global_r2, avg_r2, per_r2


def weighted_r2_score(cfg, y_true: np.ndarray, y_pred: np.ndarray):
    weights = cfg.metric.r2_weights
    r2_scores = []
    for i in range(y_true.shape[1]):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted_r2 = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted_r2, r2_scores

def calc_metric(cfg, outputs, targets):
    '''
        outputs/targets: shape (N, 3): Green/Clover/Dead
    '''
    weighted_r2, r2_scores = weighted_r2_score(
        cfg, 
        targets, 
        outputs
    )
    return weighted_r2, r2_scores