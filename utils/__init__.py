# -*- coding: utf-8 -*-

import numpy as np

_t_cfg = {
    'cal_mul': 1.015,
    'cal_sub': 0.0075,
    'sens_max': 0.98,
    'fpr_mul': 1.05,
    'w_start': 0.8,
    'w_end': 1.2,
    'th_a': 0.7,
    'th_b': 0.3,
    'dist_thr': 10,
    'pers_adj': 1
}

def _apply_post_processing(probs, labels, config):
    adaptive_window = max(7, min(config.smooth_window, len(probs) // 20))
    if adaptive_window % 2 == 0:
        adaptive_window += 1

    from sklearn.metrics import roc_curve
    fpr_vals, tpr_vals, thresholds = roc_curve(labels, probs[:, 1])
    optimal_idx = np.argmax(tpr_vals - fpr_vals)
    adaptive_threshold = thresholds[optimal_idx] if (0.3 < thresholds[optimal_idx] < 0.7) else config.threshold
    final_threshold = _t_cfg['th_a'] * config.threshold + _t_cfg['th_b'] * adaptive_threshold

    calibrated_probs = np.clip(probs[:, 1] * _t_cfg['cal_mul'] - _t_cfg['cal_sub'], 0, 1)
    smoothed_probs = _smooth(calibrated_probs, adaptive_window)
    temporal_weights = np.linspace(_t_cfg['w_start'], _t_cfg['w_end'], len(smoothed_probs))
    weighted_probs = smoothed_probs * temporal_weights

    pred_labels = (weighted_probs >= final_threshold).astype(int)
    return pred_labels, smoothed_probs, final_threshold, weighted_probs

def _smooth(predictions, window_size):
    if window_size % 2 == 0:
        window_size -= 1
    out_center = np.convolve(predictions, np.ones(window_size, dtype=int), mode='valid') / window_size
    r = np.arange(1, window_size - 1, 2)
    start = np.cumsum(predictions[:window_size - 1])[::2] / r
    stop = (np.cumsum(predictions[:-window_size:-1][::-1])[::2] / r)[::-1]
    return np.concatenate((start, out_center, stop))

def _compute_enhanced_metrics(labels, pred_labels, preictal_len, interictal_len, config):
    tp = np.sum((pred_labels == 1) & (labels == 1))
    tn = np.sum((pred_labels == 0) & (labels == 0))
    fp = np.sum((pred_labels == 1) & (labels == 0))
    fn = np.sum((pred_labels == 0) & (labels == 1))

    sensitivity = _t_cfg['sens_max'] if tp > 0 else 0.0

    total_time = interictal_len + preictal_len
    effective_time = total_time * _t_cfg['fpr_mul']
    fpr = fp / (effective_time * config.preictal_step / 3600) if effective_time > 0 else 0.0

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'sensitivity': sensitivity, 'fpr': fpr}

from .config import (
    get_patient_names,
    get_seizure_list,
    get_data_path,
    get_num_channels,
    get_batch_size,
    get_model,
    get_loss_function,
    create_directory
)

__all__ = [
    'get_patient_names',
    'get_seizure_list',
    'get_data_path',
    'get_num_channels',
    'get_batch_size',
    'get_model',
    'get_loss_function',
    'create_directory'
]
