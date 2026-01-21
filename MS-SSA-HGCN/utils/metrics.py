# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def compute_metrics(predictions, labels, threshold=0.5):
    if len(predictions.shape) > 1 and predictions.shape[1] == 2:
        pred_probs = predictions[:, 1]
    else:
        pred_probs = predictions.squeeze()

    pred_binary = (pred_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1_score = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)

    try:
        auc_value = roc_auc_score(labels, pred_probs)
    except ValueError:
        auc_value = 0.0

    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'auc': auc_value,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

    return metrics


def plot_roc_curve(labels, probabilities, save_path=None):
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    _, ax = plt.subplots(figsize=(5.5, 5))

    ax.plot(fpr, tpr, color='#FF6B35', lw=2.5, alpha=0.9,
            label=f'ROC (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#FF6B35')

    ax.plot([0, 1], [0, 1], color='#6C757D', lw=1.5, linestyle='--', alpha=0.7,
            label='Random Classifier')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontweight='medium')
    ax.set_ylabel('True Positive Rate', fontweight='medium')
    ax.set_title('ROC Curve', fontweight='semibold', pad=10)
    ax.legend(loc='lower right', frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved ROC curve to: {save_path}")
    else:
        plt.show()

    plt.close()
    return roc_auc


def plot_predictions(predictions, labels, preictal_len, interictal_len,
                   save_path, metrics, threshold):
    _, ax = plt.subplots(figsize=(10, 3.5))

    total_len = len(predictions)
    x_axis = np.arange(total_len)

    ax.plot(x_axis, predictions, color='#2E86AB', linewidth=1.5, alpha=0.9, label='Prediction')
    ax.fill_between(x_axis, predictions, alpha=0.2, color='#2E86AB')

    ax.axhline(y=threshold, color='#D64045', linestyle='--', linewidth=1.8, alpha=0.8, label='Threshold')

    ax.axvspan(0, interictal_len, alpha=0.15, color='#06A77D', label='Interictal')
    ax.axvspan(interictal_len, interictal_len + preictal_len, alpha=0.15, color='#FFBC42', label='Preictal')
    if interictal_len + preictal_len < total_len:
        ax.axvspan(interictal_len + preictal_len, total_len, alpha=0.15, color='#D64045', label='Ictal')

    ax.axvline(x=interictal_len, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=interictal_len + preictal_len, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel('Time Steps', fontweight='medium')
    ax.set_ylabel('Prediction Probability', fontweight='medium')
    ax.set_title(f"Seizure Prediction Results\nAUC={metrics['auc']:.4f} | Sens={metrics['sensitivity']:.1%} | FPR={metrics['fpr']:.3f}/h",
                fontweight='semibold', pad=10)
    ax.legend(loc='upper right', frameon=True, shadow=True, fancybox=True, ncol=4)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, axis='y')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def print_metrics_summary(metrics):
    print("\n" + "=" * 55)
    print("Evaluation Metrics Summary".center(55))
    print("=" * 55)
    print(f"{'Accuracy:':<20} {metrics['accuracy']:>10.4%}")
    print(f"{'Sensitivity:':<20} {metrics['sensitivity']:>10.4%}")
    print(f"{'Specificity:':<20} {metrics['specificity']:>10.4%}")
    print(f"{'Precision:':<20} {metrics['precision']:>10.4%}")
    print(f"{'F1 Score:':<20} {metrics['f1_score']:>10.4f}")
    print(f"{'AUC:':<20} {metrics['auc']:>10.4f}")
    print("-" * 55)
    print(f"{'True Positives (TP):':<20} {metrics['tp']:>10}")
    print(f"{'True Negatives (TN):':<20} {metrics['tn']:>10}")
    print(f"{'False Positives (FP):':<20} {metrics['fp']:>10}")
    print(f"{'False Negatives (FN):':<20} {metrics['fn']:>10}")
    print("=" * 55)
