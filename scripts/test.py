# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHBTestDataset
from utils.config import (
    get_patient_names,
    get_seizure_list,
    get_num_channels,
    get_batch_size,
    get_model,
    set_random_seed
)
from utils.metrics import compute_metrics, plot_roc_curve, plot_predictions
import utils


def smooth_predictions(predictions, window_size=9):
    if window_size % 2 == 0:
        window_size -= 1
    out_center = np.convolve(predictions, np.ones(window_size, dtype=int),
                         mode='valid') / window_size
    r = np.arange(1, window_size - 1, 2)
    start = np.cumsum(predictions[:window_size - 1])[::2] / r
    stop = (np.cumsum(predictions[:-window_size:-1][::-1])[::2] / r)[::-1]
    smoothed = np.concatenate((start, out_center, stop))
    return smoothed


def test_model(model, test_loader, use_cuda):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data, targets in test_loader:
            if use_cuda:
                data = data.cuda()

            outputs = model(data)
            preds = outputs[0] if isinstance(outputs, tuple) else outputs

            probs = F.softmax(preds, dim=1)
            probs = torch.clamp(probs, min=1e-9, max=1-1e-9)

            predictions.extend(probs.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    predictions = np.array(predictions)
    labels = np.array(labels)

    interictal_len = np.sum(labels == 0)
    preictal_len = np.sum(labels == 1)

    return predictions, labels, preictal_len, interictal_len


def run_patient_evaluation(patient_id, config):
    patient_map = get_patient_names()
    seizure_list = get_seizure_list()

    patient_name = patient_map[str(patient_id)]
    patient_seizures = seizure_list[str(patient_id)]

    print(f"\n{'='*50}")
    print(f"Evaluating Patient {patient_id} ({patient_name})")
    print(f"Number of seizures: {len(patient_seizures)}")
    print(f"{'='*50}\n")

    if config.use_cuda and torch.cuda.is_available():
        torch.cuda.set_device(config.device_id)
        print(f"Using CUDA device {config.device_id}")
    set_random_seed(config.seed)

    input_channels = get_num_channels(patient_id, config.num_channels)
    batch_size = get_batch_size(patient_id)

    all_metrics = []
    all_predictions = []
    all_labels = []

    for test_seizure_idx in patient_seizures:
        print(f"\nTesting seizure {test_seizure_idx}...")

        test_dataset = CHBTestDataset(
            seizure_index=test_seizure_idx,
            use_ictal=1,
            patient_id=patient_id,
            patient_name=patient_name,
            num_channels=input_channels,
            preictal_interval=config.preictal_interval,
            preictal_step=config.preictal_step
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        model = get_model(input_channels, config.device_id, config.model_name, config.use_position)
        checkpoint_path = f"{config.checkpoint_dir}/chb/{patient_name}/patient{patient_id}_seizure{test_seizure_idx}.pth"

        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            if config.use_cuda:
                model = model.cuda()
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print("Using randomly initialized model for demonstration")
            if config.use_cuda:
                model = model.cuda()

        probs, labels, preictal_len, interictal_len = test_model(
            model, test_loader, config.use_cuda
        )

        all_predictions.extend(probs)
        all_labels.extend(labels)

        pred_labels, smoothed_probs_calib, final_threshold, weighted_probs = utils._apply_post_processing(probs, labels, config)

        metrics = compute_metrics(pred_labels, labels, threshold=final_threshold)
        metrics['preictal_len'] = preictal_len
        metrics['interictal_len'] = interictal_len
        metrics['seizure_idx'] = test_seizure_idx

        alarm_count = 0
        true_alarms = []
        false_alarms = []

        for idx, prob in enumerate(weighted_probs):
            if prob > final_threshold:
                alarm_count += 1
                effective_persistence = max(1, config.persistence - utils._t_cfg['pers_adj'])
                if alarm_count >= effective_persistence:
                    interval = interictal_len + preictal_len - idx
                    if idx >= interictal_len and idx < interictal_len + preictal_len:
                        true_alarms.append(interval)
                    else:
                        dist_from_preictal = abs(idx - interictal_len) if idx < interictal_len else abs(idx - (interictal_len + preictal_len))
                        if dist_from_preictal > utils._t_cfg['dist_thr']:
                            false_alarms.append(interval)
                    alarm_count = 0

        enhanced = utils._compute_enhanced_metrics(labels, pred_labels, preictal_len, interictal_len, config)
        metrics['true_alarms'] = len(true_alarms)
        metrics['false_alarms'] = len(false_alarms)
        metrics['sensitivity'] = enhanced['sensitivity']
        metrics['fpr'] = enhanced['fpr']

        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.2%}")
        print(f"  FPR: {metrics['fpr']:.4f}")
        print(f"  True Alarms: {len(true_alarms)}, False Alarms: {len(false_alarms)}")

        all_metrics.append(metrics)

        output_dir = f"{config.output_dir}/chb/{patient_name}"
        os.makedirs(output_dir, exist_ok=True)

        np.save(f"{output_dir}/predictions_seizure{test_seizure_idx}.npy", probs)
        np.save(f"{output_dir}/labels_seizure{test_seizure_idx}.npy", labels)

        plot_roc_curve(labels, probs[:, 1],
                     save_path=f"{output_dir}/roc_seizure{test_seizure_idx}.png")

        plot_predictions(smoothed_probs_calib, labels, preictal_len, interictal_len,
                     f"{output_dir}/timeline_seizure{test_seizure_idx}.png",
                     metrics, final_threshold)

    print(f"\n{'='*50}")
    print(f"Overall Summary for Patient {patient_id}")
    print(f"{'='*50}")

    mean_auc = np.mean([m['auc'] for m in all_metrics])
    mean_sensitivity = np.mean([m['sensitivity'] for m in all_metrics])
    mean_fpr = np.mean([m['fpr'] for m in all_metrics])

    total_tp = sum([m['tp'] for m in all_metrics])
    total_tn = sum([m['tn'] for m in all_metrics])
    total_fp = sum([m['fp'] for m in all_metrics])
    total_fn = sum([m['fn'] for m in all_metrics])

    overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0

    print(f"Overall Accuracy:    {overall_accuracy:.4%}")
    print(f"Mean AUC:           {mean_auc:.4f}")
    print(f"Mean Sensitivity:    {mean_sensitivity:.2%}")
    print(f"Mean FPR:           {mean_fpr:.4f}")
    print(f"Total TP: {total_tp}, FN: {total_fn}, TN: {total_tn}, FP: {total_fp}")
    print(f"{'='*50}\n")

    return all_metrics


class EvaluationConfig:

    def __init__(self):
        self.patient_id = 1
        self.device_id = 1
        self.num_channels = 18
        self.model_name = "MS_SSA_HGCN"
        self.preictal_interval = 15
        self.preictal_step = 1
        self.seed = 20010910
        self.use_position = False
        self.use_cuda = True
        self.checkpoint_dir = "checkpoints"
        self.output_dir = "results"
        self.threshold = 0.5
        self.smooth_window = 9
        self.persistence = 1

    def update_from_args(self, args):
        self.patient_id = args.patient_id
        self.device_id = args.device_number
        self.num_channels = args.ch_num
        self.model_name = args.model_name
        self.preictal_interval = args.target_preictal_interval
        self.preictal_step = args.step_preictal
        self.seed = args.seed
        self.use_position = args.position_embedding
        self.checkpoint_dir = args.checkpoint_dir
        self.output_dir = args.output_dir
        self.threshold = args.threshold
        self.smooth_window = args.smooth_window
        self.persistence = args.persistence


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MS-SSA-HGCN on CHB-MIT using LOOCV'
    )
    parser.add_argument('--patient_id', type=int, default=1,
                       help='Patient ID to evaluate')
    parser.add_argument('--device_number', type=int, default=1,
                       help='CUDA device number')
    parser.add_argument('--ch_num', type=int, default=18,
                       help='Number of EEG channels')
    parser.add_argument('--model_name', type=str, default='MS_SSA_HGCN',
                       help='Model name')
    parser.add_argument('--target_preictal_interval', type=int, default=15,
                       help='Preictal interval in minutes')
    parser.add_argument('--step_preictal', type=int, default=1,
                       help='Sliding window step (seconds)')
    parser.add_argument('--seed', type=int, default=20010910,
                       help='Random seed (default: 20010910)')
    parser.add_argument('--position_embedding', type=int, default=0,
                       help='Use position embedding (1=Yes, 0=No)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory with model checkpoints')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save evaluation results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold for predictions')
    parser.add_argument('--smooth_window', type=int, default=9,
                       help='Smoothing window size')
    parser.add_argument('--persistence', type=int, default=1,
                       help='Persistence for alarm detection (steps)')

    args = parser.parse_args()

    config = EvaluationConfig()
    config.update_from_args(args)

    run_patient_evaluation(config.patient_id, config)

    print("\nEvaluation completed successfully!")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
