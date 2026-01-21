# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import torch.utils.data as Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import CHBTrainDataset, CHBTestDataset
from trainers import ModelTrainer
from utils.config import (
    get_patient_names,
    get_seizure_list,
    get_num_channels,
    get_batch_size,
    get_model,
    get_loss_function,
    create_directory,
    set_random_seed
)
from utils.metrics import compute_metrics, print_metrics_summary


def run_loocv_training(patient_id, config):
    patient_map = get_patient_names()
    seizure_list = get_seizure_list()

    patient_name = patient_map[str(patient_id)]
    patient_seizures = seizure_list[str(patient_id)]
    num_seizures = len(patient_seizures)

    print(f"\n{'='*50}")
    print(f"Training Patient {patient_id} ({patient_name})")
    print(f"Number of seizures: {num_seizures}")
    print(f"{'='*50}\n")

    if config.use_cuda and torch.cuda.is_available():
        torch.cuda.set_device(config.device_id)
        print(f"Using CUDA device {config.device_id}")
    else:
        print("Using CPU")

    set_random_seed(config.seed)

    input_channels = get_num_channels(patient_id, config.num_channels)
    model = get_model(input_channels, config.device_id, config.model_name, config.use_position)

    if config.use_cuda:
        model = model.cuda()

    loss_fn = get_loss_function(config.loss_name)

    all_results = []
    best_fold = 0
    best_auc = 0.0

    for test_seizure_idx in patient_seizures:
        train_seizures = [s for s in patient_seizures if s != test_seizure_idx]

        print(f"\n{'-'*50}")
        print(f"Fold: Testing on seizure {test_seizure_idx}")
        print(f"Training on seizures: {train_seizures}")
        print(f"{'-'*50}")

        train_dataset = CHBTrainDataset(
            seizure_indices=train_seizures,
            num_iter=1,
            augment=config.augment,
            use_ictal=1,
            balance_classes=1,
            patient_id=patient_id,
            patient_name=patient_name,
            num_channels=input_channels,
            preictal_interval=config.preictal_interval,
            preictal_step=config.preictal_step
        )

        test_dataset = CHBTestDataset(
            seizure_index=test_seizure_idx,
            use_ictal=1,
            patient_id=patient_id,
            patient_name=patient_name,
            num_channels=input_channels,
            preictal_interval=config.preictal_interval,
            preictal_step=config.preictal_step
        )

        batch_size = get_batch_size(patient_id)
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        trainer = ModelTrainer(model, loss_fn, train_loader, test_loader, config)

        trained_model, train_losses, train_accs, val_aucs = trainer.train()

        test_acc, test_loss, test_auc = trainer.test()

        fold_result = {
            'test_seizure': test_seizure_idx,
            'train_seizures': train_seizures,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'test_auc': test_auc,
            'best_val_auc': max(val_aucs) if val_aucs else 0.0,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_aucs': val_aucs
        }
        all_results.append(fold_result)

        if test_auc > best_auc:
            best_auc = test_auc
            best_fold = test_seizure_idx

        checkpoint_dir = f"{config.checkpoint_dir}/chb/{patient_name}"
        create_directory(checkpoint_dir)
        checkpoint_path = f"{checkpoint_dir}/patient{patient_id}_seizure{test_seizure_idx}.pth"
        torch.save(trained_model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")

    print(f"\n{'='*50}")
    print(f"LOOCV Summary for Patient {patient_id}")
    print(f"{'='*50}")

    mean_auc = np.mean([r['test_auc'] for r in all_results])
    std_auc = np.std([r['test_auc'] for r in all_results])
    mean_acc = np.mean([r['test_accuracy'] for r in all_results])

    print(f"Mean Test Accuracy: {mean_acc:.4%}")
    print(f"Mean Test AUC:      {mean_auc:.4f} (±{std_auc:.4f})")
    print(f"Best Fold:          Seizure {best_fold} (AUC: {best_auc:.4f})")
    print(f"{'='*50}\n")

    return {
        'patient_id': patient_id,
        'patient_name': patient_name,
        'num_seizures': num_seizures,
        'fold_results': all_results,
        'mean_accuracy': mean_acc,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'best_fold': best_fold,
        'best_auc': best_auc
    }


class TrainingConfig:

    def __init__(self):
        self.patient_id = 1
        self.device_id = 1
        self.num_channels = 18
        self.model_name = "MS_SSA_HGCN"
        self.preictal_interval = 15
        self.preictal_step = 1
        self.loss_name = "FL"
        self.seed = 20010910
        self.augment = 1
        self.use_position = False
        self.use_cuda = True
        self.checkpoint_dir = "checkpoints"
        self.batch_size = 256
        self.learning_rate = 0.001
        self.weight_decay = 5e-4
        self.num_epochs = 100
        self.early_stop_patience = 15
        self.validate_during_training = True

    def update_from_args(self, args):
        self.patient_id = args.patient_id
        self.device_id = args.device_number
        self.num_channels = args.ch_num
        self.model_name = args.model_name
        self.preictal_interval = args.target_preictal_interval
        self.preictal_step = args.step_preictal
        self.loss_name = args.loss
        self.seed = args.seed
        self.augment = args.augmentation
        self.use_position = args.position_embedding
        self.checkpoint_dir = args.checkpoint_dir
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.early_stop_patience = args.early_stop_patience


def main():
    parser = argparse.ArgumentParser(
        description='Train MS-SSA-HGCN on CHB-MIT using LOOCV'
    )
    parser.add_argument('--patient_id', type=int, default=1,
                       help='Patient ID to train')
    parser.add_argument('--device_number', type=int, default=1,
                       help='CUDA device number')
    parser.add_argument('--ch_num', type=int, default=18,
                       help='Number of EEG channels')
    parser.add_argument('--model_name', type=str, default='MS_SSA_HGCN',
                       help='Model name')
    parser.add_argument('--target_preictal_interval', type=int, default=15,
                       help='Preictal interval in minutes')
    parser.add_argument('--step_preictal', type=int, default=1,
                       help='Sliding window step for preictal (seconds)')
    parser.add_argument('--loss', type=str, default='FL',
                       choices=['CE', 'FL'],
                       help='Loss function: CE=Cross Entropy, FL=Focal Loss')
    parser.add_argument('--seed', type=int, default=20010910,
                       help='Random seed for reproducibility')
    parser.add_argument('--augmentation', type=int, default=1,
                       help='Use data augmentation (1=Yes, 0=No)')
    parser.add_argument('--position_embedding', type=int, default=0,
                       help='Use position embedding (1=Yes, 0=No)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--early_stop_patience', type=int, default=15,
                       help='Early stopping patience (epochs)')

    args = parser.parse_args()

    config = TrainingConfig()
    config.update_from_args(args)

    results = run_loocv_training(config.patient_id, config)

    print("\nTraining completed successfully!")
    print(f"Results saved to: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
