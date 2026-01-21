# -*- coding: utf-8 -*-
"""
Training Dataset Loader for CHB-MIT
Author: QinYongKang
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

# Add parent directory to path for config imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_patient_names, get_data_path


class CHBTrainDataset(Dataset):
    """
    Dataset loader for training CHB-MIT data

    Supports Leave-One-Seizure-Out (LOOCV) training strategy
    """
    def __init__(self, seizure_indices=None, num_iter=1, augment=True,
                 use_ictal=True, balance_classes=True, patient_id=None,
                 patient_name=None, num_channels=1, preictal_interval=15,
                 preictal_step=1):
        self.augment = augment
        self.use_ictal = use_ictal
        self.balance_classes = balance_classes
        self.data_path = get_data_path()
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.num_channels = num_channels
        self.preictal_interval = preictal_interval
        self.preictal_step = preictal_step

        data_samples = []
        label_samples = []

        # Load data for each seizure in the training set
        for sz_idx in seizure_indices:
            # Load preictal data
            preictal_data = np.load(
                f"{self.data_path}/{self.patient_name}/"
                f"{self.preictal_interval}min_{self.preictal_step}step_{self.num_channels}ch/"
                f"preictal{sz_idx}.npy"
            )

            # Load interictal data
            interictal_data = np.load(
                f"{self.data_path}/{self.patient_name}/"
                f"{self.preictal_interval}min_{self.preictal_step}step_{self.num_channels}ch/"
                f"interictal{sz_idx}.npy"
            )

            # Load ictal data
            ictal_data = np.load(
                f"{self.data_path}/{self.patient_name}/"
                f"{self.preictal_interval}min_{self.preictal_step}step_{self.num_channels}ch/"
                f"ictal{sz_idx}.npy"
            )

            # Transpose if needed
            if len(preictal_data.shape) == 3:
                preictal_data = preictal_data.transpose(0, 2, 1)
                ictal_data = ictal_data.transpose(0, 2, 1)
                interictal_data = interictal_data.transpose(0, 2, 1)

            # Include ictal data in preictal if specified
            if self.use_ictal:
                print("Including ictal data in training")
                preictal_data = np.concatenate((preictal_data, ictal_data), axis=0)

            # Apply data augmentation
            if self.augment:
                print("Applying data augmentation")
                augmented_samples = []
                indices = np.arange(len(preictal_data) * 2)

                for _ in range(num_iter):
                    split_data = np.split(preictal_data, 2, axis=-1)
                    concatenated = np.concatenate(split_data, axis=0)
                    np.random.shuffle(indices)
                    split_indices = np.split(indices, 2)
                    shuffled = concatenated[split_indices[0]]
                    recombined = np.concatenate(np.split(shuffled, 2, axis=0), axis=-1)
                    augmented_samples.append(recombined)

                augmented_samples.append(preictal_data)
                preictal_data = np.concatenate(augmented_samples, axis=0)

            data_samples.append(preictal_data)

            # Balance interictal and preictal classes
            idx = np.arange(len(interictal_data))
            np.random.shuffle(idx)

            if self.balance_classes:
                interictal_to_use = interictal_data[
                    idx[:int(self.balance_classes * len(preictal_data))]
                ]
                data_samples.append(interictal_to_use)
                label_samples.append(np.ones((preictal_data.shape[0], 1)))
                label_samples.append(np.zeros((interictal_to_use.shape[0], 1)))
            else:
                data_samples.append(interictal_data)
                label_samples.append(np.ones((preictal_data.shape[0], 1)))
                label_samples.append(np.zeros((interictal_data.shape[0], 1)))

            print(f"Seizure {sz_idx}: Preictal {preictal_data.shape} | "
                  f"Ictal {ictal_data.shape} | Interictal {interictal_data.shape}")

        # Convert to numpy arrays
        data_samples = np.array(data_samples, dtype="object")
        label_samples = np.array(label_samples, dtype="object")
        data = np.concatenate(data_samples, axis=0)
        labels = np.concatenate(label_samples, axis=0)

        # Add channel dimension and convert to tensors
        if len(preictal_data.shape) == 3:
            data = data[:, np.newaxis, :, :].astype('float32')
        elif len(preictal_data.shape) == 4:
            data = data[:, np.newaxis, :, :, :].astype('float32')

        labels = labels.astype('int64')
        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(labels)
        self.length = data.shape[0]

        print(f"Training dataset: {self.x_data.shape}, {self.y_data.shape}")
        print(f"Preictal samples: {np.sum(labels == 1)}, "
              f"Interictal samples: {np.sum(labels == 0)}")

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length
