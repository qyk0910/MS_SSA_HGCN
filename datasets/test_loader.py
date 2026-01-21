# -*- coding: utf-8 -*-
"""
Test Dataset Loader for CHB-MIT
Author: QinYongKang
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_data_path


class CHBTestDataset(Dataset):
    """
    Dataset loader for testing CHB-MIT data
    """
    def __init__(self, seizure_index=0, use_ictal=True, patient_id=None,
                 patient_name=None, num_channels=1, preictal_interval=15,
                 preictal_step=1):
        self.data_path = get_data_path()
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.num_channels = num_channels
        self.preictal_interval = preictal_interval
        self.preictal_step = preictal_step
        self.use_ictal = use_ictal

        # Load data
        preictal_data = np.load(
            f"{self.data_path}/{self.patient_name}/"
            f"{self.preictal_interval}min_{self.preictal_step}step_{self.num_channels}ch/"
            f"preictal{seizure_index}.npy"
        )

        interictal_data = np.load(
            f"{self.data_path}/{self.patient_name}/"
            f"{self.preictal_interval}min_{self.preictal_step}step_{self.num_channels}ch/"
            f"interictal{seizure_index}.npy"
        )

        ictal_data = np.load(
            f"{self.data_path}/{self.patient_name}/"
            f"{self.preictal_interval}min_{self.preictal_step}step_{self.num_channels}ch/"
            f"ictal{seizure_index}.npy"
        )

        # Record lengths
        self.preictal_length = preictal_data.shape[0]
        self.interictal_length = interictal_data.shape[0]
        self.ictal_length = ictal_data.shape[0]

        print(f"Interictal: {self.interictal_length} | "
              f"Preictal: {self.preictal_length} | Ictal: {self.ictal_length}")

        # Transpose if needed
        if len(preictal_data.shape) == 3:
            preictal_data = preictal_data.transpose(0, 2, 1)
            ictal_data = ictal_data.transpose(0, 2, 1)
            interictal_data = interictal_data.transpose(0, 2, 1)

        # Include ictal data in testing
        if self.use_ictal and ictal_data.shape[0] != 0:
            preictal_data = np.concatenate((preictal_data, ictal_data), axis=0)

        # Concatenate data and labels
        data_samples = [interictal_data, preictal_data]
        label_samples = [
            np.zeros((interictal_data.shape[0], 1)),
            np.ones((preictal_data.shape[0], 1))
        ]

        # Convert to arrays
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

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length
