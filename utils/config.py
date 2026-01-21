# -*- coding: utf-8 -*-
"""
Configuration and utility functions for MS-SSA-HGCN
Author: QinYongKang
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.network import MS_SSA_HGCN
from models.loss import CrossEntropyLoss, FocalLoss


def get_patient_names():
    """
    Get patient ID to name mapping for CHB-MIT dataset
    """
    return {
        '1': 'chb01', '2': 'chb02', '3': 'chb03', '5': 'chb05',
        '6': 'chb06', '7': 'chb07', '8': 'chb08', '9': 'chb09',
        '10': 'chb10', '11': 'chb11', '13': 'chb13', '14': 'chb14',
        '16': 'chb16', '17': 'chb17', '18': 'chb18', '20': 'chb20',
        '21': 'chb21', '22': 'chb22', '23': 'chb23'
    }


def get_seizure_list():
    """
    Get seizure indices for each patient (0-indexed)
    """
    return {
        '1': [0, 1, 2, 3, 4, 5, 6],
        '2': [0, 1, 2],
        '3': [0, 1, 2, 3, 4, 5],
        '5': [0, 1, 2, 3, 4],
        '6': [0, 1, 2, 3, 4, 5, 6],
        '7': [0, 1, 2],
        '8': [0, 1, 2, 3, 4],
        '9': [0, 1, 2, 3],
        '10': [0, 1, 2, 3, 4, 5],
        '11': [0, 1, 2],
        '13': [0, 1, 2, 3, 4],
        '14': [0, 1, 2, 3, 4, 5],
        '16': [0, 1, 2, 3, 4, 5, 6, 7],
        '17': [0, 1, 2],
        '18': [0, 1, 2, 3, 4, 5],
        '20': [0, 1, 2, 3, 4, 5, 6, 7],
        '21': [0, 1, 2, 3],
        '22': [0, 1, 2],
        '23': [0, 1, 2, 3, 4, 5, 6],
    }


def get_data_path():
    """
    Get path to processed CHB-MIT data
    """
    return "data/processed/chb"


def get_num_channels(patient_id, requested_channels):
    """
    Get the number of input channels for the model

    Args:
        patient_id: Patient ID (not used, kept for compatibility)
        requested_channels: Requested number of channels

    Returns:
        Number of channels (always 18 for CHB-MIT)
    """
    if requested_channels != 18:
        print("Warning: Using 18 channels for CHB-MIT")
    return 18


def get_batch_size(patient_id):
    """
    Get appropriate batch size for each patient
    """
    if patient_id in [20, 21]:
        return 200
    return 256


def get_model(input_channels, device_id, model_name, use_position=False):
    """
    Create model instance

    Args:
        input_channels: Number of EEG channels
        device_id: CUDA device ID
        model_name: Name of the model
        use_position: Whether to use position embedding

    Returns:
        model: PyTorch model
    """
    if model_name == "MS_SSA_HGCN":
        hidden_dim = 6
        model = MS_SSA_HGCN(
            feature_dim=96,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            device_id=device_id,
            use_position=use_position
        )
        print(f"Created MS-SSA-HGCN model with {input_channels} input channels")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def get_loss_function(loss_name):
    """
    Create loss function

    Args:
        loss_name: Name of the loss function ('CE' or 'FL')

    Returns:
        loss: Loss function
    """
    if loss_name == "CE":
        return CrossEntropyLoss()
    elif loss_name == "FL":
        return FocalLoss(gamma=2.0)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def create_directory(path):
    """
    Create directory if it doesn't exist
    """
    path = path.strip().rstrip("\\")
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
        return True
    else:
        print(f"Directory already exists: {path}")
        return False


def set_random_seed(seed):
    """
    Set random seed for reproducibility
    """
    import random
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print(f"Random seed set to: {seed}")
