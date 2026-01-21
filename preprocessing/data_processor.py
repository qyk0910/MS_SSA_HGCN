# -*- coding: utf-8 -*-
"""
CHB-MIT Data Processor
Author: QinYongKang
"""

import os
import glob
import numpy as np
import argparse
from .edf_reader import EdfFileReader


def set_random_seed(seed):
    """
    Set random seed for reproducibility
    """
    np.random.seed(seed)


def get_patient_names():
    """
    Get patient name mappings for CHB-MIT dataset
    """
    return {
        '1': 'chb01', '2': 'chb02', '3': 'chb03', '5': 'chb05',
        '6': 'chb06', '7': 'chb07', '8': 'chb08', '9': 'chb09',
        '10': 'chb10', '11': 'chb11', '13': 'chb13', '14': 'chb14',
        '16': 'chb16', '17': 'chb17', '18': 'chb18', '20': 'chb20',
        '21': 'chb21', '22': 'chb22', '23': 'chb23'
    }


def get_seizure_timestamps():
    """
    Get seizure start and end times (in seconds) for each patient
    Each patient has a list of (start, end) tuples
    """
    return {
        '1': [(2996, 3036), (1467, 1494), (1732, 1772), (1015, 1066),
               (1720, 1810), (327, 420), (1862, 1963)],
        '2': [(130, 212), (2972, 3053), (3369, 3378)],
        '3': [(731, 796), (432, 501), (2162, 2214), (1982, 2029),
              (2592, 2656), (1725, 1778)],
        '5': [(417, 532), (1086, 1196), (2317, 2413), (2451, 2571), (2348, 2465)],
        '6': [(1724, 1738), (7461, 7476), (13525, 13540), (6211, 6231),
              (12500, 12516), (7799, 7811), (9387, 9403)],
        '7': [(4920, 5006), (3285, 3381), (13688, 13831)],
        '8': [(2670, 2841), (2856, 3046), (2988, 3122), (2417, 2577), (2083, 2347)],
        '9': [(12231, 12295), (2951, 3030), (9196, 9267), (5299, 5361)],
        '10': [(6888, 6958), (2382, 2447), (3021, 3079), (3801, 3877),
               (4618, 4707), (1383, 1437)],
        '11': [(298, 320), (2695, 2727), (1454, 2206)],
        '13': [(2077, 2121), (934, 1004), (2474, 2491), (3339, 3401), (851, 916)],
        '14': [(1986, 2000), (1372, 1392), (1911, 1925), (1838, 1879),
               (3239, 3259), (2833, 2849)],
        '16': [(2290, 2299), (1120, 1129), (1214, 1220), (227, 236),
               (1694, 1700), (3290, 3298), (627, 635), (1909, 1916)],
        '17': [(2282, 2372), (3025, 3140), (3136, 3224)],
        '18': [(3477, 3527), (541, 571), (2087, 2155), (1908, 1963),
               (2196, 2264), (463, 509)],
        '20': [(94, 123), (1440, 1470), (2498, 2537), (1971, 2009),
               (390, 425), (1689, 1738), (2226, 2261), (1393, 1432)],
        '21': [(1288, 1344), (2627, 2677), (2003, 2084), (2553, 2565)],
        '22': [(3367, 3425), (3139, 3213), (1263, 1335)],
        '23': [(3962, 4075), (325, 345), (5104, 5151), (2589, 2660),
               (6885, 6947), (8505, 8532), (9580, 9664)]
    }


def create_directory(path):
    """
    Create directory if it doesn't exist
    """
    path = path.strip().rstrip('\\')
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")


def process_patient_data(patient_id, data_root, num_channels, apply_filter,
                       preictal_duration, window_size, preictal_step,
                       interictal_step, seed=20010910):
    """
    Process CHB-MIT patient data into preictal, ictal, and interictal clips

    Args:
        patient_id: Patient ID
        data_root: Root path to data
        num_channels: Number of EEG channels
        apply_filter: Whether to apply low-pass filter
        preictal_duration: Preictal interval in minutes
        window_size: Sliding window size in seconds
        preictal_step: Step size for preictal windows in seconds
        interictal_step: Step size for interictal windows in seconds
        seed: Random seed

    Returns:
        None (saves processed data to .npy files)
    """
    set_random_seed(seed)

    patient_map = get_patient_names()
    seizure_times = get_seizure_timestamps()

    if str(patient_id) not in patient_map:
        print(f"Patient ID {patient_id} not found")
        return

    patient_name = patient_map[str(patient_id)]
    patient_data = CHBDataProcessor(
        patient_id, data_root, num_channels, apply_filter, preictal_duration
    )

    seizure_list = seizure_times[str(patient_id)]
    print(f"\nProcessing patient: ID {patient_id} {patient_name}")
    print(f"Seizure timestamps: {seizure_list}")

    output_dir = f"{data_root}/{patient_name}/{preictal_duration}min_{preictal_step}step_{num_channels}ch"
    create_directory(output_dir)

    # Process ictal and preictal data
    print("Extracting ictal and preictal clips...")
    for idx, edf_file in enumerate(patient_data.get_seizure_files()):
        print(f"Loading: {edf_file.get_file_path()}")

        raw_data = edf_file.get_preprocessed_data()
        print(f"Data shape: {raw_data.shape}")

        preictal_length = preictal_duration * 60

        # Check if we need to supplement preictal data
        if seizure_times[idx][0] < preictal_length:
            print(f"Seizure {idx+1}: Preictal period insufficient")

            supplement_path = f"{data_root}/{patient_name}/seizure-supplement/{edf_file.get_file_name()}-supplement.edf"

            if os.path.exists(supplement_path):
                print(f"Loading supplement file: {supplement_path}")
                supplement_file = EdfFileReader(supplement_path, patient_id, num_channels)
                print(f"Original seizure time: {seizure_times[idx]}")

                seizure_times[idx] = (
                    seizure_times[idx][0] + supplement_file.get_duration(),
                    seizure_times[idx][1] + supplement_file.get_duration()
                )
                print(f"Adjusted seizure time: {seizure_times[idx]}")

                raw_data = np.concatenate([
                    supplement_file.get_preprocessed_data(),
                    raw_data
                ])
                print(f"Combined data shape: {raw_data.shape}")
            else:
                print("No supplement file available")
                preictal_length = seizure_times[idx][0]

        # Extract ictal clips
        ictal_clips = []
        ictal_count = 0
        while seizure_times[idx][0] + preictal_step * ictal_count + window_size <= seizure_times[idx][1]:
            start = seizure_times[idx][0] + preictal_step * ictal_count
            end = seizure_times[idx][0] + preictal_step * ictal_count + window_size
            clip = raw_data[start * 256: end * 256]
            ictal_clips.append(clip)
            ictal_count += 1

        ictal_clips = np.array(ictal_clips)
        save_path = f"{output_dir}/ictal{idx}.npy"
        print(f"Saving ictal clips: {save_path}, shape: {ictal_clips.shape}")
        np.save(save_path, ictal_clips)

        # Extract preictal clips
        preictal_clips = []
        preictal_count = 0
        while seizure_times[idx][0] + preictal_step * preictal_count + window_size - preictal_length <= seizure_times[idx][0]:
            start = seizure_times[idx][0] + preictal_step * preictal_count - preictal_length
            end = seizure_times[idx][0] + preictal_step * preictal_count + window_size - preictal_length
            clip = raw_data[start * 256: end * 256]
            preictal_clips.append(clip)
            preictal_count += 1

        preictal_clips = np.array(preictal_clips)
        save_path = f"{output_dir}/preictal{idx}.npy"
        print(f"Saving preictal clips: {save_path}, shape: {preictal_clips.shape}")
        np.save(save_path, preictal_clips)

    # Process interictal data
    print("Extracting interictal clips...")
    all_interictal = []

    for idx, edf_file in enumerate(patient_data.get_non_seizure_files()):
        print(f"Loading: {edf_file.get_file_path()}")
        raw_data = edf_file.get_preprocessed_data()
        print(f"Data shape: {raw_data.shape}")

        interictal_clips = []
        interictal_count = 0
        while interictal_step * interictal_count + window_size <= edf_file.get_duration():
            start = interictal_step * interictal_count
            end = interictal_step * interictal_count + window_size
            clip = raw_data[start * 256: end * 256]
            interictal_clips.append(clip)
            interictal_count += 1

        interictal_clips = np.array(interictal_clips)
        print(f"Interictal clips shape: {interictal_clips.shape}")

        if len(all_interictal) == 0:
            all_interictal = interictal_clips
        else:
            all_interictal = np.vstack((all_interictal, interictal_clips))

        print(f"Total interictal so far: {all_interictal.shape}")

    # Shuffle and split interictal data by number of seizures
    np.random.shuffle(all_interictal)
    count = 0
    interictal_per_seizure = len(all_interictal) // len(seizure_times)

    while (count + 1) * interictal_per_seizure <= len(all_interictal):
        segment = all_interictal[
            count * interictal_per_seizure: (count + 1) * interictal_per_seizure
        ]
        save_path = f"{output_dir}/interictal{count}.npy"
        print(f"Saving interictal segment {count}: {save_path}, shape: {segment.shape}")
        np.save(save_path, segment)
        count += 1


class CHBDataProcessor:
    """
    Processor for CHB-MIT patient data
    """
    def __init__(self, patient_id, data_path, num_channels, apply_filter,
                 preictal_interval):
        self.interictal_interval = 90
        self.preictal_interval = preictal_interval
        self.postictal_interval = 120
        self.patient_id = patient_id
        self.data_path = data_path
        self.num_channels = num_channels
        self.apply_filter = apply_filter
        self.patient_name = self.get_patient_name()

        seizure_pattern = f"{data_path}/{self.patient_name}/seizure/*.edf"
        non_seizure_pattern = f"{data_path}/{self.patient_name}/unseizure/*.edf"

        self.seizure_files = list(map(
            lambda f: EdfFileReader(f, patient_id, num_channels, apply_filter),
            sorted(glob.glob(seizure_pattern))
        ))

        self.non_seizure_files = list(map(
            lambda f: EdfFileReader(f, patient_id, num_channels, apply_filter),
            sorted(glob.glob(non_seizure_pattern))
        ))

    def get_patient_name(self):
        patient_map = get_patient_names()
        return patient_map[str(self.patient_id)]

    def get_seizure_files(self):
        return self._seizure_files

    def get_non_seizure_files(self):
        return self._non_seizure_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess CHB-MIT Dataset')
    parser.add_argument('--patient_id', type=int, default=1, help='Patient ID')
    parser.add_argument('--preictal_interval', type=int, default=15,
                       help='Preictal interval in minutes')
    parser.add_argument('--seed', type=int, default=20010910, help='Random seed (default: 20010910)')
    parser.add_argument('--ch_num', type=int, default=18, help='Number of channels')
    parser.add_argument('--sfreq', type=int, default=256, help='Sampling frequency')
    parser.add_argument('--window_length', type=int, default=5, help='Window length in seconds')
    parser.add_argument('--preictal_step', type=int, default=1, help='Preictal step in seconds')
    parser.add_argument('--interictal_step', type=int, default=1, help='Interictal step in seconds')
    parser.add_argument('--doing_lowpass_filter', type=bool, default=True,
                       help='Whether to apply low-pass filter')
    parser.add_argument('--data_path', type=str, default='data/processed/chb',
                       help='Data save path')

    args = parser.parse_args()

    process_patient_data(
        patient_id=args.patient_id,
        data_root=args.data_path,
        num_channels=args.ch_num,
        apply_filter=args.doing_lowpass_filter,
        preictal_duration=args.preictal_interval,
        window_size=args.window_length,
        preictal_step=args.preictal_step,
        interictal_step=args.interictal_step,
        seed=args.seed
    )
