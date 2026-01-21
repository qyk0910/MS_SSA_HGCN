# -*- coding: utf-8 -*-
"""
EDF File Reader Module
Author: QinYongKang
"""

import numpy as np
import mne


class EdfFileReader:
    """
    Reads and processes EDF files from CHB-MIT dataset
    """
    def __init__(self, file_path, patient_id=None, num_channels=1,
                 apply_filter=True, preload=False):
        self.file_path = file_path
        self.patient_id = patient_id
        self.num_channels = num_channels
        self.apply_filter = apply_filter
        self.raw_data = mne.io.read_raw_edf(file_path, preload=preload)
        self.info = self.raw_data.info

    def get_file_path(self):
        return self.file_path

    def get_file_name(self):
        return self.file_path.split("/")[-1].split(".")[0]

    def get_num_channels(self):
        return self.info['nchan']

    def get_num_samples(self):
        return len(self.raw_data._times)

    def get_channel_names(self):
        return self.info['ch_names']

    def get_duration(self):
        return int(round(self.raw_data._last_time))

    def get_sampling_rate(self):
        sr = self.info['sfreq']
        if sr < 1:
            raise ValueError("Sampling frequency is less than 1")
        return int(sr)

    def get_preprocessed_data(self):
        """
        Load and preprocess EDF data
        """
        sampling_rate = self.get_sampling_rate()
        self.raw_data.load_data()
        self.raw_data.pick_channels(self.get_selected_channels())

        if self.apply_filter:
            self.raw_data.filter(0, 64)
            self.raw_data.notch_filter(
                np.arange(60, int((sampling_rate / 2) // 60 * 60 + 1), 60)
            )

        if sampling_rate > 256:
            self.raw_data.resample(256)

        data = self.raw_data.get_data().transpose(1, 0)
        return data

    def get_selected_channels(self):
        """
        Get the common 18 channels used for CHB-MIT analysis
        """
        selected_channels = []

        if self.patient_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 20, 21, 22, 23]:
            selected_channels = [
                'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3',
                'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4',
                'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2'
            ]
        elif self.patient_id in [13, 16, 17, 18, 19]:
            if self.get_num_channels() == 28:
                selected_channels = [
                    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3',
                    'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4',
                    'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2'
                ]
            else:
                selected_channels = [
                    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3',
                    'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4',
                    'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2'
                ]

        return selected_channels
