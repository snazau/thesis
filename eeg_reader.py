import os

import mne
import numpy as np


class EEGReader:
    supported_extensions = ['.dat', '.edf', '.fif']

    @staticmethod
    def read_eeg(file_path, preload=False):
        file_name, file_ext = os.path.splitext(file_path)

        assert os.path.exists(file_path), file_path
        assert file_ext in EEGReader.supported_extensions

        if file_ext == '.dat':
            mne_raw = EEGReader.read_dat(file_path)
        elif file_ext == '.edf':
            mne_raw = EEGReader.read_edf(file_path, preload)
        elif file_ext == '.fif':
            mne_raw = EEGReader.read_fif(file_path, preload)
        else:
            raise NotImplementedError
        return mne_raw

    @staticmethod
    def read_dat(file_path):
        raw_data = np.loadtxt(file_path)
        # print(raw_data.min(), raw_data.mean(), raw_data.max())
        raw_data = raw_data.T * 1e-6
        # raw_data = raw_data.T

        channel_num = raw_data.shape[0]
        channel_names = [
            'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5',
            'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'F9', 'T9', 'ECG', 'P9', 'F10', 'T10', 'P10', 'MKR+ MKR-',
        ]
        channel_names = [f'EEG {channel_name}' for channel_name in channel_names]

        if channel_num == 27:
            del channel_names[1]
        elif channel_num == 28:
            pass
        else:
            raise NotImplementedError

        # info = mne.create_info(ch_names=channel_names, ch_types='misc', sfreq=128)
        info = mne.create_info(ch_names=channel_names, ch_types='eeg', sfreq=128)
        mne_raw = mne.io.RawArray(raw_data, info)
        return mne_raw

    @staticmethod
    def read_edf(file_path, preload=False):
        mne_raw = mne.io.read_raw_edf(file_path, preload=preload)
        return mne_raw

    @staticmethod
    def read_fif(file_path, preload=False):
        mne_raw = mne.io.read_raw_fif(file_path, preload=preload)
        return mne_raw
