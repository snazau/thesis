from functools import partial
import json
import os
import pickle

import mne
import numpy as np
import torch

import datasets
import eeg_reader


def calc_stats_v2(subject_eeg_path, baseline_length_in_seconds=500):
    raw_data = eeg_reader.EEGReader.read_eeg(subject_eeg_path)
    if 'data1' in subject_eeg_path:
        channels_to_drop = ['EEG ECG', 'EEG MKR+ MKR-', 'EEG Fpz', 'EEG EMG']
    elif 'data2' in subject_eeg_path:
        channels_to_drop = ['EEG ECG', 'Value MKR+', 'EEG Fpz', 'EEG EMG']
    else:
        raise NotImplementedError
    channels_num = len(raw_data.info['ch_names'])
    channels_to_drop = channels_to_drop[:2 + (channels_num - 27)]
    raw_data.drop_channels(channels_to_drop)

    min_time, max_time = raw_data.times.min(), raw_data.times.max()

    # sample_start_times = np.arange(min_time, max_time - baseline_length_in_seconds, baseline_length_in_seconds // 2)
    sample_start_times = np.arange(min_time, max_time - baseline_length_in_seconds, max(baseline_length_in_seconds, max_time // baseline_length_in_seconds))
    samples, _, _ = datasets.generate_raw_samples(
        raw_data,
        sample_start_times,
        baseline_length_in_seconds
    )

    samples_std = np.std(samples, axis=2)  # std over time
    samples_avg_std = np.mean(samples_std, axis=1)  # mean over channels
    baseline_idx = np.argmin(samples_avg_std)
    baseline_segment = samples[baseline_idx:baseline_idx + 1]

    freqs = np.arange(1, 40.01, 0.1)
    baseline_power_spectrum = mne.time_frequency.tfr_array_morlet(
        baseline_segment,
        sfreq=128,
        freqs=freqs,
        n_cycles=freqs,
        output='power',
        n_jobs=-1
    )
    baseline_power_spectrum_log = np.log(baseline_power_spectrum)
    baseline_mean = np.squeeze(np.mean(baseline_power_spectrum_log[0], axis=2, keepdims=True))
    baseline_std = np.squeeze(np.std(baseline_power_spectrum_log[0], axis=2, keepdims=True))

    return baseline_mean, baseline_std


def calc_stats(subject_eeg_path, subject_seizures):
    subject_dataset = datasets.SubjectRandomDataset(
        subject_eeg_path,
        subject_seizures,
        samples_num=100,
        sample_duration=10,
        normal_samples_fraction=1.0,
        data_type='raw',
        baseline_correction=False,
        normalization=None,
    )

    collate_fn = partial(
        datasets.custom_collate_function,
        data_type='power_spectrum',
        baseline_correction=False,
        log=True,
        normalization=None,
        transform=None
    )
    loader = torch.utils.data.DataLoader(subject_dataset, batch_size=16, collate_fn=collate_fn)

    global_mean_stats = torch.zeros((1, 25, 391, 1), dtype=torch.float32)
    global_std_stats = torch.zeros((1, 25, 391, 1), dtype=torch.float32)
    for batch_idx, batch in enumerate(loader):
        batch_data = batch['data']

        batch_std_stats = torch.std(batch_data, dim=(0, 3), keepdim=True)
        batch_mean_stats = torch.mean(batch_data, dim=(0, 3), keepdim=True)

        global_std_stats += batch_std_stats
        global_mean_stats += batch_mean_stats
        # break
    global_std_stats = torch.squeeze(global_std_stats / len(loader)).detach().cpu().numpy()
    global_mean_stats = torch.squeeze(global_mean_stats / len(loader)).detach().cpu().numpy()

    return global_mean_stats, global_std_stats


if __name__ == '__main__':
    dataset_info_path = r'D:\Study\asp\thesis\implementation\data\dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    data_dir = r'D:\Study\asp\thesis\implementation\data'
    # save_dir = os.path.join(data_dir, 'cwt_log_stats')
    save_dir = os.path.join(data_dir, 'cwt_log_stats_v2')
    os.makedirs(save_dir, exist_ok=True)
    for subject_idx, subject_key in enumerate(dataset_info['subjects_info'].keys()):
        print(f'\rProgress {subject_idx + 1}/{len(dataset_info["subjects_info"].keys())} {subject_key}', end='')

        recording_duration = dataset_info['subjects_info'][subject_key]['duration_in_seconds']
        subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
        subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

        try:
            # subject_mean_stats, subject_std_stats = calc_stats(subject_eeg_path, subject_seizures)
            subject_mean_stats, subject_std_stats = calc_stats_v2(subject_eeg_path)
            subject_stats = {
                'mean': subject_mean_stats,
                'std': subject_std_stats,
            }

            save_path = os.path.join(save_dir, f'{subject_key}.npy')
            save_dataset_dir = os.path.dirname(save_path)
            os.makedirs(save_dataset_dir, exist_ok=True)

            with open(save_path, 'wb') as handle:
                pickle.dump(subject_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f'Something went wrong {subject_key}')
            print(e)
            print('\n\n\n')
            raise e
        # break
